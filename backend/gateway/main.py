"""ServeLLM gateway — Phase 13: OpenAI-compatible serving on vLLM, routed
across multiple models by served_model_id or capability tag, with static
(Phase 3) and dynamic (Phase 4) LoRA adapters both addressed as
"<served_model_id>:<adapter_name>". Dynamic adapters are registered via
POST /v1/admin/adapters at runtime — no gateway restart required. GPU
utilization/memory are polled directly via NVML (Phase 5) and exposed
alongside the inference-level Prometheus metrics from Phase 1. Inference
endpoints are rate-limited (Phase 13) and every request is logged to
RequestLog (GET /v1/admin/requests), auth accepts a static API key or a
short-lived JWT issued from one (POST /v1/admin/auth/token).

Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /v1/admin/adapters,
/v1/admin/gpu, /v1/admin/requests, /v1/admin/auth/token, /healthz, /metrics.
Everything here runs against locally-hosted open-weight models through vLLM;
no calls to any paid third-party LLM API are made anywhere in this service.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from vllm.lora.request import LoRARequest

from backend.core.config import get_settings
from backend.core.schemas import (
    AdapterInfo,
    AdapterRegisterRequest,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    RequestLogEntry,
    TokenRequest,
    TokenResponse,
    UsageInfo,
)
from backend.database import adapter_catalog, request_log
from backend.database.session import get_session, init_db
from backend.gateway.auth import create_access_token, require_api_key
from backend.gateway.prompt import render_chat_prompt
from backend.gateway.rate_limit import enforce_rate_limit
from backend.lora.dynamic import run_idle_sweep
from backend.metrics.gpu import GpuMetricsPoller, run_gpu_poll_loop
from backend.metrics.middleware import (
    COMPLETION_TOKENS,
    INFERENCE_REQUESTS,
    INFLIGHT_REQUESTS,
    PROMPT_TOKENS,
    PrometheusMiddleware,
    TIME_TO_FIRST_TOKEN,
    TOKENS_PER_SECOND,
)
from backend.router.registry import ModelRegistry, build_registry
from backend.vllm_engine.engine import VLLMEngineWrapper, new_request_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("servellm.gateway")

registry: ModelRegistry | None = None
_sweep_task: asyncio.Task | None = None
_gpu_poll_task: asyncio.Task | None = None
_gpu_poller: GpuMetricsPoller | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, _sweep_task, _gpu_poll_task, _gpu_poller
    settings = get_settings()
    init_db(settings.database_url)
    registry = build_registry(settings)
    logger.info("starting models: %s", registry.list_ids())
    await registry.start()
    logger.info("all models ready")
    _sweep_task = asyncio.create_task(
        run_idle_sweep(
            registry.dynamic_cache,
            idle_seconds=settings.adapter_idle_ttl_seconds,
            interval_seconds=settings.adapter_sweep_interval_seconds,
        )
    )
    _gpu_poller = GpuMetricsPoller()
    _gpu_poller.start()
    _gpu_poll_task = asyncio.create_task(run_gpu_poll_loop(_gpu_poller))
    yield
    if _sweep_task is not None:
        _sweep_task.cancel()
    if _gpu_poll_task is not None:
        _gpu_poll_task.cancel()
    if _gpu_poller is not None:
        _gpu_poller.shutdown()
    if registry is not None:
        await registry.stop()


app = FastAPI(title="ServeLLM", version="0.13.0", lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)


def _stop_list(stop: list[str] | str | None) -> list[str] | None:
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else stop


def _observe_tokens_per_second(model: str, completion_tokens: int, elapsed_s: float) -> None:
    """Phase 11 uncovered that this histogram was defined (Phase 1) but never
    actually observed anywhere — would have been a permanently empty
    dashboard panel. elapsed_s guarded since a 0-token/instant response
    (e.g. max_tokens=0) would otherwise divide by ~0."""
    if completion_tokens > 0 and elapsed_s > 0:
        TOKENS_PER_SECOND.labels(model).observe(completion_tokens / elapsed_s)


def _split_model_adapter(model_id: str) -> tuple[str, str | None]:
    if ":" in model_id:
        base, adapter = model_id.split(":", 1)
        return base, adapter
    return model_id, None


def _log_request(
    *,
    request_id: str,
    model: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_seconds: float,
    ttft_seconds: float | None,
    status: str,
) -> None:
    """Phase 13 request history. Best-effort: a logging failure should never
    take down an otherwise-successful inference response."""
    base_model, adapter = _split_model_adapter(model)
    session = get_session()
    try:
        request_log.log_request(
            session,
            request_id=request_id,
            model=base_model,
            adapter=adapter,
            endpoint=endpoint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency_seconds,
            ttft_seconds=ttft_seconds,
            status=status,
        )
    except Exception:
        logger.exception("failed to write request log for request_id=%s", request_id)
    finally:
        session.close()


def _resolve_target(model_id: str) -> tuple[VLLMEngineWrapper, LoRARequest | None]:
    """Looks up model_id as an exact served_model_id, a capability tag, or a
    "<served_model_id>:<adapter_name>" LoRA route (Phase 3)."""
    if registry is None:
        raise HTTPException(503, "registry not ready")
    engine, lora_request = registry.resolve(model_id)
    if engine is None:
        raise HTTPException(
            404, f"unknown model '{model_id}', available: {registry.list_ids()}"
        )
    if not engine.ready:
        raise HTTPException(503, f"model '{model_id}' not ready")
    return engine, lora_request


@app.get("/healthz")
async def healthz():
    return {"status": "ok" if registry is not None and registry.ready else "starting"}


@app.get("/metrics")
async def metrics():
    return StreamingResponse(iter([generate_latest()]), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/admin/gpu", dependencies=[Depends(require_api_key)])
async def gpu_snapshot():
    """Human-readable current GPU utilization/memory (Phase 5) — the same
    numbers as the servellm_gpu_* series on /metrics, just without needing a
    Prometheus text-exposition parser for a quick check."""
    if _gpu_poller is None:
        raise HTTPException(503, "GPU poller not started")
    return {"gpus": _gpu_poller.snapshot()}


@app.post("/v1/admin/auth/token", response_model=TokenResponse)
async def issue_token(req: TokenRequest):
    """Phase 13: trades a static API key for a short-lived JWT. Deliberately
    not behind require_api_key — this endpoint IS how a caller gets a
    credential in the first place; it validates req.api_key directly."""
    settings = get_settings()
    allowed = settings.allowed_api_keys()
    if not allowed:
        raise HTTPException(400, "auth is disabled (SERVELLM_API_KEYS unset) — no token to issue")
    if req.api_key not in allowed:
        raise HTTPException(401, "invalid api key")
    token, expires_in = create_access_token(settings)
    return TokenResponse(access_token=token, expires_in=expires_in)


@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(require_api_key)])
async def list_models():
    if registry is None:
        return ModelList(data=[])
    return ModelList(data=[ModelCard(id=model_id) for model_id in registry.list_ids()])


def _adapter_info(row) -> AdapterInfo:
    return AdapterInfo(
        base_model=row.base_model,
        name=row.name,
        hf_repo=row.hf_repo,
        version=row.version,
        status=row.status,
        hits=row.hits,
        last_used_at=row.last_used_at.isoformat() if row.last_used_at else None,
        created_at=row.created_at.isoformat(),
    )


@app.post(
    "/v1/admin/adapters",
    response_model=AdapterInfo,
    dependencies=[Depends(require_api_key)],
)
async def register_adapter(req: AdapterRegisterRequest):
    """Phase 4: register a LoRA adapter for an already-running base model.
    No restart needed — the next request for "<base_model>:<name>" resolves
    and loads it lazily via registry.resolve() -> DynamicAdapterCache."""
    if registry is None or registry.get(req.base_model) is None:
        raise HTTPException(
            404,
            f"unknown base_model '{req.base_model}', available: "
            f"{registry.list_ids() if registry else []}",
        )
    session = get_session()
    try:
        row = adapter_catalog.register_adapter(
            session,
            base_model=req.base_model,
            name=req.name,
            hf_repo=req.hf_repo,
            version=req.version,
        )
        return _adapter_info(row)
    finally:
        session.close()


@app.get(
    "/v1/admin/adapters",
    response_model=list[AdapterInfo],
    dependencies=[Depends(require_api_key)],
)
async def list_adapters(base_model: str | None = None):
    session = get_session()
    try:
        rows = adapter_catalog.list_adapters(session, base_model=base_model)
        return [_adapter_info(r) for r in rows]
    finally:
        session.close()


@app.delete("/v1/admin/adapters/{base_model}/{name}", dependencies=[Depends(require_api_key)])
async def delete_adapter(base_model: str, name: str):
    session = get_session()
    try:
        deleted = adapter_catalog.delete_adapter(session, base_model=base_model, name=name)
    finally:
        session.close()
    if not deleted:
        raise HTTPException(404, f"no adapter '{name}' registered for base_model '{base_model}'")
    if registry is not None:
        registry.dynamic_cache.evict(base_model, name)
    return {"status": "deleted", "base_model": base_model, "name": name}


@app.get(
    "/v1/admin/requests",
    response_model=list[RequestLogEntry],
    dependencies=[Depends(require_api_key)],
)
async def request_history(model: str | None = None, status: str | None = None, limit: int = 50):
    """Phase 13: recent inference requests (RequestLog, scaffolded all the
    way back in Phase 1, wired in here). Newest first."""
    session = get_session()
    try:
        rows = request_log.list_requests(session, model=model, status=status, limit=min(limit, 500))
        return [
            RequestLogEntry(
                request_id=r.request_id,
                model=r.model,
                adapter=r.adapter,
                endpoint=r.endpoint,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                latency_seconds=r.latency_seconds,
                ttft_seconds=r.ttft_seconds,
                status=r.status,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
    finally:
        session.close()


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def chat_completions(req: ChatCompletionRequest):
    engine, lora_request = _resolve_target(req.model)
    prompt = render_chat_prompt(engine.model_name, req.messages)
    request_id = new_request_id()
    stop = _stop_list(req.stop)
    start = time.perf_counter()

    INFLIGHT_REQUESTS.labels(req.model).inc()
    try:
        if req.stream:
            return StreamingResponse(
                _stream_chat(engine, lora_request, request_id, prompt, req, stop),
                media_type="text/event-stream",
            )

        final_text, finish_reason, prompt_tokens, completion_tokens = "", None, 0, 0
        async for text, reason, ptok, ctok in engine.generate(
            prompt,
            request_id,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=stop,
            n=req.n,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
            lora_request=lora_request,
            priority=req.priority,
        ):
            final_text, finish_reason, prompt_tokens, completion_tokens = text, reason, ptok, ctok

        PROMPT_TOKENS.labels(req.model).inc(prompt_tokens)
        COMPLETION_TOKENS.labels(req.model).inc(completion_tokens)
        INFERENCE_REQUESTS.labels(req.model, "chat.completions", "ok").inc()
        elapsed = time.perf_counter() - start
        _observe_tokens_per_second(req.model, completion_tokens, elapsed)
        _log_request(
            request_id=request_id,
            model=req.model,
            endpoint="chat.completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=elapsed,
            ttft_seconds=None,
            status="ok",
        )

        return ChatCompletionResponse(
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=final_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception:
        INFERENCE_REQUESTS.labels(req.model, "chat.completions", "error").inc()
        _log_request(
            request_id=request_id,
            model=req.model,
            endpoint="chat.completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=time.perf_counter() - start,
            ttft_seconds=None,
            status="error",
        )
        raise
    finally:
        INFLIGHT_REQUESTS.labels(req.model).dec()


async def _stream_chat(
    engine: VLLMEngineWrapper,
    lora_request: LoRARequest | None,
    request_id,
    prompt,
    req: ChatCompletionRequest,
    stop,
):
    start = time.perf_counter()
    first_token_sent = False
    prev_text = ""
    completion_tokens = 0
    prompt_tokens = 0
    ttft: float | None = None

    yield _sse(
        ChatCompletionChunk(
            model=req.model,
            choices=[
                ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(role="assistant"))
            ],
        )
    )

    async for text, finish_reason, ptok, ctok in engine.generate(
        prompt,
        request_id,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=stop,
        n=req.n,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        lora_request=lora_request,
        priority=req.priority,
    ):
        completion_tokens = ctok
        prompt_tokens = ptok
        if not first_token_sent:
            ttft = time.perf_counter() - start
            TIME_TO_FIRST_TOKEN.labels(req.model).observe(ttft)
            first_token_sent = True

        delta = text[len(prev_text):]
        prev_text = text
        if delta:
            yield _sse(
                ChatCompletionChunk(
                    model=req.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0, delta=ChatCompletionChunkDelta(content=delta)
                        )
                    ],
                )
            )
        if finish_reason is not None:
            yield _sse(
                ChatCompletionChunk(
                    model=req.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0, delta=ChatCompletionChunkDelta(), finish_reason=finish_reason
                        )
                    ],
                )
            )

    INFERENCE_REQUESTS.labels(req.model, "chat.completions.stream", "ok").inc()
    _observe_tokens_per_second(req.model, completion_tokens, time.perf_counter() - start)
    _log_request(
        request_id=request_id,
        model=req.model,
        endpoint="chat.completions.stream",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_seconds=time.perf_counter() - start,
        ttft_seconds=ttft,
        status="ok",
    )
    yield "data: [DONE]\n\n"


def _sse(chunk: ChatCompletionChunk) -> str:
    return f"data: {json.dumps(chunk.model_dump())}\n\n"


@app.post(
    "/v1/completions",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def completions(req: CompletionRequest):
    engine, lora_request = _resolve_target(req.model)
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    request_id = new_request_id()
    stop = _stop_list(req.stop)
    start = time.perf_counter()

    INFLIGHT_REQUESTS.labels(req.model).inc()
    try:
        final_text, finish_reason, prompt_tokens, completion_tokens = "", None, 0, 0
        async for text, reason, ptok, ctok in engine.generate(
            prompt,
            request_id,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=stop,
            n=req.n,
            lora_request=lora_request,
            priority=req.priority,
        ):
            final_text, finish_reason, prompt_tokens, completion_tokens = text, reason, ptok, ctok

        PROMPT_TOKENS.labels(req.model).inc(prompt_tokens)
        COMPLETION_TOKENS.labels(req.model).inc(completion_tokens)
        INFERENCE_REQUESTS.labels(req.model, "completions", "ok").inc()
        elapsed = time.perf_counter() - start
        _observe_tokens_per_second(req.model, completion_tokens, elapsed)
        _log_request(
            request_id=request_id,
            model=req.model,
            endpoint="completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=elapsed,
            ttft_seconds=None,
            status="ok",
        )

        return CompletionResponse(
            model=req.model,
            choices=[CompletionChoice(index=0, text=final_text, finish_reason=finish_reason)],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception:
        INFERENCE_REQUESTS.labels(req.model, "completions", "error").inc()
        _log_request(
            request_id=request_id,
            model=req.model,
            endpoint="completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=time.perf_counter() - start,
            ttft_seconds=None,
            status="error",
        )
        raise
    finally:
        INFLIGHT_REQUESTS.labels(req.model).dec()
