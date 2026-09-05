"""Thin async wrapper around vLLM's AsyncLLMEngine.

Kept separate from the FastAPI layer (backend/gateway) so that later phases
(router, scheduler, LoRA manager) can hold references to one or more engines
without depending on HTTP concerns. generate() takes an optional LoRARequest
(Phase 3) — the base model itself doesn't change, vLLM swaps the adapter
weights in per-request when one is passed. It also takes an optional
priority (Phase 6) forwarded straight to vLLM's own scheduler — lower value
= served first, matching vllm.core.scheduler._get_priority's own convention
(sorts by (priority, arrival_time) ascending), not something we invert here.
enable_prefix_caching (Phase 8) is engine-wide, not per-request — vLLM
reuses KV cache blocks for any shared prompt prefix across requests once on,
no request-level opt-in.
"""

import time
from collections.abc import AsyncGenerator

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest


class VLLMEngineWrapper:
    """Owns exactly one vLLM AsyncLLMEngine instance for one base model.

    Takes explicit params rather than the global Settings object because
    Phase 2 constructs one of these per configured model, each with its own
    gpu_memory_utilization slice of the same physical GPU.
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        max_loras: int = 1,
        max_num_seqs: int | None = None,
        scheduling_policy: str = "fcfs",
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
    ):
        self._model_name = model_name
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._tensor_parallel_size = tensor_parallel_size
        self._enable_lora = enable_lora
        self._max_lora_rank = max_lora_rank
        self._max_loras = max_loras
        self._max_num_seqs = max_num_seqs
        self._scheduling_policy = scheduling_policy
        self._enable_prefix_caching = enable_prefix_caching
        self._enforce_eager = enforce_eager
        self._engine: AsyncLLMEngine | None = None

    async def start(self) -> None:
        args = AsyncEngineArgs(
            model=self._model_name,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            tensor_parallel_size=self._tensor_parallel_size,
            enable_lora=self._enable_lora,
            max_lora_rank=self._max_lora_rank if self._enable_lora else None,
            max_loras=self._max_loras if self._enable_lora else 1,
            max_num_seqs=self._max_num_seqs if self._max_num_seqs is not None else 256,
            scheduling_policy=self._scheduling_policy,
            enable_prefix_caching=self._enable_prefix_caching,
            enforce_eager=self._enforce_eager,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)

    async def stop(self) -> None:
        # AsyncLLMEngine has no explicit teardown; drop the reference so the
        # background loop and GPU memory can be reclaimed by the process exit.
        self._engine = None

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        request_id: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        stop: list[str] | None,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[tuple[str, str | None, int, int], None]:
        """Yields (accumulated_text, finish_reason, prompt_tokens, completion_tokens)
        once per new token, for both streaming and non-streaming callers."""
        if self._engine is None:
            raise RuntimeError("engine not started")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        results_generator = self._engine.generate(
            prompt, sampling_params, request_id, lora_request=lora_request, priority=priority
        )
        async for request_output in results_generator:
            output = request_output.outputs[0]
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = len(output.token_ids)
            yield output.text, output.finish_reason, prompt_tokens, completion_tokens

    async def abort(self, request_id: str) -> None:
        if self._engine is not None:
            await self._engine.abort(request_id)


def new_request_id() -> str:
    return f"servellm-{int(time.time() * 1e6)}"
