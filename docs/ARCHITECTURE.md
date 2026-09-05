# ServeLLM Architecture

```
                    Client (OpenAI SDK / curl)
                              │
                      REST — /v1/*  (FastAPI)
                              │
                 ┌────────────────────────┐
                 │   backend/gateway       │  auth, schemas, SSE streaming
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │  backend/router (P2)    │  model selection by capability
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │ backend/scheduler (P6-7)│  queueing / batching experiments
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │ backend/lora (P3-4)     │  adapter registry, load/unload
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │ backend/vllm_engine     │  AsyncLLMEngine wrapper
                 └────────────┬────────────┘
                              │
                        vLLM (PagedAttention, continuous batching)
                              │
                         GPU (CUDA)

  backend/metrics ─→ Prometheus ─→ Grafana   (cross-cutting, wired from Phase 1)
  backend/database ─→ Postgres                (request log, adapter registry)
  Redis                                        (Phase 6 queue backend)
```

Each `backend/*` module is a separate Python package on purpose — mirrors how the
phases in `docs/ROADMAP.md` land as additions to specific modules rather than
rewrites of the gateway. `backend/vllm_engine` (not `backend/vllm`) is named to
avoid shadowing the actual `vllm` package on `sys.path`.

## Why AsyncLLMEngine directly, not vLLM's own OpenAI server

vLLM ships `vllm.entrypoints.openai.api_server`, which already does most of Phase 1.
Wrapping it as a subprocess would have been faster to stand up, but everything past
Phase 2 (routing across models, per-adapter accounting, custom scheduling, request
logging to Postgres) needs to sit *inside* the request path, not proxy in front of
a black box. Owning the `AsyncLLMEngine` directly is what makes Phases 2-8 additive
instead of a rewrite.
