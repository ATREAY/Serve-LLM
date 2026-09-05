# ServeLLM

**A multi-tenant LLM serving platform built on vLLM** — OpenAI-compatible API, dynamic
LoRA adapter loading, priority-based intelligent routing, measured continuous batching,
GPU telemetry, and full Prometheus/Grafana observability, deployed and load-tested on
real GPU hardware.

Runs entirely on open-weight models (TinyLlama, Qwen2.5-Coder, Qwen3). **No OpenAI,
Anthropic, Gemini, or other paid API is called anywhere in this stack** —
"OpenAI-compatible" here means only that the HTTP request/response shape matches, so
standard clients (`openai-python`, LangChain, curl) work unmodified against a
self-hosted model served through vLLM's `AsyncLLMEngine`.

## Highlights

- **7.7-7.9x throughput improvement from continuous batching**, measured directly
  (not quoted from vLLM's own docs): 16 concurrent chat requests against the same
  model/hardware, batching on vs. forced serialization —
  1228.7 vs. 155.5 tokens/sec, 1.84s vs. 14.18s wall-clock, 8.68 vs. 1.13 req/s.
  Consistent across all three metrics.
- **Priority scheduling verified under real contention**: 24 concurrent requests
  (12 high / 12 low priority) — all 12 high-priority requests completed before any
  low-priority one did (1.15s vs. 2.34s avg latency), confirmed against vLLM's actual
  scheduler source before wiring it up (priority is silently ignored unless the
  engine's scheduling policy is explicitly set to `"priority"`).
- **Dynamic LoRA adapters**, hot-loaded via API against a running server (no restart,
  no redeploy) — `POST /v1/admin/adapters` registers a new adapter, `model:
  "general:tarot"` in the next request routes to it immediately.
- **Direct GPU telemetry** via NVML (`pynvml`), not just process-level metrics — live
  utilization/memory exposed at `/v1/admin/gpu` and in an 11-panel Grafana dashboard
  (latency percentiles, TTFT, tokens/sec, per-model/adapter usage).
- **Security**: JWT-based short-lived tokens traded for a static API key, sliding-window
  rate limiting (429 + `Retry-After`), full request history in `GET /v1/admin/requests`.
- **A real, schema-validated Kubernetes/Helm chart** (`helm lint` + `helm template` +
  `kubeconform`, plus the rendered `ConfigMap` round-tripped through the actual
  application parser) — verified without a live cluster available in this environment.
- **One hardware limitation found, root-caused, and documented rather than hidden**:
  enabling vLLM's prefix caching hard-crashes the server process on this deployment's
  Volta (V100) GPU — a Triton kernel assertion that specifically requires Ampere
  (compute capability ≥ 8.0). Confirmed via the actual crash trace, confirmed no
  workaround exists on this hardware, documented as a known limitation instead of
  silently disabled or glossed over. See [docs/ROADMAP.md](docs/ROADMAP.md#phase-8--attempted-blocked-by-a-real-hardware-incompatibility).

## Architecture

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
                 │  backend/router         │  model selection by capability
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │ backend/scheduler       │  priority queueing / batching
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │ backend/lora            │  adapter registry, load/unload
                 └────────────┬────────────┘
                              │
                 ┌────────────────────────┐
                 │ backend/vllm_engine     │  AsyncLLMEngine wrapper
                 └────────────┬────────────┘
                              │
                        vLLM (PagedAttention, continuous batching)
                              │
                         GPU (CUDA)

  backend/metrics ─→ Prometheus ─→ Grafana   (cross-cutting)
  backend/database ─→ SQLite (default) / Postgres (docker-compose)
```

Each `backend/*` module is a separate package by design, so routing, scheduling, and
LoRA management sit inside the request path rather than in front of vLLM's own
bundled OpenAI server — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#why-asyncllmengine-directly-not-vllms-own-openai-server)
for why that specific tradeoff was made.

## Tech stack

Python · FastAPI · vLLM (`AsyncLLMEngine`, PagedAttention) · PyTorch · SQLAlchemy ·
SQLite/PostgreSQL · Redis · Prometheus · Grafana · NVML (`pynvml`) · JWT (`PyJWT`) ·
Docker Compose · Kubernetes/Helm · SLURM (GPU scheduling on the dev cluster).

## Quickstart (GPU node via SLURM)

```bash
conda create -n servellm python=3.10 -y
conda activate servellm
pip install -r backend/requirements.txt   # exact-pinned versions — see file comments

cp .env.example .env

# pre-download weights (compute nodes here have no internet egress)
python3 -c "
from huggingface_hub import snapshot_download as s
s('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
s('Qwen/Qwen2.5-Coder-1.5B-Instruct')
"

sbatch scripts/sbatch_serve.sh
tail -f servellm-<jobid>.log   # wait for "model ready"

python scripts/dev_client_smoke_test.py --base-url http://<node>:18742
```

## Quickstart (Docker, on a machine with an NVIDIA runtime)

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

Brings up: backend, Postgres, Redis, Prometheus, Grafana.

Full command reference for every feature (Grafana without Docker, Helm chart
validation without a live cluster, dynamic LoRA, priority scheduling, batching
benchmarks, security) is in [TESTING.md](TESTING.md).

## Notable engineering decisions

- **Owning `AsyncLLMEngine` directly instead of wrapping vLLM's own OpenAI server** —
  necessary once multi-model routing, per-adapter accounting, custom scheduling, and
  request logging all need to live inside the request path rather than in front of a
  black box.
- **Exact-pinned dependency versions**, not ranges: an unpinned `vllm>=0.6.0` resolved
  to a CUDA-13-only release with no compatible driver on this cluster; `transformers`
  similarly needed pinning below its next major version, which broke vLLM's tokenizer
  loading path. Both found by an actual failed deploy, not anticipated in advance.
- **`float16`, not `auto`/`bfloat16`, for `SERVELLM_DTYPE`** — `auto` resolves to
  `bfloat16` for these models, which vLLM hard-rejects below compute capability 8.0;
  this cluster's working node is a V100 (7.0).
- **Prometheus/Grafana as plain background processes, not Docker containers** — this
  cluster's login node has no container runtime, so both observability tools run as
  self-contained static binaries instead, with a custom port range to avoid colliding
  with other users' existing instances on shared hardware.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phase-by-phase build log — what
was implemented, what broke, and how each issue was actually diagnosed and fixed —
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design rationale.

## License

[MIT](LICENSE)
