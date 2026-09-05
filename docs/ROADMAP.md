# ServeLLM Roadmap

Status column reflects what actually exists in this repo, not aspiration.

| Phase | Scope | Status |
|---|---|---|
| 1 | Single-model OpenAI-compatible serving (chat, completions, streaming) on vLLM | **Done and verified end-to-end on GPU** (see below) |
| 2 | Multi-model routing by served_model_id / capability | **Done** (see below) |
| 3 | Static LoRA adapters | **Done** (see below) |
| 4 | Dynamic LoRA manager + adapter registry | **Done** (see below) |
| 5 | GPU memory manager / dashboard | **Done** (see below) |
| 6 | Priority-based scheduling | **Done** (see below) |
| 7 | Continuous batching comparison | **Done** (see below) |
| 8 | Prefix cache measurement | **Attempted — blocked by hardware** (see below) |
| 9 | Streaming | **Done** (Phase 1) |
| 10 | Metrics service (Prometheus counters/histograms) | **Done** (Phase 1, wired into gateway) |
| 11 | Grafana dashboards | **Done** (see below) |
| 12 | Benchmark suite | **Done** (see below) |
| 13 | Security (JWT, rate limiting, admin) | **Done** (see below) |
| 14 | Kubernetes/Helm | **Done — chart written, verified, not live-deployed** (see below) |

## Phase 1 — what's implemented

- `backend/vllm_engine/engine.py` — wraps `vllm.AsyncLLMEngine`, one engine per process.
- `backend/gateway/main.py` — FastAPI app: `/v1/chat/completions` (streaming + non-streaming
  via SSE), `/v1/completions`, `/v1/models`, `/healthz`, `/metrics`.
- `backend/gateway/prompt.py` — chat messages rendered through the model's own HF chat
  template, so any instruct model works without a hand-rolled prompt format.
- `backend/metrics/middleware.py` — HTTP + inference-level Prometheus metrics
  (TTFT, tokens/sec, in-flight gauge) from day one, not bolted on later.
- `backend/database/models.py` — `RequestLog` / `AdapterRegistry` schema defined ahead of
  Phase 4/6 so those phases are additive, not migrations.
- `docker/` — Dockerfile + compose (backend + Postgres + Redis + Prometheus + Grafana).
- `scripts/sbatch_serve.sh` — SLURM launch script, pinned to `dgx-v100-01`; see the
  comment block in that file for why (not `docs/` — it needs to stay next to the
  directive it explains). Short version: `dgx-a100-02` looked ideal on paper but had
  severe CPU/NFS contention from other users; the P100 nodes fail hard at inference
  time (`no kernel image is available for execution on the device` — this torch build's
  kernels don't target Pascal at all).

## Verified end-to-end (2026-08-25)

Ran on `dgx-v100-01` via `sbatch scripts/sbatch_serve.sh`: model loads, engine builds
(CUDA graphs captured, XFormers attention backend — no FlashAttention-2 on pre-Ampere),
`/healthz`, `/v1/models`, non-streaming `/v1/chat/completions`, and token-by-token SSE
streaming all confirmed working via `scripts/dev_client_smoke_test.py`.

Two real environment bugs found and fixed along the way, both now reflected in the
code/config rather than being one-off manual fixes:
- `dtype=auto` resolves to the model's declared `bfloat16`, which `AsyncLLMEngine`
  hard-rejects below compute capability 8.0 — `.env.example` now sets `float16` explicitly.
- Port 8000 was already bound by another user's process on the shared node — default
  port moved to `18742` in `.env.example`.

`backend/requirements.txt` also had to move off the naive `vllm>=0.6.0` pin: the
unconstrained resolver picks the latest vllm, which hard-pins `torch==<latest>+cu13`,
and no driver on this cluster (535.x / 570.x) supports CUDA 13. Pinned instead to
`vllm==0.6.3.post1` + `torch==2.4.0+cu121` + `transformers==4.45.2` (the newer
default `transformers>=5` refuses to bind to torch<2.5 and breaks vllm's tokenizer
code) — see the comments in `backend/requirements.txt` for the full chain.

## Phase 2 — what's implemented

- `backend/router/registry.py` — `ModelSpec` + `ModelRegistry`: one `VLLMEngineWrapper`
  per configured model, all sharing the one physical GPU. `build_registry()` loads
  `backend/router/models.yaml` if present, else falls back to a single-entry registry
  built from the Phase 1 `SERVELLM_MODEL_NAME` settings — existing single-model
  deployments don't need a models.yaml at all.
- `backend/router/models.yaml` — two models on `dgx-v100-01`: `general`
  (TinyLlama-1.1B) and `code` (Qwen2.5-Coder-1.5B-Instruct), each capped to
  `gpu_memory_utilization: 0.35` so both fit on one 32GB V100 alongside CUDA graph
  capture overhead. Bigger models (the originally planned Qwen3-8B/Gemma-3-4B) are a
  config change away once running somewhere with more headroom (A100, or only one
  resident model at a time).
- `backend/gateway/main.py` — `_resolve_target()` (superseded by Phase 3's version below) looked up `request.model` as an exact
  `served_model_id` first, then as a capability tag, then 404s with the list of what's
  actually available. `/v1/models` now lists every configured model, not a single id.
- `backend/vllm_engine/engine.py` — `VLLMEngineWrapper` now takes explicit
  constructor params instead of the global `Settings` object, so the registry can
  build one per model with independent config (this is the actual Phase 2 change to
  that file; Phase 1's behavior is unchanged when only one model is configured).

Startup loads models sequentially (not concurrently) — each engine claims its GPU
memory slice at construction time, so starting them one at a time avoids two engines
racing to read "free" GPU memory at once.

## Phase 3 — what's implemented

- `backend/lora/manager.py` — `AdapterEntry` + `resolve_adapter_requests()`: resolves
  each configured adapter to a local weight path (via `huggingface_hub.snapshot_download`,
  which no-ops to the already-cached path under `HF_HUB_OFFLINE=1`) and assigns it a
  unique `lora_int_id`, building the `vllm.lora.request.LoRARequest` each `generate()`
  call needs.
- `backend/router/registry.py` — `ModelSpec.adapters` declares static adapters per
  model; `ModelRegistry.resolve()` parses `"<served_model_id>:<adapter_name>"` (e.g.
  `general:colorist`), returning the base engine plus the matching `LoRARequest`, or
  `(None, None)` if either half doesn't exist. `list_ids()` includes the adapter routes
  so `/v1/models` advertises them alongside the base models.
- `backend/vllm_engine/engine.py` — `generate()` now accepts an optional `lora_request`
  and forwards it straight through to `AsyncLLMEngine.generate()`; the base model engine
  itself is unchanged, vLLM swaps the adapter weights in per-request.
- `backend/router/models.yaml` — `general` (TinyLlama) declares one adapter: `colorist`
  (`BOT365/tinyllama-colorist-lora`), vLLM's own canonical LoRA demo adapter, not
  something trained for this project — chosen because its `adapter_config.json`
  `base_model_name_or_path` is verified to exactly match `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

### Verified (2026-08-26)

`general:colorist` asked "What color is the sky at sunset?" answered *"At sunset, the
sky is a deep, rich shade of blue, almost like a dark navy blue. It's a vibrant and..."*
— coherent, on-topic, and visibly different from the base `general` model's generic
replies to the same kind of prompt on identical underlying weights. That's the adapter
actually being applied, not just the routing accepting the request. `/v1/models` lists
`general`, `code`, and `general:colorist`; requesting an unconfigured adapter route
(or unconfigured model) 404s with the full list of what's available.

## Phase 4 — what's implemented

- `backend/database/session.py` — sync SQLAlchemy engine/session, SQLite by default
  (`sqlite:///./servellm.db`) so the DB-backed registry needs zero extra infra on the
  GPU compute node; `DATABASE_URL` can point at real Postgres instead for the
  docker-compose deployment, same models either way.
- `backend/database/adapter_catalog.py` — CRUD over `AdapterRegistry`
  (`backend/database/models.py`, scaffolded back in Phase 1): register/get/list/delete,
  `touch_adapter()` (hits + last_used_at + status on every use), `find_idle_since()`
  for the sweep below. Unique on `(base_model, name)`, not just `name` — Phase 3's
  `AdapterRegistry.name` had a bare unique constraint; relaxed since the same adapter
  name could plausibly exist for two different base models.
- `backend/lora/dynamic.py` — `DynamicAdapterCache`: resolves a DB-registered adapter
  to a `LoRARequest` lazily on first use (downloads/finds weights via
  `snapshot_download`, same offline-cache pattern as every other model in this repo),
  assigns `lora_int_id` from a range starting at 1000 so it never collides with Phase
  3's static adapter ids. `run_idle_sweep()` is a background asyncio task (started in
  the gateway `lifespan`) that evicts adapters idle past `SERVELLM_ADAPTER_IDLE_TTL_SECONDS`
  from the in-memory cache and marks them idle in the DB — this does *not* force vLLM's
  own GPU-resident LoRA cache to evict anything; that's controlled by `max_loras` /
  `max_cpu_loras` on the engine itself, a separate, engine-internal LRU.
- `backend/router/registry.py` — `ModelSpec.max_dynamic_adapters` reserves vLLM LoRA
  slots at engine startup (fixed then, can't grow later) for adapters that don't exist
  yet; `resolve()` falls back from the static adapter dict to `DynamicAdapterCache`
  when a `"<base>:<adapter>"` route isn't statically declared.
- `backend/gateway/main.py` — `POST /v1/admin/adapters` (register, no restart needed),
  `GET /v1/admin/adapters` (list with hits/status/last_used_at), `DELETE
  /v1/admin/adapters/{base_model}/{name}` (unregister + immediate cache eviction,
  doesn't wait for the idle sweep). All behind the same `require_api_key` dependency
  as the inference endpoints.

### Verified (2026-08-26) — genuinely dynamic, not just DB-backed

Against the *already-running* server (job never restarted): registered a second
TinyLlama adapter, `barissglc/tinyllama-tarot-v1` (not declared in `models.yaml`,
verified via its own `adapter_config.json` to match the same base model as `colorist`)
via `POST /v1/admin/adapters`, then immediately called `general:tarot` — no reload, no
restart — and got a real, on-theme tarot reading: *"The tarot card for this reading is
the Hermit, which means you are in need of reflection and introspection..."* `/v1/models`
picked it up automatically once loaded. `DELETE /v1/admin/adapters/general/tarot`
evicted it immediately; the next `general:tarot` request correctly 404s listing what's
actually still available, and deleting an already-deleted adapter also 404s cleanly.

## Phase 5 — what's implemented

- `backend/metrics/gpu.py` — `GpuMetricsPoller`: queries NVML directly (`pynvml`, via
  the `nvidia-ml-py` package) for per-GPU utilization % and memory used/total, exposed
  as `servellm_gpu_utilization_percent` / `servellm_gpu_memory_used_bytes` /
  `servellm_gpu_memory_total_bytes` on `/metrics`, plus a human-readable
  `GET /v1/admin/gpu` snapshot. A background asyncio task polls every 5s (started/
  cancelled in the gateway `lifespan`, same pattern as the Phase 4 idle sweep).
- Only polls the GPU index(es) this job actually has — parses `CUDA_VISIBLE_DEVICES`
  rather than trusting NVML's default enumeration (NVML sees every physical GPU on
  the node regardless of what SLURM allocated; on a shared node with unrelated users'
  jobs on the other GPUs, blindly reading index 0 would silently report the wrong
  GPU's stats — confirmed correct against `scontrol show job -d` before trusting it).

### Why NVML instead of vLLM's own Prometheus metrics

vLLM's `AsyncLLMEngine` auto-registers its own KV-cache/request-count Prometheus
metrics (`vllm:*`) into the same process-global registry. That works cleanly with one
engine per process; with two co-resident engines (Phase 2's `general` + `code`), the
second engine's registration is silently incomplete — hitting `/metrics` showed only
one `vllm:cache_config_info` line for two running models, no crash or warning logged
either. Rather than patch vLLM's internal stat logger for a multi-engine-per-process
setup it wasn't obviously designed for, Phase 5 reports GPU-level telemetry directly
from NVML instead, which doesn't care how many engines share the device — and is
closer to what the original architecture's per-GPU dashboard mockup wanted anyway.

### Verified (2026-08-26)

`GET /v1/admin/gpu` against the live server reported 22,308 MiB used on GPU index 0;
a direct `nvidia-smi` query on the same node at nearly the same moment showed 22,577
MiB on GPU 0 (small gap is just measurement-instant drift, same physical GPU) — and
`scontrol show job -d <jobid>` confirmed `GRES=gpu:1(IDX:0)`, so the poller is
definitely reading this job's actual allocated GPU, not guessing. `/metrics` shows all
three `servellm_gpu_*` series correctly labeled by `gpu_index`.

## Phase 6 — what's implemented

- `backend/core/schemas.py` — `priority: int = 0` added to `ChatCompletionRequest` and
  `CompletionRequest`. A ServeLLM extension, not part of the OpenAI API; lower value =
  served first, matching vLLM's own scheduler convention directly (no inversion).
- `backend/vllm_engine/engine.py` — `VLLMEngineWrapper.generate()` forwards `priority`
  straight to `AsyncLLMEngine.generate()`; constructor gained `scheduling_policy`
  (`"fcfs"` default, matching vLLM) and `max_num_seqs`.
- `backend/router/registry.py` — `ModelSpec.scheduling_policy` / `.max_num_seqs`, so
  each model can opt into priority scheduling independently. Verified against
  `vllm/core/scheduler.py` before wiring anything up: priority-ordered sorting of the
  waiting queue only runs inside `_schedule_priority_preemption`, which only runs when
  `scheduler_config.policy == "priority"` — under the vLLM default (`"fcfs"`), a
  request's `priority` field is accepted but silently never consulted. `general` in
  `models.yaml` now sets `scheduling_policy: priority`; `code` stays on the default.
- `scripts/bench_priority.py` — fires N concurrent chat requests split high/low
  priority and reports completion order + latency per group, against a live running
  instance (not vLLM's scheduler in isolation).

### Verified (2026-08-26) — real contention, not a trivial pass

Priority has nothing to act on if every concurrent request fits in the running batch
immediately (vLLM's own `max_num_seqs` default is 256) — so the experiment temporarily
set `general`'s `max_num_seqs: 3` (redeployed, benchmarked, then reverted — see the
git-visible round-trip in `backend/router/models.yaml`'s history, or just its current
comment) and fired 24 concurrent requests (12 priority=0, 12 priority=100) via
`scripts/bench_priority.py --concurrency 24 --max-tokens 200`. Result: **all 12
high-priority requests completed before any low-priority request did** — completion
ranks 0-11 were entirely priority=0, ranks 12-23 entirely priority=100. Average
latency: 1.15s (high) vs 2.34s (low). Reproducible by re-running the same
constrain-benchmark-revert sequence.

## Phase 7 — what's implemented

- `scripts/bench_batching.py` — fires N concurrent chat requests against a live
  instance, reports wall-clock time, per-request latency, aggregate token throughput,
  and requests/sec. Reuses the `max_num_seqs` lever from Phase 6 rather than
  introducing a second serving framework (e.g. plain HF `transformers.generate()`) as
  the "without batching" baseline — vLLM has no separate batching on/off switch since
  continuous batching *is* its execution model, and comparing across frameworks would
  confound the result with differences in kernels/memory layout beyond just batching.
  `max_num_seqs: 1` forces true serialization on the exact same engine/model/hardware,
  which isolates the one variable actually being measured.

### Verified (2026-08-26) — real measured speedup, not a claim

16 concurrent chat requests, 150 max_tokens each, against `general` on `dgx-v100-01`,
same two-config round-trip pattern as Phase 6 (constrain → benchmark → revert →
benchmark again):

| | without batching (`max_num_seqs: 1`) | continuous batching (default: 256) |
|---|---|---|
| wall-clock time | 14.18s | 1.84s |
| aggregate throughput | 155.5 tokens/s | 1228.7 tokens/s |
| requests/s | 1.13 | 8.68 |
| avg per-request latency | 7.63s | 1.75s |

**~7.7-7.9x speedup**, consistent across all three independent measures (wall-clock,
throughput, requests/s) — not cherry-picked from one metric. This is the concrete,
measured version of "continuous batching is one of the major reasons vLLM is fast,"
on this project's actual hardware and models rather than taken on faith.

## Phase 8 — attempted, blocked by a real hardware incompatibility

- `backend/vllm_engine/engine.py` / `backend/router/registry.py` —
  `enable_prefix_caching` wired through exactly like every other engine-level flag
  (`ModelSpec` → `VLLMEngineWrapper` → `AsyncEngineArgs`). This part is correct and
  works — confirmed by the engine constructing and serving its first request
  successfully with the flag on (see the crash trace below: the *first* request
  succeeded normally).
- `scripts/bench_prefix_cache.py` — fires sequential (not concurrent — deliberately,
  to isolate the prefix-cache effect from Phase 7's continuous batching) streamed
  requests sharing a long common system-prompt-style prefix, measuring true TTFT (time
  to the first chunk with actual generated content, not the empty role-delta chunk the
  gateway always emits first).

### What actually happened (2026-08-26)

Baseline (`enable_prefix_caching: false`) ran cleanly: TTFT flat at ~0.05-0.06s across
8 sequential requests sharing the same prefix, as expected with no cache reuse.

Enabling it (`enable_prefix_caching: true`) and redeploying: **the first request
succeeded** (ttft=0.439s, includes cache population), but **the second request crashed
the entire server process** — not a caught Python exception, a hard process abort:

```
python3: /project/lib/Analysis/Allocation.cpp:47: ... mlir::triton::getCvtOrder(...):
Assertion `!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) &&
"mma -> mma layout conversion is only supported on Ampere"' failed.
srun: error: dgx-v100-01: task 0: Aborted (core dumped)
```

This is a Triton (the JIT kernel compiler vLLM/xformers use) assertion inside a
KV-block layout conversion path that prefix-cache reuse triggers on the second request
— and it requires compute capability ≥ 8.0 (Ampere). `dgx-v100-01`, the only working
serving node on this cluster (see Phase 1's node-selection writeup), is Volta (cc 7.0).
Unlike FlashAttention-2 — which vLLM detects isn't available on pre-Ampere GPUs and
gracefully falls back to XFormers for — prefix caching's Triton kernel path has no such
fallback here; it just aborts the process.

Reverted `enable_prefix_caching` to `false` and redeployed immediately (job 35800) to
restore service — confirmed healthy with a normal chat completion afterward. Not
pursuing a workaround: patching vLLM's/Triton's kernel selection for Volta is out of
scope for an application-layer serving project, and the honest, useful finding here is
exactly this — measured, root-caused, and documented, not glossed over or silently
left broken. The code path is real and would work as designed on Ampere+ hardware.

## Phase 11 — what's implemented

- `dashboard/grafana/servellm.json` — 11-panel dashboard: GPU utilization, GPU memory
  used/total, in-flight requests, HTTP latency (p50/p95), TTFT (p50/p95), tokens/sec,
  requests/sec by model+status, model usage share (pie), LoRA adapter usage (filters
  `model=~".*:.*"` — Phase 3/4's routing convention), prompt vs completion tokens,
  HTTP status codes.
- `dashboard/grafana/provisioning/` — datasource + dashboard auto-provisioning, so the
  dashboard is there on first login, not a manual import step.
- `scripts/observability_start.sh` / `observability_stop.sh` — this cluster's login
  node has no Docker/Podman/Singularity, so `docker/docker-compose.yml`'s Grafana
  service isn't usable here. Both Prometheus and Grafana ship as self-contained static
  binaries needing neither root nor a container runtime — these scripts run them as
  plain background processes instead, on ports 9091/3001 (not the defaults 9090/3000:
  this account already runs an unrelated Prometheus+Grafana pair for a different
  project bound to the defaults — checked with `ss -tln` before assuming they were free).
  Config templates (`dashboard/prometheus.local.yml`, `dashboard/grafana.ini`,
  the provisioning YAML) live in the repo with `__PLACEHOLDER__` tokens the start
  script substitutes with real absolute paths at launch, so they're portable across
  checkouts rather than hardcoded to one machine.

### A real bug this surfaced (2026-08-26)

Building a dashboard against real metrics found that `TOKENS_PER_SECOND` (defined in
`backend/metrics/middleware.py` since **Phase 1**) was never actually `.observe()`'d
anywhere — a permanently empty panel waiting to happen. Fixed: `backend/gateway/main.py`
now computes `completion_tokens / elapsed_seconds` and records it at all three
completion points (non-streaming chat, streaming chat, `/v1/completions`).

### Verified (2026-08-26)

Ran both binaries against the live gateway, generated real traffic across all three
routes (`general`, `code`, `general:colorist`), then queried Prometheus directly with
every panel's exact PromQL expression: 10 of 11 returned real, non-empty, correct
values (e.g. `sum(servellm_completion_tokens_total) by (model)` → 77/51/90 matching
actual traffic sent). The 11th (TTFT p50 quantile) initially returned `NaN` —
investigated rather than dismissed: the raw `servellm_ttft_seconds_bucket` data was
confirmed correctly populated (2 observations landing in the expected 0.05-0.075s
buckets, consistent with earlier measured TTFT numbers), so this is `histogram_quantile`
+ `rate()`'s well-known low-sample-count edge case, not a metrics bug — resolves under
any sustained traffic. Also confirmed the full path works end-to-end through the actual
committed `scripts/observability_start.sh` (not just ad-hoc manual commands): killed
the manually-started processes, reran the script from a clean state, re-verified
Prometheus target health, Grafana datasource, and dashboard provisioning all came up
correctly with the path-substitution logic.

## Phase 12 — what's implemented

- `benchmark/bench_chat.py` — general-purpose concurrent load generator: latency
  percentiles (p50/p95/p99), TTFT via streaming, aggregate token throughput,
  requests/sec. The reusable tool the narrower Phase 6-8 scripts
  (`scripts/bench_priority.py`, `bench_batching.py`, `bench_prefix_cache.py`) are each
  purpose-built variants of.
- `benchmark/bench_vllm_vs_transformers.py` — the comparison the original
  architecture doc specifically asked for and Phase 7 didn't cover (Phase 7 compared
  vLLM against *itself* with continuous batching on/off, not against a different
  framework). Loads TinyLlama directly via plain `transformers.generate()` and
  compares against the same model served through vLLM (via the live gateway), both
  sequential/single-stream so the comparison isolates per-framework overhead rather
  than mixing in Phase 7's batching effect.
- GPU utilization/memory during benchmark runs: not duplicated here — already covered
  live by Phase 5's `/v1/admin/gpu` and Phase 11's dashboard; this suite's job is
  request-level timing, not system telemetry the platform already exposes.

### Verified (2026-08-26)

`benchmark/bench_chat.py` against `general`, 32 requests at concurrency 8: non-streaming
p50/p95/p99 latency 0.79s/1.33s/1.37s, 579 tokens/s aggregate; streaming adds TTFT
p50/p95 of 0.080s/0.125s (consistent with earlier Phase 3/4 measurements), throughput
essentially unchanged (580 tokens/s) — streaming's cost is real-time delivery, not
reduced total throughput.

`benchmark/bench_vllm_vs_transformers.py`, 5 sequential requests, 100 max_tokens, run
via a dedicated GPU allocation (not the login node — `transformers` needs its own CUDA
context separate from ServeLLM's own engines): plain `transformers.generate()` averaged
1.71s/request (35.1 tokens/s); the same model served through vLLM averaged 0.44s/request
(135.7 tokens/s) — **vLLM is 3.87x faster even with zero batching involved on either
side**, purely from PagedAttention and vLLM's more efficient decode kernels. This is
the "why vLLM over naive HF serving" case made concrete with numbers, distinct from
Phase 7's "why continuous batching" case (~7.8x, a different and additive effect).

## Phase 13 — what's implemented

- `backend/gateway/auth.py` — `require_api_key` now accepts either a static API key
  (Phase 1) or a JWT (new). `POST /v1/admin/auth/token` trades a valid API key for a
  short-lived JWT — deliberately not itself behind `require_api_key`, since it's how a
  caller gets a credential in the first place; it validates the key in the request body
  directly. JWTs don't add a second layer of trust, just a credential with a built-in
  expiry that can be handed to something less trusted than the raw API key.
- `backend/gateway/rate_limit.py` — in-memory sliding-window limiter (client identity =
  the raw Authorization header if present, else IP), applied to `/v1/chat/completions`
  and `/v1/completions`. 429 with a `Retry-After` header when exceeded. In-memory is
  correct for this project's single-process uvicorn deployment; documented as needing a
  shared store (Redis, already have `redis_url` unused since Phase 1) if ever run with
  multiple worker processes.
- `backend/database/request_log.py` — CRUD over `RequestLog` (scaffolded in Phase 1,
  unused until now). `GET /v1/admin/requests` (filterable by `model`/`status`) is the
  "request history" + "admin dashboard" surface — consistent with treating Grafana
  (Phase 11) as the numeric/visual admin dashboard and `/v1/admin/*` as the
  programmatic one, rather than building a second, redundant web UI.
- `backend/gateway/main.py` — every `/v1/chat/completions` and `/v1/completions` call
  (streaming and non-streaming, success and error) now writes a `RequestLog` row:
  model, adapter (parsed from `"<base>:<adapter>"`), tokens, latency, TTFT (streaming
  only), status. Logging failures are caught and logged, never allowed to break an
  otherwise-successful response.

### Verified (2026-08-26)

All local logic unit-tested against SQLite before deploying (JWT create/verify —
valid/garbage/expired/wrong-secret all behave correctly; rate limiter blocks exactly at
the configured threshold; request_log CRUD round-trips and filters correctly), then the
full stack against the live server:
- **Rate limiting**: 65 rapid requests → 57 succeeded, 8 got `429` with
  `Retry-After: 32` — the cutover landed exactly where expected once accounting for 3
  earlier requests still inside the 60s sliding window from a prior test, confirming
  the window persists correctly across separate calls, not per-script-run.
- **Request history**: `GET /v1/admin/requests` returned real logged rows matching
  actual traffic sent (tokens, latency, timestamps all correct).
- **JWT flow**, auth temporarily enabled for this test then reverted (same
  constrain-verify-revert pattern as Phases 6-8): no-auth → `401`; wrong key → `401`;
  correct key → `200`; `POST /v1/admin/auth/token` wrong key → `401`, correct key →
  `200` with a real JWT; that JWT then used successfully for both an inference request
  and an admin endpoint. Every case behaved correctly on the first deploy.

## Phase 14 — what's implemented

- `deploy/helm/servellm/` — a real Helm chart: `Deployment` (readiness/liveness probes
  tuned to this project's own measured cold-start time — 1-2 minutes to load two
  models, per Phase 1-2's numbers, not guessed), `Service`, `ConfigMap` (renders
  `values.yaml`'s `models:` list into exactly the YAML shape
  `backend/router/registry.py`'s `load_model_specs()` expects), `Secret` (API keys /
  JWT secret), `PersistentVolumeClaim` (HF model cache — assumes normal internet
  egress, unlike this project's actual offline dev compute nodes), optional `HPA`
  (CPU/memory only — GPU-aware autoscaling needs a custom metrics pipeline reading the
  Phase 5 `servellm_gpu_*` series, not set up here) and `ServiceMonitor` (Prometheus
  Operator CRD, gated behind `serviceMonitor.enabled`).
- `scripts/validate_helm_chart.sh` — `helm lint`, `helm template` (twice: default
  values and with `autoscaling`/`serviceMonitor` enabled), `kubeconform` schema
  validation against real Kubernetes API schemas (plus the community CRD catalog for
  `ServiceMonitor`, not a core k8s type), and — the piece that actually connects this
  to the running application rather than just being YAML that looks right — extracting
  the rendered `ConfigMap`'s `models.yaml` and parsing it through the real
  `load_model_specs()` from `backend/router/registry.py`.

### What "done" means here, precisely

This project's dev environment (a shared academic HPC login node) has no Docker daemon
and no Kubernetes cluster — confirmed back in Phase 11 when the same constraint forced
Prometheus/Grafana to run as plain binaries instead of via `docker-compose`. There is
nothing to `helm install` this chart onto from here, and no way to build/push the image
`docker/Dockerfile.backend` describes. **Verified**: the chart is syntactically and
schema-correct Kubernetes YAML that would plausibly deploy, and its model configuration
round-trips correctly through the actual application parsing code — both real,
automatable checks, not visual inspection. **Not verified, and can't be from here**: an
actual `helm install` against a running cluster, a built container image actually
starting, `nvidia.com/gpu` scheduling actually working, `readinessProbe` timing being
right in practice rather than just plausible. `deploy/helm/servellm/templates/NOTES.txt`
spells out the concrete missing steps (build+push the image, install the NVIDIA device
plugin) for whoever does have a cluster to try it on.

### Verified (2026-08-26)

```
$ bash scripts/validate_helm_chart.sh
=== helm lint ===                                                    0 charts failed
=== helm template (default values) + kubeconform ===                 5/5 valid
=== helm template (autoscaling + serviceMonitor enabled) + kubeconform ===  7/7 valid
=== rendered models ConfigMap parses through the real registry code ===
2 model(s) parsed correctly: ['general', 'code']
```

`kubeconform`'s first pass flagged `ServiceMonitor` as "could not find schema" — investigated
rather than dismissed: that's expected, since it's a Prometheus Operator CRD, not a core
Kubernetes type, and isn't in `kubeconform`'s default schema set. Re-ran pointed at the
community CRD schema catalog and it validated clean (7/7) — confirming the earlier
result really was a missing-schema-source gap, not an actual problem with the manifest.

## Operational incident — submitting shell's environment hijacked the job (2026-09-05)

`sbatch scripts/sbatch_serve.sh` failed with `ModuleNotFoundError: No module named 'vllm'`
— misleading, since `vllm` was never missing. The job was submitted from a terminal where
an unrelated project's virtualenv had been auto-activated on top of conda `base`. `sbatch`'s
default behavior inherits the submitting shell's entire environment (`--export=ALL` is the
default), so that venv's `PATH`/`VIRTUAL_ENV` state rode straight into the job — the
script's own `conda activate servellm` line executed without error, but `python3` still
resolved to a different conda env, with `uvicorn` imported from the other project's `.venv`
instead. Confirmed by reading the actual traceback rather than guessing from the top-level
error message: it named the wrong environment paths outright.

Fixed in `scripts/sbatch_serve.sh` with three layers, each verified independently:
1. `#SBATCH --export=NONE` — refuse to inherit the submitting shell's environment at all.
2. Defensive `unset`/`conda deactivate` loop plus an explicit interpreter-path resolution
   with a fail-fast `import vllm` check, so a similar issue fails loudly at the top of the
   script instead of three files deep in an unrelated traceback.
3. `srun --export=ALL` on the actual server-launching step — needed because this cluster's
   `srun`, called from within a job submitted with `--export=NONE`, otherwise inherits that
   same NONE policy instead of the batch script's own (by-then-clean) rebuilt environment,
   so step 1 alone silently undid step 2's `export PYTHONNOUSERSITE=1` for the actual server
   process. Caught by resubmitting after fix #1+#2 alone and finding a *different* stale
   `~/.local` package (`triton`) still leaking in, not by reasoning it through in advance.

### Verified (2026-09-05)

Resubmitted after all three fixes: both models (`general`, `code`) loaded successfully end
to end (`all models ready`, `Application startup complete`) under the corrected script. The
only errors seen afterward (a port-bind conflict, then a CUDA OOM on a scheduler
auto-requeue) were both artifacts of test-submitting a second instance while the original,
already-healthy job was still holding the same GPU node — not defects in the fix — and
resolved by cancelling the redundant test jobs, confirmed via `sacct`.
