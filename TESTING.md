# Testing ServeLLM

Quick reference for exercising every phase of this project. Assumes you're on the
login node with the `servellm` conda env available, and that a server job is already
running (see "Starting / restarting the server" if not).

Working directory for all commands below: `~/MLSys_Projects/LLM-Systems-Portfolio/ServeLLM`

```bash
conda activate servellm   # or: source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && conda activate servellm
```

Server base URL throughout: `http://dgx-v100-01:18742` — swap the node if
`scripts/sbatch_serve.sh` is ever changed to target a different one.

---

## 0. Starting / restarting the server

The SLURM job has a 24h time limit (courtesy on a shared cluster, not a technical
requirement — see `scripts/sbatch_serve.sh`), so it will eventually need restarting.

```bash
# check if it's already running
squeue -u $USER | grep servellm

# if not (or to pick up code changes), submit a fresh job
cd ~/MLSys_Projects/LLM-Systems-Portfolio/ServeLLM
sbatch scripts/sbatch_serve.sh

# watch it come up (takes ~2-3 min: two models load + CUDA graph capture)
tail -f servellm-<jobid>.log
# ready once you see: "all models ready" / "Application startup complete"

# stop it
scancel <jobid>
```

---

## 1. Quickest check — is it alive?

```bash
curl http://dgx-v100-01:18742/healthz
curl http://dgx-v100-01:18742/v1/models
```

---

## 2. Basic inference (Phases 1-3: streaming, multi-model routing, static LoRA)

Phase 3's static LoRA demo (`general:colorist`) no longer exists — it was
trained against TinyLlama-1.1B specifically and isn't compatible with
`general`'s current base model (see the 2026-09-05 quality-upgrade entry in
`docs/ROADMAP.md`). The dynamic-adapter path (section 3 below) still works
unchanged with any adapter trained for the current base model.

```bash
# non-streaming, the general model
curl -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20}'

# the code model
curl -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"code","messages":[{"role":"user","content":"Write a python fibonacci function."}],"max_tokens":80}'

# streaming (watch tokens arrive one at a time; -N disables curl's own buffering)
curl -N -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general","messages":[{"role":"user","content":"Count to 5."}],"max_tokens":20,"stream":true}'

# legacy /v1/completions
curl -X POST http://dgx-v100-01:18742/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general","prompt":"The capital of France is","max_tokens":10}'

# unknown model — should 404 listing what's actually available
curl -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"does-not-exist","messages":[{"role":"user","content":"hi"}]}'
```

---

## 3. Dynamic LoRA (Phase 4) — register and use a new adapter, no restart needed

```bash
# register (weights must already be cached — see README's pre-download step
# if this is a genuinely new adapter, not tarot which is already cached)
curl -X POST http://dgx-v100-01:18742/v1/admin/adapters \
  -H "Content-Type: application/json" \
  -d '{"base_model":"general","name":"tarot","hf_repo":"barissglc/tinyllama-tarot-v1"}'

# use it immediately — first request triggers the lazy load
curl -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general:tarot","messages":[{"role":"user","content":"Draw me a card."}],"max_tokens":40}'

# list registered adapters (hits, status, last_used_at)
curl http://dgx-v100-01:18742/v1/admin/adapters

# unregister
curl -X DELETE http://dgx-v100-01:18742/v1/admin/adapters/general/tarot
```

---

## 4. GPU telemetry (Phase 5)

```bash
curl http://dgx-v100-01:18742/v1/admin/gpu
curl http://dgx-v100-01:18742/metrics | grep servellm_gpu
```

---

## 5. Priority scheduling (Phase 6)

`general` runs `scheduling_policy: priority` by default, but priority only visibly
reorders anything under real queueing contention. To reproduce the measured result
(see `docs/ROADMAP.md`'s Phase 6 section), temporarily constrain concurrency:

```bash
# edit backend/router/models.yaml: add `max_num_seqs: 3` under the `general` entry,
# then redeploy (scancel + sbatch scripts/sbatch_serve.sh), then:
python scripts/bench_priority.py --base-url http://dgx-v100-01:18742 --model general --concurrency 24

# revert max_num_seqs afterward and redeploy again for normal use
```

---

## 6. Continuous batching comparison (Phase 7)

Same constrain/revert pattern as Phase 6 — `max_num_seqs: 1` forces serialization:

```bash
python scripts/bench_batching.py --base-url http://dgx-v100-01:18742 --model general --concurrency 16
```

---

## 7. Prefix caching (Phase 8) — known blocked, do not enable on this hardware

`enable_prefix_caching: true` on `general` in `backend/router/models.yaml` **crashes
the whole server process** on this cluster's V100 node (Triton kernel requires
Ampere). Leave this off — see `docs/ROADMAP.md`'s Phase 8 section for the full trace
if you want to see why, not to retry it here.

---

## 8. Benchmark suite (Phase 12)

```bash
# general concurrent load test: latency percentiles, TTFT, throughput
python benchmark/bench_chat.py --base-url http://dgx-v100-01:18742 --model general --concurrency 8 --requests 32
python benchmark/bench_chat.py --base-url http://dgx-v100-01:18742 --model general --concurrency 8 --requests 32 --stream

# vLLM vs plain HuggingFace transformers.generate() — needs its own GPU allocation
srun --partition=cse-cpu-all --nodelist=dgx-v100-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=00:10:00 \
  python benchmark/bench_vllm_vs_transformers.py --base-url http://dgx-v100-01:18742 --model general --requests 5
```

---

## 9. Grafana dashboard (Phase 11) — visual view

Prometheus + Grafana run as standalone background processes on the login node
(no Docker here — see `scripts/observability_start.sh`).

```bash
# start them if not already running
bash scripts/observability_start.sh

# check
curl http://127.0.0.1:9091/-/healthy    # Prometheus
curl http://127.0.0.1:3001/api/health   # Grafana

# stop
bash scripts/observability_stop.sh
```

From your own machine (not the cluster):
```bash
ssh -L 3001:localhost:3001 <you>@<login-node>
```
Then open `http://localhost:3001` (login: `admin` / `admin`) — dashboard **"ServeLLM"**
is already there, no manual import needed.

---

## 10. Security: auth, JWT, rate limiting, request history (Phase 13)

Request history works regardless of auth state:
```bash
curl http://dgx-v100-01:18742/v1/admin/requests?limit=10
curl "http://dgx-v100-01:18742/v1/admin/requests?model=general&status=ok&limit=5"
```

Rate limiting is always on (default 60 req/60s per client) — fire enough requests
fast enough and you'll see `429` with a `Retry-After` header.

Auth is **disabled by default** (matches Phase 1's "works out of the box" design).
To try the JWT flow:
```bash
# 1. edit .env: set SERVELLM_API_KEYS=your-test-key
# 2. redeploy: scancel <jobid> && sbatch scripts/sbatch_serve.sh
# 3. then:
curl -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Authorization: Bearer wrong-key" \
  -d '{"model":"general","messages":[{"role":"user","content":"hi"}]}'   # expect 401

curl -X POST http://dgx-v100-01:18742/v1/admin/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key":"your-test-key"}'   # returns a JWT

curl -X POST http://dgx-v100-01:18742/v1/chat/completions \
  -H "Authorization: Bearer <jwt-from-above>" \
  -d '{"model":"general","messages":[{"role":"user","content":"hi via jwt"}],"max_tokens":10}'   # expect 200

# 4. revert .env's SERVELLM_API_KEYS back to empty and redeploy when done
```

---

## 11. Kubernetes/Helm chart (Phase 14) — no cluster needed

```bash
bash scripts/validate_helm_chart.sh
```

Runs `helm lint`, `helm template` (twice — default values and with
`autoscaling`/`serviceMonitor` enabled), `kubeconform` schema validation, and
round-trips the rendered model config through the real `backend/router/registry.py`
parser. Requires the one-time `helm`/`kubeconform` binary setup described in that
script's header comment (already done in this environment, under `~/tools/`).

---

## Other useful checks

```bash
# full smoke test script (health, models, chat, streaming)
python scripts/dev_client_smoke_test.py --base-url http://dgx-v100-01:18742

# syntax-check everything after making code changes
python3 -c "
import ast, sys, pathlib
errs = []
for p in list(pathlib.Path('backend').rglob('*.py')) + list(pathlib.Path('scripts').rglob('*.py')) + list(pathlib.Path('benchmark').rglob('*.py')):
    try: ast.parse(p.read_text(), filename=str(p))
    except SyntaxError as e: errs.append((p, e))
print('all OK' if not errs else errs)
"
```

See `docs/ROADMAP.md` for what was actually measured/verified at each phase (real
numbers, not just "it should work"), and `README.md` for setup from scratch.
