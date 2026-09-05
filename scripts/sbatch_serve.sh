#!/bin/bash
#SBATCH --job-name=servellm
#SBATCH --partition=cse-cpu-all
#SBATCH --nodelist=dgx-v100-01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# 24h, not --time=infinite (which the partition does allow): this is a
# shared multi-tenant cluster (~60 concurrent users seen on dgx-a100-02
# alone) — holding a GPU indefinitely for a portfolio demo is inconsiderate.
# Raise this (or scancel + resubmit) if a longer live session is actually needed.
#SBATCH --output=%x-%j.log
#SBATCH --export=NONE
# --export=NONE: don't inherit the submitting shell's environment at all.
# Real incident (2026-09-05): submitted from a terminal where an unrelated
# project's virtualenv (FlowForge/.venv) had been auto-activated on top of
# conda base — sbatch's default --export=ALL carried that PATH/VIRTUAL_ENV
# pollution straight into the job, so this script's own `conda activate
# servellm` below never actually took effect: `python3` still resolved to a
# different conda env (`env`) with `uvicorn` imported from FlowForge's
# venv, and the job died on `ModuleNotFoundError: No module named 'vllm'`
# — not because vllm was missing, but because the wrong Python ran. Fixed
# by refusing to inherit anything and rebuilding the environment from
# scratch below (belt-and-suspenders: also explicitly deactivates/unsets
# any conda env or venv state that could still leak in some other way).
# Pinned to dgx-v100-01, not dgx-a100-02 or the P100 nodes:
#  - dgx-a100-02 (Ampere, ideal on paper) had CPU load ~18 from ~60 other
#    users' sessions; even `import torch` hung for 10+ min despite the GPUs
#    themselves sitting idle. Node-level contention, not a code issue.
#  - dgx-p100-01 / cse-node009-012 (Pascal, sm_60) fail hard at inference
#    time: "CUDA error: no kernel image is available for execution on the
#    device" — this torch/xformers build's compiled kernels don't target
#    Pascal at all. Not fixable by config.
#  - dgx-v100-01 (Volta, sm_70) verified working end-to-end (see
#    scripts/dev_client_smoke_test.py) and had much lower load than a100-02.
# vLLM auto-falls-back to the XFormers attention backend here (no
# FlashAttention-2 on pre-Ampere) — logged as INFO, not an error.

set -euo pipefail

# Defensive, on top of --export=NONE above: strip any venv/conda state that
# could still be present (e.g. SLURM/PAM re-sourcing a login profile that
# auto-activates something) before setting up the one environment this job
# actually wants.
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME || true
for _ in 1 2 3; do conda deactivate 2>/dev/null || break; done

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# Compute nodes on this cluster have no internet egress. Model weights must
# already be in ~/.cache/huggingface (pre-download from the login node, which
# does have internet) — offline mode skips the network metadata check that
# otherwise hangs for minutes before timing out.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(pwd)"

# One-time setup: `conda create -n servellm python=3.10 -y` then
# `conda activate servellm && pip install -r backend/requirements.txt`.
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate servellm

# Belt-and-suspenders on top of `conda activate` above: resolve and print
# the interpreter that will actually run, and hard-fail loudly here rather
# than dying inside a Python traceback three files deep if activation
# silently didn't take effect the way the 2026-09-05 incident (see
# --export=NONE comment above) did.
PYTHON_BIN="$(command -v python3)"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import vllm" || {
    echo "FATAL: '$PYTHON_BIN' cannot import vllm — conda activate servellm" \
         "did not take effect (check for inherited venv/conda state)." >&2
    exit 1
}

export SERVELLM_HOST=0.0.0.0
export SERVELLM_PORT=8000
[ -f "$PROJECT_ROOT/.env" ] && set -a && source "$PROJECT_ROOT/.env" && set +a

echo "Node: $(hostname)"
echo "PWD: $(pwd)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# --export=ALL here (distinct from #SBATCH --export=NONE above): NONE
# governs what's inherited from the *submitting* shell (deliberately
# nothing, per the 2026-09-05 incident); ALL here means srun instead
# inherits *this script's own* current environment — the clean one just
# rebuilt above (PYTHONNOUSERSITE, conda activate servellm, HF_HUB_OFFLINE)
# — not the submission-time one. Without this, PYTHONNOUSERSITE silently
# didn't reach the srun task even though it was exported earlier in this
# same script, and triton loaded from ~/.local instead of the servellm
# env's own copy, since this cluster's srun appears to default to the
# same NONE policy as its parent sbatch job rather than the shell it's
# actually invoked from.
srun --ntasks=1 --export=ALL "$PYTHON_BIN" -m uvicorn backend.gateway.main:app \
    --host "$SERVELLM_HOST" --port "$SERVELLM_PORT"
