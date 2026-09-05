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

export SERVELLM_HOST=0.0.0.0
export SERVELLM_PORT=8000
[ -f "$PROJECT_ROOT/.env" ] && set -a && source "$PROJECT_ROOT/.env" && set +a

echo "Node: $(hostname)"
echo "PWD: $(pwd)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

srun --ntasks=1 python3 -m uvicorn backend.gateway.main:app \
    --host "$SERVELLM_HOST" --port "$SERVELLM_PORT"
