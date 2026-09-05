"""Phase 5: real GPU telemetry via NVML, not vLLM's own internal stats.

vLLM's AsyncLLMEngine registers its own Prometheus metrics (KV cache usage,
running/waiting request counts, etc.) automatically, but that only works
cleanly with one engine per process. With two co-resident engines (Phase 2),
the second engine's metric registration is silently incomplete — confirmed
by hitting /metrics and seeing only one `vllm:cache_config_info` line for
two running models, no crash, no warning logged either. Rather than patch
vLLM's internal stat logger, this module reports GPU-level telemetry
(utilization, memory) directly from NVML, which doesn't care how many
engines are sharing the device — closer to what the original architecture's
per-GPU dashboard mockup wanted anyway.

NVML enumerates every physical GPU on the node regardless of
CUDA_VISIBLE_DEVICES (that's a CUDA-runtime concept, not an NVML one) — on a
shared multi-tenant node that means indices for GPUs other users are using
too. This module only queries the index(es) SLURM actually gave this job.
"""

import asyncio
import logging
import os

import pynvml
from prometheus_client import Gauge

logger = logging.getLogger("servellm.metrics.gpu")

GPU_UTILIZATION_PERCENT = Gauge(
    "servellm_gpu_utilization_percent", "GPU compute utilization", ["gpu_index"]
)
GPU_MEMORY_USED_BYTES = Gauge(
    "servellm_gpu_memory_used_bytes", "GPU memory in use", ["gpu_index"]
)
GPU_MEMORY_TOTAL_BYTES = Gauge(
    "servellm_gpu_memory_total_bytes", "GPU total memory", ["gpu_index"]
)


def _visible_gpu_indices() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw:
        return [0]
    return [int(x) for x in raw.split(",") if x.strip() != ""]


class GpuMetricsPoller:
    def __init__(self):
        self._indices: list[int] = []
        self._handles: dict[int, object] = {}
        # Cached alongside the Prometheus gauges so GET /v1/admin/gpu (a
        # human-readable snapshot, not just /metrics' text-exposition format)
        # doesn't have to reach into prometheus_client's internals to read a
        # gauge's current value back out.
        self._last: dict[int, dict] = {}

    def start(self) -> None:
        pynvml.nvmlInit()
        self._indices = _visible_gpu_indices()
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self._indices}
        logger.info("GPU metrics polling NVML indices: %s", self._indices)

    def poll_once(self) -> None:
        for index, handle in self._handles.items():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            label = str(index)
            GPU_UTILIZATION_PERCENT.labels(gpu_index=label).set(util.gpu)
            GPU_MEMORY_USED_BYTES.labels(gpu_index=label).set(mem.used)
            GPU_MEMORY_TOTAL_BYTES.labels(gpu_index=label).set(mem.total)
            self._last[index] = {
                "gpu_index": index,
                "utilization_percent": util.gpu,
                "memory_used_bytes": mem.used,
                "memory_total_bytes": mem.total,
            }

    def snapshot(self) -> list[dict]:
        return [self._last[i] for i in self._indices if i in self._last]

    def shutdown(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


async def run_gpu_poll_loop(poller: GpuMetricsPoller, *, interval_seconds: float = 5.0):
    """Background task, cancelled from the gateway lifespan on shutdown."""
    while True:
        try:
            poller.poll_once()
        except pynvml.NVMLError as e:
            logger.warning("GPU metrics poll failed: %s", e)
        await asyncio.sleep(interval_seconds)
