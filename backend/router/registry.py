"""Phase 2: routes requests to one of several base models by served_model_id.
Phase 3: each model may also declare static LoRA adapters, addressed as
"<served_model_id>:<adapter_name>" (e.g. "general:colorist").
Phase 4: adapters can also be registered at runtime via the DB-backed
DynamicAdapterCache, resolved on first use — no gateway restart required.

Loads a static config (backend/router/models.yaml by default) describing each
model to serve, and owns one VLLMEngineWrapper per entry — all sharing the
same physical GPU, each capped to its own slice of gpu_memory_utilization so
they don't fight over KV cache space at startup.
"""

import logging
import os
from dataclasses import dataclass, field

import yaml
from vllm.lora.request import LoRARequest

from backend.lora.dynamic import DynamicAdapterCache
from backend.lora.manager import AdapterEntry, resolve_adapter_requests
from backend.vllm_engine.engine import VLLMEngineWrapper

logger = logging.getLogger("servellm.router")


@dataclass(frozen=True)
class ModelSpec:
    served_model_id: str
    hf_model: str
    capability: str = "general"
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.4
    tensor_parallel_size: int = 1
    max_lora_rank: int = 16
    adapters: list[AdapterEntry] = field(default_factory=list)
    # Reserved vLLM LoRA slots beyond the static adapters above, for Phase 4
    # adapters registered after startup. max_loras is fixed at engine
    # construction time — it can't grow later, so this headroom has to be
    # decided upfront even though no dynamic adapter exists yet at boot.
    max_dynamic_adapters: int = 0
    # Phase 6: request-level scheduling. "priority" only has any effect if
    # scheduling_policy is also "priority" — under vLLM's default "fcfs" the
    # per-request priority value is accepted but never consulted (verified
    # against vllm/core/scheduler.py: the priority-ordered sort only runs
    # inside _schedule_priority_preemption, which only runs under that policy).
    scheduling_policy: str = "fcfs"
    max_num_seqs: int | None = None
    # Phase 8: engine-wide, reuses KV cache blocks for any shared prompt
    # prefix across requests (e.g. a common system prompt) once enabled —
    # unlike priority, there's no per-request opt-in. Crashes the process on
    # this cluster's Volta node when combined with enforce_eager: false — see
    # docs/ROADMAP.md's Phase 8 section before turning this on.
    enable_prefix_caching: bool = False
    enforce_eager: bool = False


def load_model_specs(config_path: str) -> list[ModelSpec]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    specs = []
    for entry in raw["models"]:
        entry = dict(entry)
        entry["adapters"] = [AdapterEntry(**a) for a in entry.get("adapters", [])]
        specs.append(ModelSpec(**entry))
    if not specs:
        raise ValueError(f"{config_path} declares no models")
    ids = [s.served_model_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate served_model_id in {config_path}: {ids}")
    return specs


class ModelRegistry:
    """Holds one running engine per configured served_model_id, plus the
    resolved LoRARequest for each of its static adapters."""

    def __init__(self, specs: list[ModelSpec]):
        self._specs = {s.served_model_id: s for s in specs}
        self._engines: dict[str, VLLMEngineWrapper] = {}
        # served_model_id -> adapter_name -> LoRARequest
        self._adapters: dict[str, dict[str, LoRARequest]] = {}
        self._dynamic = DynamicAdapterCache()

    async def start(self) -> None:
        # Sequential, not concurrent: each engine claims its
        # gpu_memory_utilization slice at construction time, so starting
        # them one at a time avoids racing over the same free-memory read.
        for spec in self._specs.values():
            logger.info(
                "loading model served_model_id=%s (%s, capability=%s, adapters=%s, "
                "max_dynamic_adapters=%s) ...",
                spec.served_model_id,
                spec.hf_model,
                spec.capability,
                [a.name for a in spec.adapters],
                spec.max_dynamic_adapters,
            )
            enable_lora = bool(spec.adapters) or spec.max_dynamic_adapters > 0
            engine = VLLMEngineWrapper(
                spec.hf_model,
                dtype=spec.dtype,
                max_model_len=spec.max_model_len,
                gpu_memory_utilization=spec.gpu_memory_utilization,
                tensor_parallel_size=spec.tensor_parallel_size,
                enable_lora=enable_lora,
                max_lora_rank=spec.max_lora_rank,
                max_loras=max(len(spec.adapters) + spec.max_dynamic_adapters, 1),
                max_num_seqs=spec.max_num_seqs,
                scheduling_policy=spec.scheduling_policy,
                enable_prefix_caching=spec.enable_prefix_caching,
                enforce_eager=spec.enforce_eager,
            )
            await engine.start()
            self._engines[spec.served_model_id] = engine
            if spec.adapters:
                self._adapters[spec.served_model_id] = resolve_adapter_requests(spec.adapters)
            logger.info("served_model_id=%s ready", spec.served_model_id)

    async def stop(self) -> None:
        for engine in self._engines.values():
            await engine.stop()
        self._engines.clear()
        self._adapters.clear()

    @property
    def ready(self) -> bool:
        return bool(self._engines) and all(e.ready for e in self._engines.values())

    def get(self, served_model_id: str) -> VLLMEngineWrapper | None:
        return self._engines.get(served_model_id)

    def resolve_capability(self, capability: str) -> VLLMEngineWrapper | None:
        """Fallback lookup: if the caller passes a capability tag (e.g. "code")
        instead of an exact served_model_id, route to the first match."""
        for served_model_id, spec in self._specs.items():
            if spec.capability == capability:
                return self._engines.get(served_model_id)
        return None

    def resolve(self, model_id: str) -> tuple[VLLMEngineWrapper | None, LoRARequest | None]:
        """Resolves a request's `model` field. "<base>:<adapter>" routes to
        the base model's engine with that adapter's LoRARequest attached —
        checking the static (Phase 3) adapters first, then the DB-backed
        dynamic (Phase 4) cache, which resolves and caches on first use.
        Anything else falls through to get()/resolve_capability() with no
        adapter (the base model runs unmodified)."""
        if ":" in model_id:
            base_id, adapter_name = model_id.split(":", 1)
            engine = self._engines.get(base_id)
            if engine is None:
                return None, None
            lora_request = self._adapters.get(base_id, {}).get(
                adapter_name
            ) or self._dynamic.get_or_load(base_id, adapter_name)
            if lora_request is None:
                return None, None
            return engine, lora_request
        engine = self._engines.get(model_id) or self.resolve_capability(model_id)
        return engine, None

    def list_ids(self) -> list[str]:
        ids = list(self._specs.keys())
        for served_model_id, adapters in self._adapters.items():
            ids.extend(f"{served_model_id}:{name}" for name in adapters)
        ids.extend(self._dynamic.loaded_ids())
        return ids

    def spec_for(self, served_model_id: str) -> ModelSpec | None:
        return self._specs.get(served_model_id)

    @property
    def dynamic_cache(self) -> DynamicAdapterCache:
        """For the admin endpoints (backend/gateway/main.py) to evict a
        specific adapter immediately after a DELETE, rather than waiting for
        the idle sweep."""
        return self._dynamic


def build_registry(settings) -> "ModelRegistry":
    """Loads models_config_path if it exists (Phase 2, multi-model); otherwise
    falls back to a single-entry registry built from the Phase 1 settings
    fields, so single-model deployments don't need a models.yaml at all."""
    if os.path.isfile(settings.models_config_path):
        specs = load_model_specs(settings.models_config_path)
    else:
        specs = [
            ModelSpec(
                served_model_id=settings.served_model_id,
                hf_model=settings.model_name,
                dtype=settings.dtype,
                max_model_len=settings.max_model_len,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                tensor_parallel_size=settings.tensor_parallel_size,
            )
        ]
    return ModelRegistry(specs)
