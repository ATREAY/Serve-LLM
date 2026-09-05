"""Phase 4: dynamic LoRA adapters — register/unregister at runtime, no
gateway restart required. Static Phase 3 adapters (backend/router/models.yaml)
are resolved once at startup with fixed lora_int_ids; these are resolved
lazily on first use and given ids from a separate range so the two schemes
never collide.

"Unload idle adapters" here means: dropped from our own in-memory cache and
marked idle in the DB, so /v1/models stops advertising it and the next
request re-resolves (and re-downloads, if evicted) it from scratch. It does
NOT force vLLM to evict the adapter's weights from GPU/CPU memory — that's
governed by max_loras/max_cpu_loras on the engine itself, which applies its
own LRU policy across however many distinct LoRARequests it has actually
seen. Our eviction keeps the *catalog* accurate; vLLM's own cache is a
separate, engine-internal concern the gateway doesn't control per-adapter.
"""

import asyncio
import datetime
import logging
import threading

from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest

from backend.database import adapter_catalog
from backend.database.session import get_session

logger = logging.getLogger("servellm.lora.dynamic")

# Static adapters (backend/router/registry.py) use ids starting at 1; this
# range starts well clear of any realistic static adapter count.
_DYNAMIC_ID_START = 1000


class DynamicAdapterCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = _DYNAMIC_ID_START
        # (base_model, name) -> LoRARequest
        self._loaded: dict[tuple[str, str], LoRARequest] = {}

    def get_or_load(self, base_model: str, name: str) -> LoRARequest | None:
        key = (base_model, name)
        with self._lock:
            cached = self._loaded.get(key)
        if cached is not None:
            self._record_use(base_model, name)
            return cached

        session = get_session()
        try:
            row = adapter_catalog.get_adapter(session, base_model=base_model, name=name)
            if row is None:
                return None
            local_path = snapshot_download(row.hf_repo)
            with self._lock:
                lora_request = self._loaded.get(key)
                if lora_request is None:
                    lora_request = LoRARequest(
                        lora_name=f"{base_model}:{name}",
                        lora_int_id=self._next_id,
                        lora_path=local_path,
                    )
                    self._next_id += 1
                    self._loaded[key] = lora_request
        finally:
            session.close()

        self._record_use(base_model, name)
        return self._loaded[key]

    def _record_use(self, base_model: str, name: str) -> None:
        session = get_session()
        try:
            adapter_catalog.touch_adapter(session, base_model=base_model, name=name)
        finally:
            session.close()

    def loaded_ids(self) -> list[str]:
        with self._lock:
            return [f"{base}:{name}" for base, name in self._loaded.keys()]

    def evict(self, base_model: str, name: str) -> None:
        with self._lock:
            self._loaded.pop((base_model, name), None)


async def run_idle_sweep(cache: DynamicAdapterCache, *, idle_seconds: int, interval_seconds: int):
    """Background task: periodically drops adapters idle past idle_seconds
    from the in-memory cache and marks them idle in the DB. Cancelled from
    the gateway lifespan on shutdown."""
    while True:
        await asyncio.sleep(interval_seconds)
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=idle_seconds)
        session = get_session()
        try:
            idle_rows = adapter_catalog.find_idle_since(session, cutoff=cutoff)
            for row in idle_rows:
                cache.evict(row.base_model, row.name)
                adapter_catalog.mark_idle(session, base_model=row.base_model, name=row.name)
                logger.info(
                    "evicted idle adapter base_model=%s name=%s (idle > %ss)",
                    row.base_model,
                    row.name,
                    idle_seconds,
                )
        finally:
            session.close()
