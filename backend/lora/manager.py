"""Phase 3: static LoRA adapters — resolves configured adapters to local
weight paths and builds the vllm.lora.request.LoRARequest each generate()
call needs. "Static" means the adapter list comes from models.yaml and is
resolved once at startup; Phase 4 adds a DB-backed registry with dynamic
load/unload while the server is running (see backend/database/models.py).
"""

from dataclasses import dataclass

from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest


@dataclass(frozen=True)
class AdapterEntry:
    name: str
    hf_repo: str


def resolve_adapter_requests(
    adapters: list[AdapterEntry], *, start_id: int = 1
) -> dict[str, LoRARequest]:
    """Downloads (or finds already-cached) weights for each adapter and
    assigns each a unique lora_int_id. Requires the weights to already be in
    the local HF cache when running with HF_HUB_OFFLINE=1 (compute nodes on
    this cluster have no internet — pre-download from the login node, same
    as base model weights)."""
    requests: dict[str, LoRARequest] = {}
    for i, adapter in enumerate(adapters):
        local_path = snapshot_download(adapter.hf_repo)
        requests[adapter.name] = LoRARequest(
            lora_name=adapter.name,
            lora_int_id=start_id + i,
            lora_path=local_path,
        )
    return requests
