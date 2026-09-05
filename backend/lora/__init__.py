# Phase 3: static LoRA adapters — manager.py resolves each configured adapter
# to a local weight path and builds the vllm.lora.request.LoRARequest each
# generate() call needs. Adapters are declared in backend/router/models.yaml
# and resolved once at startup (see backend/router/registry.py).
#
# Phase 4 (not yet built): dynamic load/unload of adapters while the server
# is running, backed by AdapterRegistry in backend/database/models.py instead
# of the static models.yaml list.
