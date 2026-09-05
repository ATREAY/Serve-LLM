"""Central settings for ServeLLM, loaded from environment / .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SERVELLM_", extra="ignore")

    # -- Model (Phase 1 single-model fallback; ignored if models_config_path
    # exists — see backend/router/registry.py) --
    model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    served_model_id: str = Field(default="default")
    dtype: str = Field(default="auto")
    max_model_len: int | None = Field(default=None)
    gpu_memory_utilization: float = Field(default=0.85)
    tensor_parallel_size: int = Field(default=1)
    enable_lora: bool = Field(default=False)
    max_lora_rank: int = Field(default=64)

    # -- Model (Phase 2 multi-model routing) --
    # If this file exists, the gateway loads it via backend/router/registry.py
    # and serves every model it lists instead of the single model_name above.
    models_config_path: str = Field(default="backend/router/models.yaml")

    # -- Server --
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    api_keys: str = Field(default="")  # comma-separated; empty disables auth

    # -- Datastores --
    # SQLite by default so Phase 4 (dynamic adapter registry) works with zero
    # extra infra on the GPU compute node; point at a real Postgres via this
    # var for the docker-compose deployment. Same SQLAlchemy models either way.
    database_url: str = Field(default="sqlite:///./servellm.db")
    redis_url: str | None = Field(default=None)

    # -- Phase 4: dynamic LoRA adapters --
    adapter_idle_ttl_seconds: int = Field(default=600)
    adapter_sweep_interval_seconds: int = Field(default=60)

    # -- Phase 13: security --
    # Dev default, not a real secret — fine as shipped because this server is
    # never reached except via SSH port-forward (see scripts/observability_start.sh
    # for the same pattern) or from other jobs on this private cluster subnet.
    # Set explicitly (and keep private) before exposing this anywhere less trusted.
    jwt_secret: str = Field(default="servellm-dev-secret-change-me-before-any-real-use")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_seconds: int = Field(default=3600)
    # Sliding window, in-memory (see backend/gateway/rate_limit.py) — correct
    # for this single-process uvicorn deployment; would need a shared store
    # (e.g. Redis, via redis_url above) behind multiple worker processes.
    rate_limit_requests: int = Field(default=60)
    rate_limit_window_seconds: int = Field(default=60)

    def allowed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
