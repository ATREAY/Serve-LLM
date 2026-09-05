"""SQLAlchemy models for the request log and the LoRA adapter registry.

RequestLog backs Phase 13's request history (GET /v1/admin/requests) — see
backend/database/request_log.py. AdapterRegistry backs Phase 4's dynamic
adapter registration — see backend/lora/dynamic.py.
"""

import datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True)
    model: Mapped[str] = mapped_column(String(128))
    adapter: Mapped[str | None] = mapped_column(String(128), nullable=True)
    endpoint: Mapped[str] = mapped_column(String(64))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    ttft_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class AdapterRegistry(Base):
    __tablename__ = "adapter_registry"
    __table_args__ = (UniqueConstraint("base_model", "name", name="uq_adapter_base_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32), default="v1")
    base_model: Mapped[str] = mapped_column(String(128))
    # HF repo id (or local path), resolved to an actual local snapshot lazily
    # at load time via huggingface_hub.snapshot_download — same pattern as
    # the static adapters in backend/lora/manager.py, not a path stored here.
    hf_repo: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="registered")
    hits: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
