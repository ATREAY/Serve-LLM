"""CRUD over the AdapterRegistry table — the DB-backed half of Phase 4's
dynamic adapter registry. backend/lora/dynamic.py is what actually resolves
these rows to loadable LoRARequests; this module only touches the DB.
"""

import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import AdapterRegistry


def register_adapter(
    session: Session, *, base_model: str, name: str, hf_repo: str, version: str = "v1"
) -> AdapterRegistry:
    existing = get_adapter(session, base_model=base_model, name=name)
    if existing is not None:
        existing.hf_repo = hf_repo
        existing.version = version
        existing.status = "registered"
        session.commit()
        session.refresh(existing)
        return existing

    row = AdapterRegistry(base_model=base_model, name=name, hf_repo=hf_repo, version=version)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_adapter(session: Session, *, base_model: str, name: str) -> AdapterRegistry | None:
    stmt = select(AdapterRegistry).where(
        AdapterRegistry.base_model == base_model, AdapterRegistry.name == name
    )
    return session.execute(stmt).scalar_one_or_none()


def list_adapters(session: Session, *, base_model: str | None = None) -> list[AdapterRegistry]:
    stmt = select(AdapterRegistry)
    if base_model is not None:
        stmt = stmt.where(AdapterRegistry.base_model == base_model)
    return list(session.execute(stmt).scalars().all())


def touch_adapter(session: Session, *, base_model: str, name: str) -> None:
    """Called on every successful use: bumps hits, marks last_used_at/status."""
    row = get_adapter(session, base_model=base_model, name=name)
    if row is None:
        return
    row.hits += 1
    row.last_used_at = datetime.datetime.utcnow()
    row.status = "loaded"
    session.commit()


def mark_idle(session: Session, *, base_model: str, name: str) -> None:
    row = get_adapter(session, base_model=base_model, name=name)
    if row is None:
        return
    row.status = "idle"
    session.commit()


def delete_adapter(session: Session, *, base_model: str, name: str) -> bool:
    row = get_adapter(session, base_model=base_model, name=name)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


def find_idle_since(
    session: Session, *, cutoff: datetime.datetime
) -> list[AdapterRegistry]:
    stmt = select(AdapterRegistry).where(
        AdapterRegistry.status == "loaded",
        AdapterRegistry.last_used_at.is_not(None),
        AdapterRegistry.last_used_at < cutoff,
    )
    return list(session.execute(stmt).scalars().all())
