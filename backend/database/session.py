"""Sync SQLAlchemy engine/session for the adapter registry.

Sync, not async: the actual queries here (a handful of rows, indexed lookups)
are fast enough that the brief event-loop block from calling them directly in
an async handler is not worth the added complexity of an async driver — see
backend/lora/dynamic.py, which is the only caller. Defaults to a local
SQLite file so Phase 4 doesn't need a Postgres instance reachable from the
GPU compute node (which the Docker path in docker-compose.yml does provide,
via DATABASE_URL, for a closer-to-production deployment).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base

_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(database_url: str) -> None:
    global _engine, _SessionLocal
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    _engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("init_db() not called yet")
    return _SessionLocal()
