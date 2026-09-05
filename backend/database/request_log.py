"""CRUD over the RequestLog table — Phase 13's request history. Called from
backend/gateway/main.py's completion endpoints (write) and the
GET /v1/admin/requests endpoint (read)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import RequestLog


def log_request(
    session: Session,
    *,
    request_id: str,
    model: str,
    adapter: str | None,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_seconds: float,
    ttft_seconds: float | None,
    status: str,
) -> None:
    row = RequestLog(
        request_id=request_id,
        model=model,
        adapter=adapter,
        endpoint=endpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_seconds=latency_seconds,
        ttft_seconds=ttft_seconds,
        status=status,
    )
    session.add(row)
    session.commit()


def list_requests(
    session: Session,
    *,
    model: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[RequestLog]:
    stmt = select(RequestLog).order_by(RequestLog.created_at.desc()).limit(limit)
    if model is not None:
        stmt = stmt.where(RequestLog.model == model)
    if status is not None:
        stmt = stmt.where(RequestLog.status == status)
    return list(session.execute(stmt).scalars().all())
