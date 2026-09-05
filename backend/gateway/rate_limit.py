"""Phase 13: in-memory sliding-window rate limiting, per client.

Client identity for rate-limiting purposes is whatever's in the Authorization
header (raw, unvalidated — a wrong/expired token still gets its own bucket
rather than falling back to a shared one) if present, else the connecting IP.
This is a coarser identity than auth's (which validates the token) — good
enough to stop one client from hammering the server, not meant as an
authentication signal itself.

In-memory, not Redis-backed: correct for this project's single-process
uvicorn deployment (see scripts/sbatch_serve.sh — one process, one GPU worth
of engines). A multi-worker deployment would need a shared store instead,
since each worker would otherwise track its own independent limit.
"""

import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from backend.core.config import get_settings

_buckets: dict[str, deque] = defaultdict(deque)


def _client_identity(request: Request, authorization: str | None) -> str:
    if authorization:
        return authorization
    if request.client:
        return request.client.host
    return "unknown"


async def enforce_rate_limit(request: Request, authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    limit = settings.rate_limit_requests
    window = settings.rate_limit_window_seconds
    if limit <= 0:
        return  # rate limiting disabled

    identity = _client_identity(request, authorization)
    now = time.monotonic()
    bucket = _buckets[identity]

    while bucket and now - bucket[0] > window:
        bucket.popleft()

    if len(bucket) >= limit:
        retry_after = max(0, window - (now - bucket[0]))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"rate limit exceeded: {limit} requests per {window}s",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    bucket.append(now)
