"""Auth: static API keys (Phase 1) or short-lived JWTs issued from one
(Phase 13). Disabled entirely when SERVELLM_API_KEYS is unset, so Phase 1
still works out of the box with zero config.

JWTs don't add a second layer of trust here — issuing one still requires a
valid API key up front (see create_access_token / POST /v1/admin/auth/token
in backend/gateway/main.py). What they add: a credential with a built-in
expiry that can be handed to something less trusted than the raw API key
without that raw key ever leaving wherever it's actually stored.
"""

import time

import jwt
from fastapi import Header, HTTPException, status

from backend.core.config import Settings, get_settings


def create_access_token(settings: Settings) -> tuple[str, int]:
    """Requires a caller to already hold a valid API key (checked by
    POST /v1/admin/auth/token before calling this) — this just encodes that
    fact into a token with an expiry, it doesn't independently authenticate
    anyone."""
    now = int(time.time())
    expires_in = settings.jwt_expiry_seconds
    payload = {"iat": now, "exp": now + expires_in, "sub": "servellm-api-key-holder"}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def _verify_jwt(token: str, settings: Settings) -> bool:
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return True
    except jwt.InvalidTokenError:
        return False


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    allowed = settings.allowed_api_keys()
    if not allowed:
        return  # auth disabled

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()

    if token in allowed:
        return
    if _verify_jwt(token, settings):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key or token")
