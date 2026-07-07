"""
Shared-password authentication for the public API.

The dashboard is served from GitHub Pages (static, no server), so the API
is the only place a password check can actually be enforced — any client-
side gate alone would be decoration. Every endpoint except /api/health
(keep-warm pings), /api/auth/login and /static is denied without a valid
token.

Model:
  - The password lives in the APP_PASSWORD environment variable (a Hugging
    Face Space secret in production). It is never embedded in the frontend
    build and never logged.
  - POST /api/auth/login exchanges the password for a stateless, expiring
    token: "v1.<expiry_unix>.<hmac_sha256(key, 'v1.<expiry>')>". Stateless
    matters because the Space sleeps/restarts and wipes memory; tokens keep
    working across restarts as long as APP_PASSWORD is unchanged, and
    rotating the password instantly invalidates every issued token.
  - All comparisons are constant-time (hmac.compare_digest over fixed-size
    digests) so response timing leaks nothing about the password.
  - Login attempts are rate-limited per client IP to blunt brute force.

If APP_PASSWORD is unset (local dev, CI tests) the API runs open and logs
a loud warning at startup.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = 12 * 3600
TOKEN_VERSION = "v1"

# Paths that must stay reachable without a token. Everything else on the
# app — including /docs and /openapi.json — is denied by default, so new
# endpoints are protected automatically.
PUBLIC_PATHS = frozenset({"/", "/api/health", "/api/auth/login"})
PUBLIC_PREFIXES = ("/static",)

MAX_LOGIN_ATTEMPTS = 8
LOGIN_WINDOW_SECONDS = 300


def app_password() -> str | None:
    """Read at call time (not import time) so tests and the Space's secret
    manager can set it without an app restart-ordering headache."""
    password = os.environ.get("APP_PASSWORD", "").strip()
    return password or None


def _signing_key(password: str) -> bytes:
    return hashlib.sha256(f"anomaly-dashboard-token:{password}".encode()).digest()


def _sign(payload: str, password: str) -> str:
    return hmac.new(_signing_key(password), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(password: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> tuple[str, int]:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{TOKEN_VERSION}.{expires_at}"
    return f"{payload}.{_sign(payload, password)}", expires_at


def verify_token(token: str, password: str) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    version, expiry_str, signature = parts
    if version != TOKEN_VERSION or not expiry_str.isdigit():
        return False
    expected = _sign(f"{version}.{expiry_str}", password)
    if not hmac.compare_digest(expected, signature):
        return False
    return int(expiry_str) >= time.time()


def password_matches(candidate: str, password: str) -> bool:
    """Constant-time equality via fixed-length digests, so neither length
    nor prefix of the real password leaks through timing."""
    return hmac.compare_digest(
        hashlib.sha256(candidate.encode()).digest(),
        hashlib.sha256(password.encode()).digest(),
    )


class LoginRateLimiter:
    """Sliding-window failed-attempt limiter, per client IP, in memory.
    Single-process deployment (one uvicorn worker), so this is sufficient;
    a Space restart resets it, which only ever errs toward letting the
    legitimate user back in.
    """

    def __init__(self, max_attempts: int = MAX_LOGIN_ATTEMPTS, window_seconds: int = LOGIN_WINDOW_SECONDS):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, attempts: deque[float], now: float) -> None:
        while attempts and attempts[0] <= now - self.window_seconds:
            attempts.popleft()

    def allow(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            attempts = self._attempts.get(client_ip)
            if not attempts:
                return True
            self._prune(attempts, now)
            return len(attempts) < self.max_attempts

    def record_failure(self, client_ip: str) -> None:
        now = time.time()
        with self._lock:
            attempts = self._attempts.setdefault(client_ip, deque())
            self._prune(attempts, now)
            attempts.append(now)

    def clear(self, client_ip: str) -> None:
        with self._lock:
            self._attempts.pop(client_ip, None)


LOGIN_RATE_LIMITER = LoginRateLimiter()


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def request_is_authorized(authorization_header: str | None) -> bool:
    password = app_password()
    if password is None:
        return True  # open mode (no APP_PASSWORD configured)
    if not authorization_header:
        return False
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    return verify_token(token.strip(), password)


def warn_if_open() -> None:
    if app_password() is None:
        logger.warning(
            "APP_PASSWORD is not set — the API is running WITHOUT authentication. "
            "Set the APP_PASSWORD environment variable (Space secret) to require a password."
        )
