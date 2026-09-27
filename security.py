"""
Security helpers: password hashing and signed session tokens.

Uses PBKDF2-HMAC-SHA256 from the standard library for password hashing (no
native build step required) and ``itsdangerous``-style HMAC signing for the
opaque session token that the browser cookie carries.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time

from config import settings

_ITERATIONS = 260_000
_ALGO = "pbkdf2_sha256"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return "$".join(
        [
            _ALGO,
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_b64, digest_b64 = stored.split("$")
        if algo != _ALGO:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
        return hmac.compare_digest(expected, actual)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Signed tokens (session cookie + /token JSON Web Token style response)
# ---------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: bytes) -> str:
    return _b64e(
        hmac.new(settings.secret_key.encode(), payload, hashlib.sha256).digest()
    )


def create_access_token(subject: str | int, expires_minutes: int | None = None) -> str:
    """Create a compact signed token of the form ``payload.signature``."""
    minutes = expires_minutes or settings.access_token_expire_minutes
    body = {
        "sub": str(subject),
        "exp": int(time.time()) + minutes * 60,
        "jti": secrets.token_hex(8),
    }
    import json

    payload = _b64e(json.dumps(body, separators=(",", ":")).encode())
    return f"{payload}.{_sign(payload.encode())}"


def decode_access_token(token: str) -> dict | None:
    """Return token claims, or ``None`` when invalid/expired."""
    import json

    try:
        payload, signature = token.split(".")
    except ValueError:
        return None
    expected = _sign(payload.encode())
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        claims = json.loads(_b64d(payload))
    except (ValueError, TypeError):
        return None
    if claims.get("exp", 0) < time.time():
        return None
    return claims
