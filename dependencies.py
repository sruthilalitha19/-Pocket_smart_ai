"""
FastAPI dependencies: session/cookie auth, form parsing and upload guards.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import database
import security
from config import settings

SESSION_COOKIE = "pocketsmart_session"

# Optional bearer support for the JSON API; pages use the cookie instead.
bearer_scheme = HTTPBearer(auto_error=False)

# Upload guard rails.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")


# ---------------------------------------------------------------------------
# Current user
# ---------------------------------------------------------------------------
def _user_from_token(token: str | None):
    if not token:
        return None
    claims = security.decode_access_token(token)
    if not claims:
        return None
    try:
        user_id = int(claims["sub"])
    except (KeyError, TypeError, ValueError):
        return None
    return database.get_user_by_id(user_id)


def get_current_user_optional(request: Request):
    """Resolve the signed-in user, or ``None`` for anonymous visitors."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        auth = getattr(request.state, "auth", None)
        token = auth
    return _user_from_token(token)


def get_current_user(
    user=Depends(get_current_user_optional),
) -> Any:
    """Require a signed-in user; redirect targets handle the 401 themselves."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return user


async def get_bearer_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    """Resolve a user from an ``Authorization: Bearer`` header."""
    if credentials is None:
        return None
    return _user_from_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Session cookie helpers
# ---------------------------------------------------------------------------
def set_session_cookie(response, user_id: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=security.create_access_token(user_id),
        max_age=settings.access_token_expire_minutes * 60,
        httponly=True,
        samesite="lax",
        secure=False,  # set True behind HTTPS in production
        path="/",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


# ---------------------------------------------------------------------------
# Form parsing helpers
# ---------------------------------------------------------------------------
def as_int(value: str | None, default: int = 0, minimum: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def as_float(value: str | None, default: float = 0.0, minimum: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip().replace(",", "").replace("₹", ""))
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def checkbox_list(values: list[str] | None) -> list[str]:
    """Keep only recognised options from a checkbox group."""
    return [v for v in (values or []) if isinstance(v, str) and v.strip()]


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------
async def save_outfit_image(
    file: UploadFile | None = File(None),
) -> tuple[str, bytes] | None:
    """
    Validate and store an uploaded outfit photo.

    Returns ``(filename, raw_bytes)`` or ``None`` when no file was sent.
    Raises :class:`HTTPException` for unsupported or oversized uploads.
    """
    if file is None or not file.filename:
        return None

    content_type = (file.content_type or "").lower().strip()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG, PNG or WebP images are allowed.",
        )

    data = await file.read()
    if not data:
        return None
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image is larger than the 5 MB limit.",
        )

    # Confirm the bytes really are an image (magic-number sniff).
    if not _looks_like_image(data, content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That file is not a valid image.",
        )

    settings.ensure_directories()
    ext = ALLOWED_IMAGE_TYPES[content_type]
    stem = _SAFE_STEM.sub("-", Path(file.filename).stem)[:40] or "outfit"
    filename = f"{stem}-{secrets.token_hex(6)}{ext}"
    (settings.uploads_dir / filename).write_bytes(data)
    return filename, data


_MAGIC = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/jpg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/webp": b"RIFF",
}


def _looks_like_image(data: bytes, content_type: str) -> bool:
    magic = _MAGIC.get(content_type)
    if not magic or not data.startswith(magic):
        return False
    if content_type == "image/webp":
        # RIFF header is followed by the WEBP fourcc at offset 8.
        return len(data) >= 12 and data[8:12] == b"WEBP"
    return True
