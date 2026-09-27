"""
Application configuration for PocketSmart AI.

Loads settings from environment variables (``.env`` is read by python-dotenv)
and exposes a single ``Settings`` instance used across the FastAPI app.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root == directory that contains this package.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime configuration for PocketSmart AI."""

    def __init__(self) -> None:
        # --- Paths -----------------------------------------------------
        self.base_dir = BASE_DIR
        self.static_dir = BASE_DIR / "static"
        self.uploads_dir = self.static_dir / "uploads"
        self.templates_dir = BASE_DIR / "templates"
        self.database_path = Path(
            os.getenv("DATABASE_PATH", str(BASE_DIR / "pocketsmart.db"))
        )

        # --- Gemini 1.5 Flash Pro --------------------------------------
        # The design document stores the key as ``Gemini_Api_Key``; both that
        # name and the conventional ``GEMINI_API_KEY`` are supported.
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("Gemini_Api_Key")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        # The design document names "Gemini 1.5 Flash Pro". The 1.5 family has
        # been retired, and 2.5 Flash is restricted for new projects, so the
        # default is the currently recommended Flash-tier model. Override with
        # GEMINI_MODEL if you want a different one.
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        # Tried in order when a model is unavailable, so a retired or renamed
        # model does not take the whole feature offline.
        self.gemini_model_fallbacks = [
            m.strip()
            for m in os.getenv(
                "GEMINI_MODEL_FALLBACKS",
                "gemini-3.5-flash,gemini-3.7-flash,gemini-flash-latest",
            ).split(",")
            if m.strip()
        ]

        # When true the deterministic local engine is used even if a key is
        # present. Handy for offline demos and automated tests.
        self.force_mock_ai = _as_bool(os.getenv("FORCE_MOCK_AI"), False)
        # Keep the worst case short: a planner page should not hang for
        # minutes. 2 attempts x 30s is the ceiling before we fall back.
        self.ai_timeout_seconds = float(os.getenv("AI_TIMEOUT_SECONDS", "30"))
        self.ai_max_attempts = max(1, int(os.getenv("AI_MAX_ATTEMPTS", "2")))
        self.ai_retry_backoff_seconds = float(
            os.getenv("AI_RETRY_BACKOFF_SECONDS", "1.5")
        )

        # --- Web -------------------------------------------------------
        self.app_name = os.getenv("APP_NAME", "PocketSmart AI")
        self.secret_key = os.getenv("SECRET_KEY", "pocketsmart-dev-secret-change-me")
        self.access_token_expire_minutes = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720")
        )
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]

    @property
    def ai_enabled(self) -> bool:
        """True when a Gemini key is present and mock mode is disabled."""
        return bool(self.gemini_api_key) and not self.force_mock_ai

    def ensure_directories(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
