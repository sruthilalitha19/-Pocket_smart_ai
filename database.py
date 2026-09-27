"""
SQLite persistence layer for PocketSmart AI.

Stores users, recommendation history and uploaded outfit images. Uses the
standard library ``sqlite3`` driver so the project stays dependency-light.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from config import settings

_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    category     TEXT    NOT NULL,
    budget       REAL    NOT NULL,
    allocated    REAL    NOT NULL DEFAULT 0,
    remaining    REAL    NOT NULL DEFAULT 0,
    summary      TEXT    NOT NULL DEFAULT '',
    input_json   TEXT    NOT NULL DEFAULT '{}',
    result_json  TEXT    NOT NULL DEFAULT '{}',
    has_image    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reco_user ON recommendations (user_id, id DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    """Return a per-thread SQLite connection with row access by name."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        settings.ensure_directories()
        conn = sqlite3.connect(
            settings.database_path, check_same_thread=False, timeout=30
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _LOCAL.conn = conn
    return conn


def init_db() -> None:
    """Create tables if they do not exist yet."""
    conn = get_connection()
    with conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def create_user(username: str, email: str, password_hash: str) -> int:
    conn = get_connection()
    with conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username, email, password_hash, _now()),
        )
    return int(cur.lastrowid)


def get_user_by_username(username: str) -> sqlite3.Row | None:
    cur = get_connection().execute(
        "SELECT * FROM users WHERE lower(username) = lower(?)", (username,)
    )
    return cur.fetchone()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    cur = get_connection().execute(
        "SELECT * FROM users WHERE lower(email) = lower(?)", (email,)
    )
    return cur.fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    cur = get_connection().execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cur.fetchone()


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def save_recommendation(
    user_id: int,
    category: str,
    budget: float,
    allocated: float,
    remaining: float,
    summary: str,
    inputs: dict[str, Any],
    result: dict[str, Any],
    has_image: bool = False,
) -> int:
    conn = get_connection()
    with conn:
        cur = conn.execute(
            "INSERT INTO recommendations "
            "(user_id, category, budget, allocated, remaining, summary, "
            " input_json, result_json, has_image, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                category,
                float(budget),
                float(allocated),
                float(remaining),
                summary,
                json.dumps(inputs, default=str),
                json.dumps(result, default=str),
                1 if has_image else 0,
                _now(),
            ),
        )
    return int(cur.lastrowid)


def _derive_tags(category: str, inputs: dict[str, Any]) -> list[str]:
    """Build the small pill list rendered on history cards."""
    if category == "home":
        tags = [str(r).lower().replace(" ", "_") for r in inputs.get("rooms", [])]
        tags += [
            f"{inputs.get('lights', 0)} lights",
            f"{inputs.get('ceiling_fans', 0)} fans",
            f"{inputs.get('furniture', 0)} furniture",
        ]
    elif category == "party":
        tags = [str(n).lower() for n in (inputs.get("needs", []) or [])]
        tags.append("contingency")
    elif category == "jewelry":
        tags = ["style-matched", "occasion-ready"]
        if inputs.get("outfit_image"):
            tags.insert(0, "outfit analyzed")
    return [t for t in tags if t][:6]


def _row_to_dict(row: sqlite3.Row, with_result: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": row["id"],
        "user_id": row["user_id"],
        "category": row["category"],
        "budget": row["budget"],
        "allocated": row["allocated"],
        "remaining": row["remaining"],
        "summary": row["summary"],
        "has_image": bool(row["has_image"]),
        "created_at": row["created_at"],
        "inputs": json.loads(row["input_json"] or "{}"),
    }
    if with_result:
        data["result"] = json.loads(row["result_json"] or "{}")
    else:
        data["tags"] = _derive_tags(data["category"], data["inputs"])
    return data


def list_recommendations(
    user_id: int, category: str | None = None, limit: int = 60
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM recommendations WHERE user_id = ?"
    params: list[Any] = [user_id]
    if category:
        sql += " AND category = ?"
        params.append(category)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    rows = get_connection().execute(sql, params).fetchall()
    return [_row_to_dict(r, with_result=False) for r in rows]


def get_recommendation(
    reco_id: int, user_id: int | None = None
) -> dict[str, Any] | None:
    sql = "SELECT * FROM recommendations WHERE id = ?"
    params: list[Any] = [reco_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    row = get_connection().execute(sql, params).fetchone()
    return _row_to_dict(row, with_result=True) if row else None


def delete_recommendation(reco_id: int, user_id: int) -> bool:
    conn = get_connection()
    with conn:
        cur = conn.execute(
            "DELETE FROM recommendations WHERE id = ? AND user_id = ?",
            (reco_id, user_id),
        )
    return cur.rowcount > 0


def dashboard_stats(user_id: int) -> dict[str, Any]:
    """Aggregate numbers shown on the dashboard."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "       COALESCE(SUM(budget), 0) AS budget_sum, "
        "       COALESCE(SUM(allocated), 0) AS allocated_sum, "
        "       COALESCE(SUM(remaining), 0) AS remaining_sum "
        "FROM recommendations WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    by_category = conn.execute(
        "SELECT category, COUNT(*) AS n FROM recommendations "
        "WHERE user_id = ? GROUP BY category",
        (user_id,),
    ).fetchall()
    return {
        "total_plans": row["total"] or 0,
        "total_budget": row["budget_sum"] or 0.0,
        "total_allocated": row["allocated_sum"] or 0.0,
        "total_remaining": row["remaining_sum"] or 0.0,
        "by_category": {r["category"]: r["n"] for r in by_category},
    }


def seed_demo_data(user_id: int, entries: list[dict[str, Any]]) -> None:
    """Insert sample history rows so a fresh account is not empty."""
    for entry in entries:
        save_recommendation(
            user_id=user_id,
            category=entry["category"],
            budget=entry["budget"],
            allocated=entry.get("allocated", entry["budget"]),
            remaining=entry.get("remaining", 0.0),
            summary=entry.get("summary", ""),
            inputs=entry.get("inputs", {}),
            result=entry.get("result", {}),
            has_image=entry.get("has_image", False),
        )


def get_user_by_email(email: str) -> sqlite3.Row | None:
    cur = get_connection().execute(
        "SELECT * FROM users WHERE lower(email) = lower(?)", (email,)
    )
    return cur.fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    cur = get_connection().execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cur.fetchone()
