"""Persistence layer for provenance analysis results.

Provides a :class:`AnalysisRepository` protocol and two concrete backends:

- :class:`SqliteRepository` — default for local development.
- :class:`PostgresRepository` — activated via ``DATABASE_URL``.

Both use ``analyses``, ``api_keys``, and ``usage_records`` tables.  Route
handlers depend only on the protocol, so swapping backends requires no
changes to API logic.

Configuration
-------------
``DATABASE_URL`` env var controls backend selection:

- unset / empty  → SQLite (``data/provenance.db``)
- ``sqlite://…``  → SQLite at the given path
- ``postgresql://…`` or ``postgres://…`` → PostgreSQL
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("provenance.api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_hash(text: str) -> str:
    """SHA-256 of the input text (used for deduplication, text is NOT stored)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _locked_method(fn: Any) -> Any:
    """Serialize one repository method on the instance lock."""
    import functools
    import threading

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock: threading.Lock = self._lock
        with lock:
            return fn(self, *args, **kwargs)

    return wrapper


def _synchronize_repository(cls: Any) -> Any:
    """Apply :func:`_locked_method` to every public method of a repository.

    Connections are shared across request and background-worker threads, so
    each method body must run atomically — otherwise concurrent writes on
    one connection corrupt each other's implicit transactions. Methods
    never call each other, so a single non-reentrant instance lock cannot
    deadlock.
    """
    for name, member in list(vars(cls).items()):
        if name.startswith("_") or not callable(member):
            continue
        setattr(cls, name, _locked_method(member))
    return cls


def _benchmark_run_to_dict(row: Any) -> dict[str, Any]:
    """Map a benchmark_runs row (SQLite Row or plain dict) to a run dict."""
    result: dict[str, Any] = {
        "run_id": row["run_id"],
        "key_id": row["key_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "config": json.loads(row["config_json"]),
        "out_dir": row["out_dir"],
        "progress": json.loads(row["progress_json"]) if row["progress_json"] else None,
        "error_message": row["error_message"],
        "duration_ms": row["duration_ms"],
        "retry_count": row["retry_count"],
    }
    if row["result_json"]:
        result["result"] = json.loads(row["result_json"])
    return result


def _subscription_to_dict(row: Any) -> dict[str, Any]:
    """Return only application-safe subscription metadata."""
    if not hasattr(row, "keys"):
        row = dict(zip((
            "id", "user_id", "provider", "provider_customer_id",
            "provider_subscription_id", "provider_price_id", "status",
            "current_period_start", "current_period_end", "cancel_at_period_end",
            "created_at", "updated_at",
        ), row))
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "provider": row["provider"],
        "provider_customer_id": row["provider_customer_id"],
        "provider_subscription_id": row["provider_subscription_id"],
        "provider_price_id": row["provider_price_id"],
        "status": row["status"],
        "current_period_start": row["current_period_start"],
        "current_period_end": row["current_period_end"],
        "cancel_at_period_end": bool(row["cancel_at_period_end"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class AnalysisRepository(Protocol):
    """Storage interface for analysis results, API keys, and usage."""

    # -- analyses --

    def save(
        self,
        *,
        text: str,
        engine_version: str,
        detectors: list[str],
        result_dict: dict[str, Any],
    ) -> str: ...

    def get(self, analysis_id: str) -> dict[str, Any] | None: ...

    def list_analyses(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]: ...

    # -- api keys --

    def create_api_key(
        self, *, key_id: str, key_hash: str, name: str
    ) -> None: ...

    def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None: ...

    def get_api_key_by_id(self, key_id: str) -> dict[str, Any] | None: ...

    def list_api_keys(self) -> list[dict[str, Any]]: ...

    def revoke_api_key(self, key_id: str) -> None: ...

    def update_api_key_last_used(self, key_id: str) -> None: ...

    # -- usage --

    def record_usage(
        self,
        *,
        key_id: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        character_count: int,
        success: bool,
        detector: str | None = None,
    ) -> None: ...

    def get_usage_summary(
        self,
        *,
        key_id: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]: ...

    def get_usage_today(self, key_id: str) -> tuple[int, int]: ...

    def get_usage_today_jobs(self, key_id: str) -> tuple[int, int]: ...

    def get_avg_duration(self) -> float | None: ...

    def get_usage_by_status(self) -> dict[str, int]: ...

    def commit_if_needed(self) -> None: ...

    # -- jobs --

    def create_job(
        self,
        *,
        job_id: str,
        key_id: str,
        detectors: list[str],
        config_path: str | None,
        input_hash: str,
        character_count: int,
        retry_count: int = 0,
        parent_job_id: str | None = None,
    ) -> None: ...

    def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    def update_job_status(
        self, *,
        job_id: str,
        status: str,
        started_at: str | None = None,
    ) -> None: ...

    def complete_job(
        self, *,
        job_id: str,
        result_json: str,
        duration_ms: float,
    ) -> None: ...

    def fail_job(
        self, *,
        job_id: str,
        error_message: str,
        duration_ms: float,
    ) -> None: ...

    def list_jobs_by_key(
        self, *,
        key_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]: ...

    def cancel_job_if_status(self, job_id: str, expected_status: str, new_status: str) -> bool: ...

    def get_job_counts(self) -> dict[str, int]: ...

    def cleanup_jobs(self, older_than: str) -> int: ...

    def recover_stale_jobs(self) -> int: ...

    def create_benchmark_run(
        self,
        *,
        run_id: str,
        key_id: str,
        config_json: str,
        out_dir: str,
    ) -> None: ...

    def get_benchmark_run(self, run_id: str) -> dict[str, Any] | None: ...

    def list_benchmark_runs(
        self,
        *,
        key_id: str,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]: ...

    def update_benchmark_run(
        self,
        *,
        run_id: str,
        status: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        progress_json: str | None = None,
        result_json: str | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
        retry_count: int | None = None,
        clear_attempt_state: bool = False,
    ) -> None: ...

    def complete_benchmark_run(
        self,
        *,
        run_id: str,
        result_json: str,
        duration_ms: float,
    ) -> None: ...

    def fail_benchmark_run(
        self,
        *,
        run_id: str,
        error_message: str,
        duration_ms: float,
    ) -> None: ...

    def cancel_benchmark_run_if_status(
        self, run_id: str, expected_status: str, new_status: str
    ) -> bool: ...

    def recover_stale_benchmark_runs(self) -> int: ...

    # -- SaaS identity / quota -------------------------------------------
    def get_or_create_visitor(self, visitor_id: str, abuse_hash: str | None) -> dict[str, Any]: ...
    def get_or_create_user(self, auth_user_id: str, email: str | None) -> dict[str, Any]: ...
    def set_user_plan(self, auth_user_id: str, plan: str) -> dict[str, Any] | None: ...
    def claim_visitor_events(self, visitor_id: str, auth_user_id: str) -> int: ...
    def count_public_uses_today(
        self, *, visitor_id: str | None, auth_user_id: str | None
    ) -> int: ...
    def try_consume_public_use(
        self, *, visitor_id: str | None, auth_user_id: str | None, chars: int, limit: int
    ) -> tuple[bool, int]: ...

    # -- Billing ---------------------------------------------------------
    def get_subscription_for_user(self, user_id: str) -> dict[str, Any] | None: ...
    def get_subscription_by_provider_customer(
        self, provider: str, provider_customer_id: str
    ) -> dict[str, Any] | None: ...
    def upsert_subscription(
        self, *, user_id: str, provider: str, provider_customer_id: str,
        provider_subscription_id: str | None, provider_price_id: str | None,
        status: str, current_period_start: str | None,
        current_period_end: str | None, cancel_at_period_end: bool,
    ) -> dict[str, Any]: ...
    def record_webhook_event(
        self, *, provider: str, event_id: str, event_type: str
    ) -> bool: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# SQLite backend (default)
# ---------------------------------------------------------------------------

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id    TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    text_hash      TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    token_count    INTEGER NOT NULL,
    status         TEXT NOT NULL,
    detectors      TEXT NOT NULL,
    result_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_text_hash ON analyses(text_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id      TEXT PRIMARY KEY,
    key_hash    TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL,
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS usage_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id         TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    status_code    INTEGER NOT NULL,
    duration_ms    REAL NOT NULL,
    character_count INTEGER NOT NULL,
    success        INTEGER NOT NULL,
    detector       TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_key_id ON usage_records(key_id);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp);

CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    key_id         TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    completed_at   TEXT,
    input_hash     TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    detectors      TEXT NOT NULL,
    config_path    TEXT,
    result_json    TEXT,
    error_message  TEXT,
    duration_ms    REAL,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    parent_job_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_key_id ON jobs(key_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

# Benchmark-run table DDL shared by both backends (idempotent).
_SCHEMA_BENCHMARK_RUNS = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id         TEXT PRIMARY KEY,
    key_id         TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    completed_at   TEXT,
    config_json    TEXT NOT NULL,
    out_dir        TEXT NOT NULL,
    progress_json  TEXT,
    result_json    TEXT,
    error_message  TEXT,
    duration_ms    REAL,
    retry_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_key_id ON benchmark_runs(key_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_status ON benchmark_runs(status);
"""

# Identity / SaaS table DDL shared by both backends (idempotent).
# No raw text, no raw IPs, no secrets are stored here — only opaque IDs,
# HMAC-hashed coarse abuse signals, aggregate counts, and plan labels.
# Timestamps are UTC ISO-8601; the daily quota boundary is the UTC date
# prefix (``created_at >= YYYY-MM-DD``).
_SCHEMA_IDENTITY = """
CREATE TABLE IF NOT EXISTS app_users (
    id            TEXT PRIMARY KEY,
    auth_user_id  TEXT NOT NULL UNIQUE,
    email         TEXT,
    plan          TEXT NOT NULL DEFAULT 'free',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS anonymous_visitors (
    visitor_id    TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    abuse_hash    TEXT
);
CREATE TABLE IF NOT EXISTS usage_events (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    visitor_id    TEXT,
    user_id       TEXT,
    chars         INTEGER NOT NULL DEFAULT 0,
    success       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_usage_events_visitor ON usage_events(visitor_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_events_user ON usage_events(user_id, created_at);
"""

# Billing tables are shared, normalized application tables. They hold only
# provider identifiers and subscription metadata—not payment methods, card
# numbers, CVVs, or raw webhook payloads.
_SCHEMA_BILLING = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id                       TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    provider                 TEXT NOT NULL,
    provider_customer_id     TEXT NOT NULL,
    provider_subscription_id TEXT,
    provider_price_id        TEXT,
    status                   TEXT NOT NULL,
    current_period_start     TEXT,
    current_period_end       TEXT,
    cancel_at_period_end     INTEGER NOT NULL DEFAULT 0,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    UNIQUE(provider, provider_subscription_id)
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(provider, provider_customer_id);

CREATE TABLE IF NOT EXISTS billing_webhook_events (
    id           TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    received_at  TEXT NOT NULL,
    processed_at TEXT,
    status       TEXT NOT NULL DEFAULT 'received',
    UNIQUE(provider, id)
);
CREATE INDEX IF NOT EXISTS idx_billing_webhook_events_received
    ON billing_webhook_events(provider, received_at);
"""

# Migration SQL for existing databases (safe to run repeatedly)
_MIGRATE_SQLITE = """
ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN parent_job_id TEXT;
"""


def _make_instance_lock() -> Any:
    import threading
    return threading.Lock()


@_synchronize_repository
class SqliteRepository:
    """SQLite-backed repository for local development."""

    def __init__(self, db_path: str | Path = "data/provenance.db") -> None:
        self._lock = _make_instance_lock()
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQLITE)
        self._conn.executescript(_SCHEMA_BENCHMARK_RUNS)
        self._conn.executescript(_SCHEMA_IDENTITY)
        self._conn.executescript(_SCHEMA_BILLING)
        self._migrate()

    def _migrate(self) -> None:
        """Apply schema migrations for existing databases."""
        for stmt in _MIGRATE_SQLITE.strip().split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save(
        self,
        *,
        text: str,
        engine_version: str,
        detectors: list[str],
        result_dict: dict[str, Any],
    ) -> str:
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        text_stats = result_dict.get("text_stats", {})
        self._conn.execute(
            "INSERT INTO analyses "
            "(analysis_id, timestamp, engine_version, text_hash, "
            "character_count, token_count, status, detectors, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis_id,
                timestamp,
                engine_version,
                _text_hash(text),
                text_stats.get("character_count", 0),
                text_stats.get("token_count", 0),
                result_dict.get("status", "unknown"),
                json.dumps(detectors),
                json.dumps(result_dict),
            ),
        )
        self._conn.commit()
        return analysis_id

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "analysis_id": row["analysis_id"],
            "timestamp": row["timestamp"],
            "engine_version": row["engine_version"],
            "text_hash": row["text_hash"],
            "character_count": row["character_count"],
            "token_count": row["token_count"],
            "status": row["status"],
            "detectors": json.loads(row["detectors"]),
            "result": json.loads(row["result_json"]),
        }

    def list_analyses(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        total = self._conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        rows = self._conn.execute(
            "SELECT analysis_id, timestamp, engine_version, text_hash, "
            "character_count, token_count, status, detectors "
            "FROM analyses ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        summaries = []
        for row in rows:
            summaries.append({
                "analysis_id": row["analysis_id"],
                "timestamp": row["timestamp"],
                "engine_version": row["engine_version"],
                "text_hash": row["text_hash"],
                "character_count": row["character_count"],
                "token_count": row["token_count"],
                "status": row["status"],
                "detector_count": len(json.loads(row["detectors"])),
            })
        return summaries, total

    # -- api keys --

    def create_api_key(
        self, *, key_id: str, key_hash: str, name: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO api_keys (key_id, key_hash, name, status, created_at) "
            "VALUES (?, ?, ?, 'active', ?)",
            (key_id, key_hash, name, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if row is None:
            return None
        return {
            "key_id": row["key_id"],
            "key_hash": row["key_hash"],
            "name": row["name"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def get_api_key_by_id(self, key_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "key_id": row["key_id"],
            "key_hash": row["key_hash"],
            "name": row["name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
        }

    def list_api_keys(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT key_id, name, status, created_at FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
        return [{"key_id": r["key_id"], "name": r["name"], "status": r["status"], "created_at": r["created_at"]} for r in rows]

    def revoke_api_key(self, key_id: str) -> None:
        self._conn.execute("UPDATE api_keys SET status = 'revoked' WHERE key_id = ?", (key_id,))
        self._conn.commit()

    def update_api_key_last_used(self, key_id: str) -> None:
        self._conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
            (datetime.now(timezone.utc).isoformat(), key_id),
        )
        self._conn.commit()

    # -- usage --

    def record_usage(
        self,
        *,
        key_id: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        character_count: int,
        success: bool,
        detector: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO usage_records "
            "(key_id, endpoint, timestamp, status_code, duration_ms, "
            "character_count, success, detector) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key_id,
                endpoint,
                datetime.now(timezone.utc).isoformat(),
                status_code,
                duration_ms,
                character_count,
                1 if success else 0,
                detector,
            ),
        )
        self._conn.commit()

    def get_usage_summary(
        self, *, key_id: str | None = None, since: str | None = None
    ) -> dict[str, Any]:
        conditions = []
        params: list[Any] = []
        if key_id:
            conditions.append("key_id = ?")
            params.append(key_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        total = self._conn.execute(f"SELECT COUNT(*) FROM usage_records{where}", params).fetchone()[0]
        successful = self._conn.execute(
            f"SELECT COUNT(*) FROM usage_records{where} AND success = 1" if where else "SELECT COUNT(*) FROM usage_records WHERE success = 1",
            params,
        ).fetchone()[0]
        failed = total - successful
        total_chars = self._conn.execute(
            f"SELECT COALESCE(SUM(character_count), 0) FROM usage_records{where}", params
        ).fetchone()[0]

        # By endpoint
        ep_rows = self._conn.execute(
            f"SELECT endpoint, COUNT(*) FROM usage_records{where} GROUP BY endpoint", params
        ).fetchall()
        by_endpoint = {r[0]: r[1] for r in ep_rows}

        # By detector
        det_where = (where + " AND") if where else "WHERE"
        det_rows = self._conn.execute(
            f"SELECT detector, COUNT(*) FROM usage_records {det_where} detector IS NOT NULL GROUP BY detector", params
        ).fetchall()
        by_detector = {r[0]: r[1] for r in det_rows}

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "total_characters": total_chars,
            "by_endpoint": by_endpoint,
            "by_detector": by_detector,
        }

    def get_usage_today(self, key_id: str) -> tuple[int, int]:
        """Return (request_count, character_count) for today for the given key."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(character_count), 0) "
            "FROM usage_records WHERE key_id = ? AND timestamp >= ?",
            (key_id, today),
        ).fetchone()
        return (row[0], row[1])

    def get_usage_today_jobs(self, key_id: str) -> tuple[int, int]:
        """Count queued/running jobs for today (for daily limit enforcement)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(character_count), 0) "
            "FROM jobs WHERE key_id = ? AND status IN ('queued', 'running') "
            "AND created_at >= ?",
            (key_id, today),
        ).fetchone()
        return (row[0], row[1])

    def get_avg_duration(self) -> float | None:
        row = self._conn.execute(
            "SELECT AVG(duration_ms) FROM usage_records"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return round(row[0], 2)

    def get_usage_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status_code, COUNT(*) FROM usage_records GROUP BY status_code"
        ).fetchall()
        return {str(r[0]): r[1] for r in rows}

    def commit_if_needed(self) -> None:
        """Release any implicit transaction started by SELECTs."""
        try:
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # No active transaction — nothing to do

    # -- jobs --

    def create_job(
        self,
        *,
        job_id: str,
        key_id: str,
        detectors: list[str],
        config_path: str | None,
        input_hash: str,
        character_count: int,
        retry_count: int = 0,
        parent_job_id: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO jobs "
            "(job_id, key_id, status, created_at, input_hash, "
            "character_count, detectors, config_path, retry_count, parent_job_id) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                key_id,
                datetime.now(timezone.utc).isoformat(),
                input_hash,
                character_count,
                json.dumps(detectors),
                config_path,
                retry_count,
                parent_job_id,
            ),
        )
        self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        result = {
            "job_id": row["job_id"],
            "key_id": row["key_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "input_hash": row["input_hash"],
            "character_count": row["character_count"],
            "detectors": json.loads(row["detectors"]),
            "config_path": row["config_path"],
            "error_message": row["error_message"],
            "duration_ms": row["duration_ms"],
            "retry_count": row["retry_count"],
            "parent_job_id": row["parent_job_id"],
        }
        if row["result_json"]:
            result["result"] = json.loads(row["result_json"])
        return result

    def update_job_status(
        self,
        *,
        job_id: str,
        status: str,
        started_at: str | None = None,
    ) -> None:
        if started_at:
            self._conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE job_id = ?",
                (status, started_at, job_id),
            )
        else:
            self._conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (status, job_id),
            )
        self._conn.commit()

    def complete_job(
        self,
        *,
        job_id: str,
        result_json: str,
        duration_ms: float,
    ) -> None:
        self._conn.execute(
            "UPDATE jobs SET status = 'completed', completed_at = ?, "
            "result_json = ?, duration_ms = ? WHERE job_id = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                result_json,
                duration_ms,
                job_id,
            ),
        )
        self._conn.commit()

    def fail_job(
        self,
        *,
        job_id: str,
        error_message: str,
        duration_ms: float,
    ) -> None:
        self._conn.execute(
            "UPDATE jobs SET status = 'failed', completed_at = ?, "
            "error_message = ?, duration_ms = ? WHERE job_id = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                error_message,
                duration_ms,
                job_id,
            ),
        )
        self._conn.commit()

    def list_jobs_by_key(
        self,
        *,
        key_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE key_id = ?", (key_id,)
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT job_id, key_id, status, created_at, started_at, "
            "completed_at, input_hash, character_count, detectors, "
            "error_message, duration_ms, retry_count, parent_job_id "
            "FROM jobs WHERE key_id = ? ORDER BY created_at DESC "
            "LIMIT ? OFFSET ?",
            (key_id, limit, offset),
        ).fetchall()
        summaries = []
        for row in rows:
            summaries.append({
                "job_id": row["job_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "character_count": row["character_count"],
                "detectors": json.loads(row["detectors"]),
                "error_message": row["error_message"],
                "duration_ms": row["duration_ms"],
                "retry_count": row["retry_count"],
                "parent_job_id": row["parent_job_id"],
            })
        return summaries, total

    def cancel_job_if_status(self, job_id: str, expected_status: str, new_status: str) -> bool:
        """Atomically transition a job's status only if it matches expected_status.

        Returns True if the transition was made, False otherwise.
        For 'cancelled' status, also sets completed_at and error_message.
        """
        if new_status == "cancelled":
            cursor = self._conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, error_message = 'Job cancelled by user' "
                "WHERE job_id = ? AND status = ?",
                (new_status, datetime.now(timezone.utc).isoformat(), job_id, expected_status),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ? AND status = ?",
                (new_status, job_id, expected_status),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_job_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        return {
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "cancelled": counts.get("cancelled", 0),
            "cancellation_requested": counts.get("cancellation_requested", 0),
        }

    def cleanup_jobs(self, older_than: str) -> int:
        cursor = self._conn.execute(
            "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'cancelled') "
            "AND completed_at < ?",
            (older_than,),
        )
        self._conn.commit()
        return cursor.rowcount

    def recover_stale_jobs(self) -> int:
        """Mark jobs stuck in 'running' or 'queued' as failed (crash recovery).

        Returns the number of jobs recovered.
        """
        now = datetime.now(timezone.utc).isoformat()
        # Running jobs: worker died before completing
        cursor = self._conn.execute(
            "UPDATE jobs SET status = 'failed', completed_at = ?, "
            "error_message = 'Job interrupted by application restart', "
            "duration_ms = 0 "
            "WHERE status = 'running'",
            (now,),
        )
        running_count = cursor.rowcount
        # Queued jobs: worker never started (or was killed before pickup)
        cursor2 = self._conn.execute(
            "UPDATE jobs SET status = 'failed', completed_at = ?, "
            "error_message = 'Job interrupted by application restart', "
            "duration_ms = 0 "
            "WHERE status = 'queued'",
            (now,),
        )
        queued_count = cursor2.rowcount
        self._conn.commit()
        return running_count + queued_count

    # ------------------------------------------------------------------
    # Benchmark runs (Phase 6C)
    # ------------------------------------------------------------------

    def create_benchmark_run(
        self,
        *,
        run_id: str,
        key_id: str,
        config_json: str,
        out_dir: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO benchmark_runs "
            "(run_id, key_id, status, created_at, config_json, out_dir) "
            "VALUES (?, ?, 'queued', ?, ?, ?)",
            (
                run_id,
                key_id,
                datetime.now(timezone.utc).isoformat(),
                config_json,
                out_dir,
            ),
        )
        self._conn.commit()

    def get_benchmark_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _benchmark_run_to_dict(row)

    def list_benchmark_runs(
        self,
        *,
        key_id: str,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        if status is not None:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM benchmark_runs WHERE key_id = ? AND status = ?",
                (key_id, status),
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT run_id, key_id, status, created_at, started_at, "
                "completed_at, config_json, out_dir, progress_json, "
                "error_message, duration_ms, retry_count "
                "FROM benchmark_runs WHERE key_id = ? AND status = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (key_id, status, limit, offset),
            ).fetchall()
        else:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM benchmark_runs WHERE key_id = ?", (key_id,)
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT run_id, key_id, status, created_at, started_at, "
                "completed_at, config_json, out_dir, progress_json, "
                "error_message, duration_ms, retry_count "
                "FROM benchmark_runs WHERE key_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (key_id, limit, offset),
            ).fetchall()
        summaries = []
        for row in rows:
            summaries.append({
                "run_id": row["run_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "config": json.loads(row["config_json"]),
                "out_dir": row["out_dir"],
                "progress": json.loads(row["progress_json"]) if row["progress_json"] else None,
                "error_message": row["error_message"],
                "duration_ms": row["duration_ms"],
                "retry_count": row["retry_count"],
            })
        return summaries, total

    def update_benchmark_run(
        self,
        *,
        run_id: str,
        status: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        progress_json: str | None = None,
        result_json: str | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
        retry_count: int | None = None,
        clear_attempt_state: bool = False,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if clear_attempt_state:
            # Retry: drop the previous attempt's timestamps/error so the
            # queued run does not display stale terminal state.
            updates.extend([
                "started_at = NULL",
                "completed_at = NULL",
                "error_message = NULL",
            ])
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if started_at is not None:
            updates.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if progress_json is not None:
            updates.append("progress_json = ?")
            params.append(progress_json)
        if result_json is not None:
            updates.append("result_json = ?")
            params.append(result_json)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if duration_ms is not None:
            updates.append("duration_ms = ?")
            params.append(duration_ms)
        if retry_count is not None:
            updates.append("retry_count = ?")
            params.append(retry_count)
        if not updates:
            return
        params.append(run_id)
        self._conn.execute(
            f"UPDATE benchmark_runs SET {', '.join(updates)} WHERE run_id = ?",
            params,
        )
        self._conn.commit()

    def complete_benchmark_run(
        self,
        *,
        run_id: str,
        result_json: str,
        duration_ms: float,
    ) -> None:
        self._conn.execute(
            "UPDATE benchmark_runs SET status = 'completed', completed_at = ?, "
            "result_json = ?, duration_ms = ? WHERE run_id = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                result_json,
                duration_ms,
                run_id,
            ),
        )
        self._conn.commit()

    def fail_benchmark_run(
        self,
        *,
        run_id: str,
        error_message: str,
        duration_ms: float,
    ) -> None:
        self._conn.execute(
            "UPDATE benchmark_runs SET status = 'failed', completed_at = ?, "
            "error_message = ?, duration_ms = ? WHERE run_id = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                error_message,
                duration_ms,
                run_id,
            ),
        )
        self._conn.commit()

    def cancel_benchmark_run_if_status(
        self, run_id: str, expected_status: str, new_status: str
    ) -> bool:
        """Atomically transition a run's status only if it matches expected_status."""
        if new_status == "cancelled":
            cursor = self._conn.execute(
                "UPDATE benchmark_runs SET status = ?, completed_at = ?, "
                "error_message = 'Benchmark run cancelled by user' "
                "WHERE run_id = ? AND status = ?",
                (new_status, datetime.now(timezone.utc).isoformat(), run_id, expected_status),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE benchmark_runs SET status = ? WHERE run_id = ? AND status = ?",
                (new_status, run_id, expected_status),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def recover_stale_benchmark_runs(self) -> int:
        """Crash recovery: interrupted runs become failed, never successful.

        Running/queued runs are marked failed (artifacts preserved on disk;
        retry resumes via the run manifest). Runs awaiting cancellation are
        marked cancelled. Returns the number of runs recovered.
        """
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "UPDATE benchmark_runs SET status = 'failed', completed_at = ?, "
            "error_message = 'Benchmark run interrupted by application restart', "
            "duration_ms = 0 "
            "WHERE status IN ('running', 'queued')",
            (now,),
        )
        interrupted = cursor.rowcount
        cursor2 = self._conn.execute(
            "UPDATE benchmark_runs SET status = 'cancelled', completed_at = ?, "
            "error_message = 'Benchmark run cancelled by user' "
            "WHERE status = 'cancellation_requested'",
            (now,),
        )
        self._conn.commit()
        return interrupted + cursor2.rowcount

    # -- SaaS identity / quota -------------------------------------------
    # The whole repository is serialized on the instance lock, so each
    # check-then-insert below is atomic within this process.

    @staticmethod
    def _utc_today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_or_create_visitor(self, visitor_id: str, abuse_hash: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        row = self._conn.execute(
            "SELECT visitor_id, created_at FROM anonymous_visitors WHERE visitor_id = ?",
            (visitor_id,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO anonymous_visitors (visitor_id, created_at, last_seen_at, abuse_hash) "
                "VALUES (?, ?, ?, ?)",
                (visitor_id, now, now, abuse_hash),
            )
            self._conn.commit()
            return {"visitor_id": visitor_id, "created_at": now, "new": True}
        self._conn.execute(
            "UPDATE anonymous_visitors SET last_seen_at = ?, abuse_hash = COALESCE(?, abuse_hash) "
            "WHERE visitor_id = ?",
            (now, abuse_hash, visitor_id),
        )
        self._conn.commit()
        return {"visitor_id": visitor_id, "created_at": row[1], "new": False}

    def get_or_create_user(self, auth_user_id: str, email: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        row = self._conn.execute(
            "SELECT id, auth_user_id, email, plan, created_at FROM app_users WHERE auth_user_id = ?",
            (auth_user_id,),
        ).fetchone()
        if row is None:
            uid = str(uuid.uuid4())
            self._conn.execute(
                "INSERT INTO app_users (id, auth_user_id, email, plan, created_at, updated_at) "
                "VALUES (?, ?, ?, 'free', ?, ?)",
                (uid, auth_user_id, email, now, now),
            )
            self._conn.commit()
            return {"id": uid, "auth_user_id": auth_user_id, "email": email, "plan": "free", "new": True}
        if email and email != row[2]:
            self._conn.execute(
                "UPDATE app_users SET email = ?, updated_at = ? WHERE auth_user_id = ?",
                (email, now, auth_user_id),
            )
            self._conn.commit()
        return {"id": row[0], "auth_user_id": row[1], "email": email or row[2], "plan": row[3], "new": False}

    def set_user_plan(self, auth_user_id: str, plan: str) -> dict[str, Any] | None:
        # NOTE: must not call get_or_create_user here — the repository
        # lock is non-reentrant, so same-thread reentry would deadlock.
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "UPDATE app_users SET plan = ?, updated_at = ? WHERE auth_user_id = ?",
            (plan, now, auth_user_id),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None
        row = self._conn.execute(
            "SELECT id, auth_user_id, email, plan FROM app_users WHERE auth_user_id = ?",
            (auth_user_id,),
        ).fetchone()
        assert row is not None
        return {"id": row[0], "auth_user_id": row[1], "email": row[2], "plan": row[3], "new": False}

    def claim_visitor_events(self, visitor_id: str, auth_user_id: str) -> int:
        """Attribute unclaimed visitor events to a user (anonymous → signup).

        Only rows not already claimed are moved, so the daily quota is not
        reset by signing up.  Returns the number of rows claimed.
        """
        cursor = self._conn.execute(
            "UPDATE usage_events SET user_id = ? "
            "WHERE visitor_id = ? AND user_id IS NULL",
            (auth_user_id, visitor_id),
        )
        self._conn.commit()
        return cursor.rowcount

    def _public_use_count(self, *, visitor_id: str | None, auth_user_id: str | None, today: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM usage_events "
            "WHERE event_type = 'public_analyze' AND success = 1 AND created_at >= ? "
            "AND (user_id = ? OR (visitor_id = ? AND user_id IS NULL))",
            (today, auth_user_id, visitor_id),
        ).fetchone()
        return int(row[0])

    def count_public_uses_today(
        self, *, visitor_id: str | None, auth_user_id: str | None
    ) -> int:
        """Successful public analyses consumed today (UTC day)."""
        return self._public_use_count(
            visitor_id=visitor_id, auth_user_id=auth_user_id, today=self._utc_today()
        )

    def try_consume_public_use(
        self, *, visitor_id: str | None, auth_user_id: str | None, chars: int, limit: int
    ) -> tuple[bool, int]:
        """Atomically consume one quota unit if under *limit*.

        Returns (consumed, remaining).  Only successful analyses should
        call this — failures must not consume quota.
        """
        today = self._utc_today()
        used = self._public_use_count(visitor_id=visitor_id, auth_user_id=auth_user_id, today=today)
        if used >= limit:
            return False, 0
        self._conn.execute(
            "INSERT INTO usage_events (id, created_at, event_type, visitor_id, user_id, chars, success) "
            "VALUES (?, ?, 'public_analyze', ?, ?, ?, 1)",
            (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(), visitor_id, auth_user_id, chars),
        )
        self._conn.commit()
        return True, limit - used - 1

    # -- Billing ---------------------------------------------------------
    def get_subscription_for_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return _subscription_to_dict(row) if row is not None else None

    def get_subscription_by_provider_customer(
        self, provider: str, provider_customer_id: str
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM subscriptions WHERE provider = ? AND provider_customer_id = ? "
            "ORDER BY updated_at DESC LIMIT 1", (provider, provider_customer_id)
        ).fetchone()
        return _subscription_to_dict(row) if row is not None else None

    def upsert_subscription(
        self, *, user_id: str, provider: str, provider_customer_id: str,
        provider_subscription_id: str | None, provider_price_id: str | None,
        status: str, current_period_start: str | None,
        current_period_end: str | None, cancel_at_period_end: bool,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        if provider_subscription_id:
            existing = self._conn.execute(
                "SELECT id FROM subscriptions WHERE provider = ? AND provider_subscription_id = ?",
                (provider, provider_subscription_id),
            ).fetchone()
        else:
            existing = self._conn.execute(
                "SELECT id FROM subscriptions WHERE provider = ? AND user_id = ? ORDER BY updated_at DESC LIMIT 1",
                (provider, user_id),
            ).fetchone()
        subscription_id = existing[0] if existing else str(uuid.uuid4())
        if provider_subscription_id:
            self._conn.execute(
                "INSERT INTO subscriptions (id, user_id, provider, provider_customer_id, provider_subscription_id, provider_price_id, status, current_period_start, current_period_end, cancel_at_period_end, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(provider, provider_subscription_id) DO UPDATE SET "
                "user_id=excluded.user_id, provider_customer_id=excluded.provider_customer_id, provider_price_id=excluded.provider_price_id, status=excluded.status, current_period_start=excluded.current_period_start, current_period_end=excluded.current_period_end, cancel_at_period_end=excluded.cancel_at_period_end, updated_at=excluded.updated_at",
                (subscription_id, user_id, provider, provider_customer_id, provider_subscription_id, provider_price_id, status, current_period_start, current_period_end, 1 if cancel_at_period_end else 0, now, now),
            )
        else:
            self._conn.execute(
                "INSERT INTO subscriptions (id, user_id, provider, provider_customer_id, provider_subscription_id, provider_price_id, status, current_period_start, current_period_end, cancel_at_period_end, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "provider_customer_id=excluded.provider_customer_id, provider_price_id=excluded.provider_price_id, status=excluded.status, current_period_start=excluded.current_period_start, current_period_end=excluded.current_period_end, cancel_at_period_end=excluded.cancel_at_period_end, updated_at=excluded.updated_at",
                (subscription_id, user_id, provider, provider_customer_id, provider_price_id, status, current_period_start, current_period_end, 1 if cancel_at_period_end else 0, now, now),
            )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        assert row is not None
        return _subscription_to_dict(row)

    def record_webhook_event(self, *, provider: str, event_id: str, event_type: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO billing_webhook_events (id, provider, event_type, received_at, processed_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'processed') ON CONFLICT(provider, id) DO NOTHING",
            (event_id, provider, event_type, now, now),
        )
        self._conn.commit()
        return cursor.rowcount == 1


# ---------------------------------------------------------------------------
# PostgreSQL backend
# ---------------------------------------------------------------------------

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id    TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    text_hash      TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    token_count    INTEGER NOT NULL,
    status         TEXT NOT NULL,
    detectors      TEXT NOT NULL,
    result_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_text_hash ON analyses(text_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id      TEXT PRIMARY KEY,
    key_hash    TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL,
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS usage_records (
    id             SERIAL PRIMARY KEY,
    key_id         TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    status_code    INTEGER NOT NULL,
    duration_ms    DOUBLE PRECISION NOT NULL,
    character_count INTEGER NOT NULL,
    success        INTEGER NOT NULL,
    detector       TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_key_id ON usage_records(key_id);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp);

CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    key_id         TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    completed_at   TEXT,
    input_hash     TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    detectors      TEXT NOT NULL,
    config_path    TEXT,
    result_json    TEXT,
    error_message  TEXT,
    duration_ms    DOUBLE PRECISION,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    parent_job_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_key_id ON jobs(key_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

# PostgreSQL migrations must not rely on catching statement errors inside a
# transaction: after any SQL error PostgreSQL rejects every following statement
# until rollback. ``IF NOT EXISTS`` keeps this migration safe on both fresh and
# existing databases without aborting the surrounding schema transaction.
_MIGRATE_PG = """
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS parent_job_id TEXT;
"""


@_synchronize_repository
class PostgresRepository:
    """PostgreSQL-backed repository for production deployments."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "psycopg2 is required for PostgreSQL support.  "
                "Install it with: pip install psycopg2-binary"
            ) from exc
        self._lock = _make_instance_lock()
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        """Create/migrate the PostgreSQL schema in one atomic transaction."""
        stage = "core_tables"
        try:
            with self._conn.cursor() as cur:
                cur.execute(_SCHEMA_PG)
                stage = "benchmark_tables"
                cur.execute(_SCHEMA_BENCHMARK_RUNS)
                stage = "identity_tables"
                cur.execute(_SCHEMA_IDENTITY)
                stage = "billing_tables"
                cur.execute(_SCHEMA_BILLING)
                stage = "jobs_columns"
                cur.execute(_MIGRATE_PG)
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            # Never log the DSN or exception string: either can contain
            # deployment details. Stage, exception type, and SQLSTATE are
            # sufficient to diagnose bootstrap permissions/DDL failures.
            logger.error(
                "PostgreSQL schema bootstrap failed stage=%s error_type=%s sqlstate=%s",
                stage,
                type(exc).__name__,
                getattr(exc, "pgcode", None) or "unavailable",
            )
            raise

    def close(self) -> None:
        self._conn.close()

    def save(
        self,
        *,
        text: str,
        engine_version: str,
        detectors: list[str],
        result_dict: dict[str, Any],
    ) -> str:
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        text_stats = result_dict.get("text_stats", {})
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO analyses "
                "(analysis_id, timestamp, engine_version, text_hash, "
                "character_count, token_count, status, detectors, result_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    analysis_id,
                    timestamp,
                    engine_version,
                    _text_hash(text),
                    text_stats.get("character_count", 0),
                    text_stats.get("token_count", 0),
                    result_dict.get("status", "unknown"),
                    json.dumps(detectors),
                    json.dumps(result_dict),
                ),
            )
        self._conn.commit()
        return analysis_id

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM analyses WHERE analysis_id = %s", (analysis_id,))
            row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]  # type: ignore[union-attr]
        d = dict(zip(cols, row))
        d["detectors"] = json.loads(d["detectors"])
        d["result"] = json.loads(d["result_json"])
        return d

    def list_analyses(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM analyses")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT analysis_id, timestamp, engine_version, text_hash, "
                "character_count, token_count, status, detectors "
                "FROM analyses ORDER BY timestamp DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            rows = cur.fetchall()
        summaries = []
        for row in rows:
            summaries.append({
                "analysis_id": row[0],
                "timestamp": row[1],
                "engine_version": row[2],
                "text_hash": row[3],
                "character_count": row[4],
                "token_count": row[5],
                "status": row[6],
                "detector_count": len(json.loads(row[7])),
            })
        return summaries, total

    # -- api keys --

    def create_api_key(
        self, *, key_id: str, key_hash: str, name: str
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_id, key_hash, name, status, created_at) "
                "VALUES (%s, %s, %s, 'active', %s)",
                (key_id, key_hash, name, datetime.now(timezone.utc).isoformat()),
            )
        self._conn.commit()

    def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM api_keys WHERE key_hash = %s", (key_hash,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        return d

    def get_api_key_by_id(self, key_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM api_keys WHERE key_id = %s", (key_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def list_api_keys(self) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT key_id, name, status, created_at FROM api_keys ORDER BY created_at DESC")
            rows = cur.fetchall()
        return [{"key_id": r[0], "name": r[1], "status": r[2], "created_at": r[3]} for r in rows]

    def revoke_api_key(self, key_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("UPDATE api_keys SET status = 'revoked' WHERE key_id = %s", (key_id,))
        self._conn.commit()

    def update_api_key_last_used(self, key_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET last_used_at = %s WHERE key_id = %s",
                (datetime.now(timezone.utc).isoformat(), key_id),
            )
        self._conn.commit()

    # -- usage --

    def record_usage(
        self,
        *,
        key_id: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        character_count: int,
        success: bool,
        detector: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO usage_records "
                "(key_id, endpoint, timestamp, status_code, duration_ms, "
                "character_count, success, detector) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    key_id,
                    endpoint,
                    datetime.now(timezone.utc).isoformat(),
                    status_code,
                    duration_ms,
                    character_count,
                    1 if success else 0,
                    detector,
                ),
            )
        self._conn.commit()

    def get_usage_summary(
        self, *, key_id: str | None = None, since: str | None = None
    ) -> dict[str, Any]:
        conditions = []
        params: list[Any] = []
        if key_id:
            conditions.append("key_id = %s")
            params.append(key_id)
        if since:
            conditions.append("timestamp >= %s")
            params.append(since)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM usage_records{where}", params)
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT COUNT(*) FROM usage_records{where}{(' AND ' if where else ' WHERE ')}success = 1",
                params,
            )
            successful = cur.fetchone()[0]
            failed = total - successful
            cur.execute(
                f"SELECT COALESCE(SUM(character_count), 0) FROM usage_records{where}",
                params,
            )
            total_chars = cur.fetchone()[0]
            cur.execute(
                f"SELECT endpoint, COUNT(*) FROM usage_records{where} GROUP BY endpoint",
                params,
            )
            by_endpoint = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute(
                f"SELECT detector, COUNT(*) FROM usage_records{where}{(' AND ' if where else ' WHERE ')}detector IS NOT NULL GROUP BY detector",
                params,
            )
            by_detector = {r[0]: r[1] for r in cur.fetchall()}

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "total_characters": total_chars,
            "by_endpoint": by_endpoint,
            "by_detector": by_detector,
        }

    def get_usage_today(self, key_id: str) -> tuple[int, int]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), COALESCE(SUM(character_count), 0) "
                "FROM usage_records WHERE key_id = %s AND timestamp >= %s",
                (key_id, today),
            )
            row = cur.fetchone()
        return (row[0], row[1])

    def get_usage_today_jobs(self, key_id: str) -> tuple[int, int]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), COALESCE(SUM(character_count), 0) "
                "FROM jobs WHERE key_id = %s AND status IN ('queued', 'running') "
                "AND created_at >= %s",
                (key_id, today),
            )
            row = cur.fetchone()
        return (row[0], row[1])

    def get_avg_duration(self) -> float | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT AVG(duration_ms) FROM usage_records")
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return round(row[0], 2)

    def get_usage_by_status(self) -> dict[str, int]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT status_code, COUNT(*) FROM usage_records GROUP BY status_code"
            )
            rows = cur.fetchall()
        return {str(r[0]): r[1] for r in rows}

    def commit_if_needed(self) -> None:
        self._conn.commit()

    # -- jobs --

    def create_job(
        self,
        *,
        job_id: str,
        key_id: str,
        detectors: list[str],
        config_path: str | None,
        input_hash: str,
        character_count: int,
        retry_count: int = 0,
        parent_job_id: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs "
                "(job_id, key_id, status, created_at, input_hash, "
                "character_count, detectors, config_path, retry_count, parent_job_id) "
                "VALUES (%s, %s, 'queued', %s, %s, %s, %s, %s, %s, %s)",
                (
                    job_id,
                    key_id,
                    datetime.now(timezone.utc).isoformat(),
                    input_hash,
                    character_count,
                    json.dumps(detectors),
                    config_path,
                    retry_count,
                    parent_job_id,
                ),
            )
        self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        d["detectors"] = json.loads(d["detectors"])
        if d.get("result_json"):
            d["result"] = json.loads(d["result_json"])
        return d

    def update_job_status(
        self,
        *,
        job_id: str,
        status: str,
        started_at: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            if started_at:
                cur.execute(
                    "UPDATE jobs SET status = %s, started_at = %s WHERE job_id = %s",
                    (status, started_at, job_id),
                )
            else:
                cur.execute(
                    "UPDATE jobs SET status = %s WHERE job_id = %s",
                    (status, job_id),
                )
        self._conn.commit()

    def complete_job(
        self,
        *,
        job_id: str,
        result_json: str,
        duration_ms: float,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'completed', completed_at = %s, "
                "result_json = %s, duration_ms = %s WHERE job_id = %s",
                (
                    datetime.now(timezone.utc).isoformat(),
                    result_json,
                    duration_ms,
                    job_id,
                ),
            )
        self._conn.commit()

    def fail_job(
        self,
        *,
        job_id: str,
        error_message: str,
        duration_ms: float,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'failed', completed_at = %s, "
                "error_message = %s, duration_ms = %s WHERE job_id = %s",
                (
                    datetime.now(timezone.utc).isoformat(),
                    error_message,
                    duration_ms,
                    job_id,
                ),
            )
        self._conn.commit()

    def list_jobs_by_key(
        self,
        *,
        key_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE key_id = %s", (key_id,)
            )
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT job_id, status, created_at, started_at, "
                "completed_at, character_count, detectors, "
                "error_message, duration_ms, retry_count, parent_job_id "
                "FROM jobs WHERE key_id = %s ORDER BY created_at DESC "
                "LIMIT %s OFFSET %s",
                (key_id, limit, offset),
            )
            rows = cur.fetchall()
        summaries = []
        for row in rows:
            summaries.append({
                "job_id": row[0],
                "status": row[1],
                "created_at": row[2],
                "started_at": row[3],
                "completed_at": row[4],
                "character_count": row[5],
                "detectors": json.loads(row[6]),
                "error_message": row[7],
                "duration_ms": row[8],
                "retry_count": row[9],
                "parent_job_id": row[10],
            })
        return summaries, total

    def cancel_job_if_status(self, job_id: str, expected_status: str, new_status: str) -> bool:
        """Atomically transition a job's status only if it matches expected_status."""
        if new_status == "cancelled":
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status = %s, completed_at = %s, "
                    "error_message = 'Job cancelled by user' "
                    "WHERE job_id = %s AND status = %s",
                    (new_status, datetime.now(timezone.utc).isoformat(), job_id, expected_status),
                )
                result = cur.rowcount > 0
        else:
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status = %s WHERE job_id = %s AND status = %s",
                    (new_status, job_id, expected_status),
                )
                result = cur.rowcount > 0
        self._conn.commit()
        return result

    def get_job_counts(self) -> dict[str, int]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            rows = cur.fetchall()
        counts = {r[0]: r[1] for r in rows}
        return {
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "cancelled": counts.get("cancelled", 0),
            "cancellation_requested": counts.get("cancellation_requested", 0),
        }

    def cleanup_jobs(self, older_than: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'cancelled') "
                "AND completed_at < %s",
                (older_than,),
            )
            count = cur.rowcount
        self._conn.commit()
        return count

    def recover_stale_jobs(self) -> int:
        """Mark jobs stuck in 'running' or 'queued' as failed (crash recovery)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'failed', completed_at = %s, "
                "error_message = 'Job interrupted by application restart', "
                "duration_ms = 0 "
                "WHERE status = 'running'",
                (now,),
            )
            running_count = cur.rowcount
            cur.execute(
                "UPDATE jobs SET status = 'failed', completed_at = %s, "
                "error_message = 'Job interrupted by application restart', "
                "duration_ms = 0 "
                "WHERE status = 'queued'",
                (now,),
            )
            queued_count = cur.rowcount
        self._conn.commit()
        return running_count + queued_count

    # ------------------------------------------------------------------
    # Benchmark runs (Phase 6C)
    # ------------------------------------------------------------------

    def create_benchmark_run(
        self,
        *,
        run_id: str,
        key_id: str,
        config_json: str,
        out_dir: str,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO benchmark_runs "
                "(run_id, key_id, status, created_at, config_json, out_dir) "
                "VALUES (%s, %s, 'queued', %s, %s, %s)",
                (
                    run_id,
                    key_id,
                    datetime.now(timezone.utc).isoformat(),
                    config_json,
                    out_dir,
                ),
            )
        self._conn.commit()

    def get_benchmark_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM benchmark_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
        return _benchmark_run_to_dict(dict(zip(cols, row)))

    def list_benchmark_runs(
        self,
        *,
        key_id: str,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._conn.cursor() as cur:
            if status is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM benchmark_runs WHERE key_id = %s AND status = %s",
                    (key_id, status),
                )
                total = cur.fetchone()[0]
                cur.execute(
                    "SELECT run_id, key_id, status, created_at, started_at, "
                    "completed_at, config_json, out_dir, progress_json, "
                    "error_message, duration_ms, retry_count "
                    "FROM benchmark_runs WHERE key_id = %s AND status = %s "
                    "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (key_id, status, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM benchmark_runs WHERE key_id = %s",
                    (key_id,),
                )
                total = cur.fetchone()[0]
                cur.execute(
                    "SELECT run_id, key_id, status, created_at, started_at, "
                    "completed_at, config_json, out_dir, progress_json, "
                    "error_message, duration_ms, retry_count "
                    "FROM benchmark_runs WHERE key_id = %s "
                    "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (key_id, limit, offset),
                )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        summaries = []
        for row in rows:
            d = dict(zip(cols, row))
            summaries.append({
                "run_id": d["run_id"],
                "status": d["status"],
                "created_at": d["created_at"],
                "started_at": d["started_at"],
                "completed_at": d["completed_at"],
                "config": json.loads(d["config_json"]),
                "out_dir": d["out_dir"],
                "progress": json.loads(d["progress_json"]) if d["progress_json"] else None,
                "error_message": d["error_message"],
                "duration_ms": d["duration_ms"],
                "retry_count": d["retry_count"],
            })
        return summaries, total

    def update_benchmark_run(
        self,
        *,
        run_id: str,
        status: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        progress_json: str | None = None,
        result_json: str | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
        retry_count: int | None = None,
        clear_attempt_state: bool = False,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if clear_attempt_state:
            # Retry: drop the previous attempt's timestamps/error so the
            # queued run does not display stale terminal state.
            updates.extend([
                "started_at = NULL",
                "completed_at = NULL",
                "error_message = NULL",
            ])
        if status is not None:
            updates.append("status = %s")
            params.append(status)
        if started_at is not None:
            updates.append("started_at = %s")
            params.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = %s")
            params.append(completed_at)
        if progress_json is not None:
            updates.append("progress_json = %s")
            params.append(progress_json)
        if result_json is not None:
            updates.append("result_json = %s")
            params.append(result_json)
        if error_message is not None:
            updates.append("error_message = %s")
            params.append(error_message)
        if duration_ms is not None:
            updates.append("duration_ms = %s")
            params.append(duration_ms)
        if retry_count is not None:
            updates.append("retry_count = %s")
            params.append(retry_count)
        if not updates:
            return
        params.append(run_id)
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE benchmark_runs SET {', '.join(updates)} WHERE run_id = %s",
                params,
            )
        self._conn.commit()

    def complete_benchmark_run(
        self,
        *,
        run_id: str,
        result_json: str,
        duration_ms: float,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_runs SET status = 'completed', completed_at = %s, "
                "result_json = %s, duration_ms = %s WHERE run_id = %s",
                (
                    datetime.now(timezone.utc).isoformat(),
                    result_json,
                    duration_ms,
                    run_id,
                ),
            )
        self._conn.commit()

    def fail_benchmark_run(
        self,
        *,
        run_id: str,
        error_message: str,
        duration_ms: float,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_runs SET status = 'failed', completed_at = %s, "
                "error_message = %s, duration_ms = %s WHERE run_id = %s",
                (
                    datetime.now(timezone.utc).isoformat(),
                    error_message,
                    duration_ms,
                    run_id,
                ),
            )
        self._conn.commit()

    def cancel_benchmark_run_if_status(
        self, run_id: str, expected_status: str, new_status: str
    ) -> bool:
        """Atomically transition a run's status only if it matches expected_status."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            if new_status == "cancelled":
                cur.execute(
                    "UPDATE benchmark_runs SET status = %s, completed_at = %s, "
                    "error_message = 'Benchmark run cancelled by user' "
                    "WHERE run_id = %s AND status = %s",
                    (new_status, now, run_id, expected_status),
                )
            else:
                cur.execute(
                    "UPDATE benchmark_runs SET status = %s WHERE run_id = %s AND status = %s",
                    (new_status, run_id, expected_status),
                )
            result = cur.rowcount > 0
        self._conn.commit()
        return result

    def recover_stale_benchmark_runs(self) -> int:
        """Crash recovery: interrupted runs become failed, never successful."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_runs SET status = 'failed', completed_at = %s, "
                "error_message = 'Benchmark run interrupted by application restart', "
                "duration_ms = 0 "
                "WHERE status IN ('running', 'queued')",
                (now,),
            )
            interrupted = cur.rowcount
            cur.execute(
                "UPDATE benchmark_runs SET status = 'cancelled', completed_at = %s, "
                "error_message = 'Benchmark run cancelled by user' "
                "WHERE status = 'cancellation_requested'",
                (now,),
            )
            cancelled = cur.rowcount
        self._conn.commit()
        return interrupted + cancelled

    # -- SaaS identity / quota -------------------------------------------
    # Cross-process race safety comes from a transaction-scoped advisory
    # lock per identity scope (the instance lock only covers threads).

    @staticmethod
    def _utc_today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _quota_scope(visitor_id: str | None, auth_user_id: str | None) -> str:
        return f"user:{auth_user_id}" if auth_user_id else f"visitor:{visitor_id}"

    def get_or_create_visitor(self, visitor_id: str, abuse_hash: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT visitor_id, created_at FROM anonymous_visitors WHERE visitor_id = %s",
                (visitor_id,),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO anonymous_visitors (visitor_id, created_at, last_seen_at, abuse_hash) "
                    "VALUES (%s, %s, %s, %s)",
                    (visitor_id, now, now, abuse_hash),
                )
                self._conn.commit()
                return {"visitor_id": visitor_id, "created_at": now, "new": True}
            cur.execute(
                "UPDATE anonymous_visitors SET last_seen_at = %s, "
                "abuse_hash = COALESCE(%s, abuse_hash) WHERE visitor_id = %s",
                (now, abuse_hash, visitor_id),
            )
        self._conn.commit()
        return {"visitor_id": row[0], "created_at": row[1], "new": False}

    def get_or_create_user(self, auth_user_id: str, email: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, auth_user_id, email, plan FROM app_users WHERE auth_user_id = %s",
                (auth_user_id,),
            )
            row = cur.fetchone()
            if row is None:
                uid = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO app_users (id, auth_user_id, email, plan, created_at, updated_at) "
                    "VALUES (%s, %s, %s, 'free', %s, %s)",
                    (uid, auth_user_id, email, now, now),
                )
                self._conn.commit()
                return {"id": uid, "auth_user_id": auth_user_id, "email": email, "plan": "free", "new": True}
            if email and email != row[2]:
                cur.execute(
                    "UPDATE app_users SET email = %s, updated_at = %s WHERE auth_user_id = %s",
                    (email, now, auth_user_id),
                )
                self._conn.commit()
                return {"id": row[0], "auth_user_id": row[1], "email": email, "plan": row[3], "new": False}
        return {"id": row[0], "auth_user_id": row[1], "email": row[2], "plan": row[3], "new": False}

    def set_user_plan(self, auth_user_id: str, plan: str) -> dict[str, Any] | None:
        # NOTE: must not call get_or_create_user here — the repository
        # lock is non-reentrant, so same-thread reentry would deadlock.
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE app_users SET plan = %s, updated_at = %s WHERE auth_user_id = %s",
                (plan, now, auth_user_id),
            )
            if cur.rowcount == 0:
                self._conn.rollback()
                return None
            cur.execute(
                "SELECT id, auth_user_id, email, plan FROM app_users WHERE auth_user_id = %s",
                (auth_user_id,),
            )
            row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        return {"id": row[0], "auth_user_id": row[1], "email": row[2], "plan": row[3], "new": False}

    def claim_visitor_events(self, visitor_id: str, auth_user_id: str) -> int:
        """Attribute unclaimed visitor events to a user (anonymous → signup)."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE usage_events SET user_id = %s "
                "WHERE visitor_id = %s AND user_id IS NULL",
                (auth_user_id, visitor_id),
            )
            claimed = cur.rowcount
        self._conn.commit()
        return claimed

    def _public_use_count(
        self, cur: Any, *, visitor_id: str | None, auth_user_id: str | None, today: str
    ) -> int:
        cur.execute(
            "SELECT COUNT(*) FROM usage_events "
            "WHERE event_type = 'public_analyze' AND success = 1 AND created_at >= %s "
            "AND (user_id = %s OR (visitor_id = %s AND user_id IS NULL))",
            (today, auth_user_id, visitor_id),
        )
        row = cur.fetchone()
        return int(row[0])

    def count_public_uses_today(
        self, *, visitor_id: str | None, auth_user_id: str | None
    ) -> int:
        """Successful public analyses consumed today (UTC day)."""
        with self._conn.cursor() as cur:
            return self._public_use_count(
                cur, visitor_id=visitor_id, auth_user_id=auth_user_id, today=self._utc_today()
            )

    def try_consume_public_use(
        self, *, visitor_id: str | None, auth_user_id: str | None, chars: int, limit: int
    ) -> tuple[bool, int]:
        """Atomically consume one quota unit if under *limit*.

        Serialized per identity scope via a transaction-scoped advisory
        lock, so concurrent requests cannot jointly exceed the quota.
        Returns (consumed, remaining).
        """
        today = self._utc_today()
        scope = self._quota_scope(visitor_id, auth_user_id)
        with self._conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (scope,))
            used = self._public_use_count(
                cur, visitor_id=visitor_id, auth_user_id=auth_user_id, today=today
            )
            if used >= limit:
                self._conn.rollback()
                return False, 0
            cur.execute(
                "INSERT INTO usage_events (id, created_at, event_type, visitor_id, user_id, chars, success) "
                "VALUES (%s, %s, 'public_analyze', %s, %s, %s, 1)",
                (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(), visitor_id, auth_user_id, chars),
            )
        self._conn.commit()
        return True, limit - used - 1

    # -- Billing ---------------------------------------------------------
    def get_subscription_for_user(self, user_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, provider, provider_customer_id, provider_subscription_id, "
                "provider_price_id, status, current_period_start, current_period_end, "
                "cancel_at_period_end, created_at, updated_at FROM subscriptions "
                "WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1", (user_id,)
            )
            row = cur.fetchone()
        return _subscription_to_dict(row) if row is not None else None

    def get_subscription_by_provider_customer(
        self, provider: str, provider_customer_id: str
    ) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, provider, provider_customer_id, provider_subscription_id, "
                "provider_price_id, status, current_period_start, current_period_end, "
                "cancel_at_period_end, created_at, updated_at FROM subscriptions "
                "WHERE provider = %s AND provider_customer_id = %s ORDER BY updated_at DESC LIMIT 1",
                (provider, provider_customer_id),
            )
            row = cur.fetchone()
        return _subscription_to_dict(row) if row is not None else None

    def upsert_subscription(self, *, user_id: str, provider: str, provider_customer_id: str,
                            provider_subscription_id: str | None, provider_price_id: str | None,
                            status: str, current_period_start: str | None,
                            current_period_end: str | None, cancel_at_period_end: bool) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            if provider_subscription_id:
                cur.execute("SELECT id FROM subscriptions WHERE provider = %s AND provider_subscription_id = %s", (provider, provider_subscription_id))
            else:
                cur.execute("SELECT id FROM subscriptions WHERE provider = %s AND user_id = %s ORDER BY updated_at DESC LIMIT 1", (provider, user_id))
            existing = cur.fetchone()
            subscription_id = existing[0] if existing else str(uuid.uuid4())
            if provider_subscription_id:
                cur.execute(
                    "INSERT INTO subscriptions (id, user_id, provider, provider_customer_id, provider_subscription_id, provider_price_id, status, current_period_start, current_period_end, cancel_at_period_end, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT(provider, provider_subscription_id) DO UPDATE SET user_id=EXCLUDED.user_id, provider_customer_id=EXCLUDED.provider_customer_id, provider_price_id=EXCLUDED.provider_price_id, status=EXCLUDED.status, current_period_start=EXCLUDED.current_period_start, current_period_end=EXCLUDED.current_period_end, cancel_at_period_end=EXCLUDED.cancel_at_period_end, updated_at=EXCLUDED.updated_at",
                    (subscription_id, user_id, provider, provider_customer_id, provider_subscription_id, provider_price_id, status, current_period_start, current_period_end, 1 if cancel_at_period_end else 0, now, now),
                )
            else:
                cur.execute(
                    "INSERT INTO subscriptions (id, user_id, provider, provider_customer_id, provider_subscription_id, provider_price_id, status, current_period_start, current_period_end, cancel_at_period_end, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET provider_customer_id=EXCLUDED.provider_customer_id, provider_price_id=EXCLUDED.provider_price_id, status=EXCLUDED.status, current_period_start=EXCLUDED.current_period_start, current_period_end=EXCLUDED.current_period_end, cancel_at_period_end=EXCLUDED.cancel_at_period_end, updated_at=EXCLUDED.updated_at",
                    (subscription_id, user_id, provider, provider_customer_id, provider_price_id, status, current_period_start, current_period_end, 1 if cancel_at_period_end else 0, now, now),
                )
        self._conn.commit()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, provider, provider_customer_id, provider_subscription_id, "
                "provider_price_id, status, current_period_start, current_period_end, "
                "cancel_at_period_end, created_at, updated_at FROM subscriptions "
                "WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1", (user_id,)
            )
            row = cur.fetchone()
        assert row is not None
        return _subscription_to_dict(row)

    def record_webhook_event(self, *, provider: str, event_id: str, event_type: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO billing_webhook_events (id, provider, event_type, received_at, processed_at, status) VALUES (%s, %s, %s, %s, %s, 'processed') ON CONFLICT(provider, id) DO NOTHING",
                (event_id, provider, event_type, now, now),
            )
            inserted = cur.rowcount == 1
        self._conn.commit()
        return inserted


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "data/provenance.db"


def create_repository(url: str | None = None) -> AnalysisRepository:
    """Create the appropriate repository based on *url* or ``DATABASE_URL``.

    Resolution order:

    1. Explicit *url* argument.
    2. ``DATABASE_URL`` environment variable.
    3. Default SQLite at ``data/provenance.db``.
    """
    raw = url or os.environ.get("DATABASE_URL") or ""
    if not raw:
        return SqliteRepository(_DEFAULT_DB_PATH)

    normalised = raw.strip()
    lower = normalised.lower()

    if lower.startswith("postgresql://") or lower.startswith("postgres://"):
        return PostgresRepository(normalised)

    if lower.startswith("sqlite://"):
        # sqlite:///relative/path  ->  relative/path
        # sqlite:////absolute/path ->  /absolute/path
        # sqlite:///<abs tmp path>  (e.g. sqlite:////tmp/x/test.db)
        path = normalised[len("sqlite://"):]
        if path.startswith("//"):
            path = "/" + path.lstrip("/")  # absolute (exactly one leading slash)
        elif path.startswith("/"):
            path = path[1:]  # relative (strip the single separator slash)
        return SqliteRepository(path)

    # Fallback: treat as a SQLite file path for convenience.
    return SqliteRepository(normalised)
