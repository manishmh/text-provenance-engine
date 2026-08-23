"""Persistence layer for provenance analysis results.

Provides a :class:`AnalysisRepository` protocol and two concrete backends:

- :class:`SqliteRepository` — default for local development.
- :class:`PostgresRepository` — activated via ``DATABASE_URL``.

Both use a single ``analyses`` table with a JSON result payload.  Route
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
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_hash(text: str) -> str:
    """SHA-256 of the input text (used for deduplication, text is NOT stored)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class AnalysisRepository(Protocol):
    """Storage interface for analysis results."""

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
"""


class SqliteRepository:
    """SQLite-backed repository for local development."""

    def __init__(self, db_path: str | Path = "data/provenance.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQLITE)

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
"""


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
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_PG)
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
        # Build dict from column names
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
        # sqlite:///relative/path  or  sqlite:////absolute/path
        path = normalised[len("sqlite://"):]
        if path.startswith("///"):
            path = path[2:]  # absolute
        return SqliteRepository(path)

    # Fallback: treat as a SQLite file path for convenience.
    return SqliteRepository(normalised)
