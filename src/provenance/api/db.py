"""SQLite persistence layer for provenance analysis results.

Uses a single ``analyses`` table with a JSON result payload.  The abstraction
allows swapping SQLite for PostgreSQL later by replacing this module only.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
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


def _text_hash(text: str) -> str:
    """SHA-256 of the input text (used for deduplication, text is NOT stored)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AnalysisRepository:
    """Repository for persisting and retrieving analysis results."""

    def __init__(self, db_path: str | Path = "data/provenance.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

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
        """Persist an analysis result. Returns the analysis_id."""
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
        """Retrieve a full analysis result by ID."""
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
        """List analyses with pagination. Returns (summaries, total_count)."""
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
