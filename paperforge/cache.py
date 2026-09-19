"""SQLite-backed HTTP response cache.

Two jobs: stop hammering public APIs on re-runs, and make a run reproducible —
replaying a cached run gives byte-identical payloads, so the evidence log can be
audited months later even if OpenAlex has moved on.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
DEFAULT_PATH = Path(".paperforge_cache.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key        TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    status     INTEGER NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source ON responses(source);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    manifest   TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class ResponseCache:
    def __init__(self, path: Path | str = DEFAULT_PATH, ttl_days: int = 30):
        self.path = Path(path)
        self.ttl = ttl_days * 86400
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        with _LOCK:
            row = self._conn.execute(
                "SELECT payload, fetched_at FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        payload, fetched_at = row
        if time.time() - fetched_at > self.ttl:
            return None
        return json.loads(payload)

    def put(self, key: str, source: str, payload: dict, status: int = 200) -> None:
        with _LOCK:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                (key, source, json.dumps(payload, ensure_ascii=False), status, time.time()),
            )
            self._conn.commit()

    def save_manifest(self, run_id: str, manifest: dict) -> None:
        with _LOCK:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
                (run_id, json.dumps(manifest, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def list_runs(self, limit: int = 20) -> list[dict]:
        with _LOCK:
            rows = self._conn.execute(
                "SELECT run_id, manifest, created_at FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"run_id": r[0], "manifest": json.loads(r[1]), "created_at": r[2]} for r in rows
        ]

    def stats(self) -> dict:
        with _LOCK:
            total = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            by_source = self._conn.execute(
                "SELECT source, COUNT(*) FROM responses GROUP BY source"
            ).fetchall()
        return {"entries": total, "by_source": dict(by_source)}

    def clear(self) -> None:
        with _LOCK:
            self._conn.execute("DELETE FROM responses")
            self._conn.commit()
