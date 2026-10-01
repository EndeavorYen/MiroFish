"""Runs and their events in SQLite, so a backend restart keeps them (#63).

``runs`` holds one row per run: status, current stage, the request
parameters, the artifacts each finished stage produced (project, graph,
simulation and report ids) and the failure reason. ``events`` is the
append-only log the SSE endpoint streams by id.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

# Run statuses. ``interrupted``: the backend stopped while the run was going.
# ``awaiting_confirmation``: paused until the roles are confirmed.
QUEUED, RUNNING, COMPLETED, FAILED, INTERRUPTED = "queued", "running", "completed", "failed", "interrupted"
AWAITING = "awaiting_confirmation"
TERMINAL = (COMPLETED, FAILED, INTERRUPTED, AWAITING)  # no thread is working on the run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT,
    params TEXT NOT NULL,
    artifacts TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    stage TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_db_path() -> str:
    from ..config import Config

    return os.environ.get("RUNS_DB_PATH") or os.path.join(Config.UPLOAD_FOLDER, "runs", "runs.sqlite")


class RunStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = os.path.abspath(path or default_db_path())
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def create(self, params: dict[str, Any]) -> str:
        run_id = "run_" + uuid.uuid4().hex[:12]
        now = _now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs (id, status, stage, params, artifacts, created_at, updated_at) "
                "VALUES (?, ?, NULL, ?, '{}', ?, ?)",
                (run_id, QUEUED, json.dumps(params, ensure_ascii=False), now, now),
            )
        return run_id

    def set_params(self, run_id: str, params: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE runs SET params = ?, updated_at = ? WHERE id = ?",
                         (json.dumps(params, ensure_ascii=False), _now(), run_id))

    @staticmethod
    def _row(row) -> dict[str, Any]:
        return {
            "run_id": row[0], "status": row[1], "stage": row[2], "params": json.loads(row[3]),
            "artifacts": json.loads(row[4]), "error": row[5], "created_at": row[6], "updated_at": row[7],
        }

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, status, stage, params, artifacts, error, created_at, updated_at FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, status, stage, params, artifacts, error, created_at, updated_at FROM runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def update(self, run_id: str, *, status: str | None = None, stage: str | None = None,
               error: str | None = None, artifacts: dict[str, Any] | None = None,
               clear_error: bool = False) -> None:
        with self._tx() as conn:
            row = conn.execute("SELECT artifacts FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            merged = json.loads(row[0])
            if artifacts:
                merged.update(artifacts)
            conn.execute(
                "UPDATE runs SET status = COALESCE(?, status), stage = COALESCE(?, stage), "
                "error = CASE WHEN ? THEN NULL ELSE COALESCE(?, error) END, artifacts = ?, updated_at = ? "
                "WHERE id = ?",
                (status, stage, clear_error, error, json.dumps(merged, ensure_ascii=False), _now(), run_id),
            )

    def add_event(self, run_id: str, stage: str | None, kind: str, payload: dict[str, Any] | None = None) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO events (run_id, ts, stage, kind, payload) VALUES (?, ?, ?, ?, ?)",
                (run_id, _now(), stage, kind, json.dumps(payload or {}, ensure_ascii=False)),
            )
            return cursor.lastrowid

    def events(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, stage, kind, payload FROM events WHERE run_id = ? AND id > ? ORDER BY id",
                (run_id, after_id),
            ).fetchall()
        return [{"id": r[0], "ts": r[1], "stage": r[2], "kind": r[3], "payload": json.loads(r[4])} for r in rows]

    def mark_interrupted(self) -> list[str]:
        """At start-up: runs that were going have lost their thread."""

        with self._tx() as conn:
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM runs WHERE status IN (?, ?)", (QUEUED, RUNNING)).fetchall()]
            for run_id in ids:
                conn.execute("UPDATE runs SET status = ?, updated_at = ? WHERE id = ?", (INTERRUPTED, _now(), run_id))
                conn.execute(
                    "INSERT INTO events (run_id, ts, stage, kind, payload) VALUES (?, ?, NULL, 'interrupted', ?)",
                    (run_id, _now(), json.dumps({"reason": "backend restarted"})),
                )
        return ids

    def close(self) -> None:
        with self._lock:
            self._conn.close()
