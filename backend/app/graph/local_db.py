"""SQLite helpers, schema and constants shared by the local graph store (#68)."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite_vec
from .store import (
    GraphNotFoundError,
)



# A fixed text whose vector identifies the embedder that wrote a graph's
# vectors (#61): a mis-converted and a correct e5-small GGUF are both 384-d.
PROBE_TEXT = "embedding fingerprint: 東海市宣佈啟動無人駕駛計程車 / Taxi drivers oppose the air taxi pilot"


PROBE_MATCH = 0.99  # cosine; the two e5-small GGUFs differ at 0.78-0.95


RRF_K = 60


MAX_SUMMARY_CHARS = 1200


MAX_ALIASES = 50


_GRAPH_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


# Role editing (#64): node_keys rows for names renamed or merged away, and
# the attribute marking a role the user excluded from preparation.
_ALIAS_KEY = "name:"


ROLE_EXCLUDED = "role_excluded"


_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ontology (id INTEGER PRIMARY KEY CHECK (id = 1), body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    rowid INTEGER PRIMARY KEY,
    uuid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT UNIQUE NOT NULL,
    labels TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL DEFAULT '{}',
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    rowid INTEGER PRIMARY KEY,
    uuid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_uuid TEXT NOT NULL,
    target_uuid TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    created_at TEXT,
    valid_at TEXT,
    invalid_at TEXT,
    expired_at TEXT,
    episodes TEXT NOT NULL DEFAULT '[]',
    fact_type TEXT,
    dedupe_key TEXT UNIQUE NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_source ON edges(source_uuid);
CREATE INDEX IF NOT EXISTS edges_target ON edges(target_uuid);
CREATE TABLE IF NOT EXISTS episodes (
    uuid TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT,
    source TEXT,
    metadata TEXT,
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS episode_links (
    episode_uuid TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('node', 'edge')),
    target_uuid TEXT NOT NULL,
    PRIMARY KEY (episode_uuid, kind, target_uuid)
);
CREATE TABLE IF NOT EXISTS node_keys (key TEXT PRIMARY KEY, node_uuid TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(body);
CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(body);
"""


_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS graphs (graph_id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS node_index (node_uuid TEXT PRIMARY KEY, graph_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episode_index (episode_uuid TEXT PRIMARY KEY, graph_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS node_index_graph ON node_index(graph_id);
CREATE INDEX IF NOT EXISTS episode_index_graph ON episode_index(graph_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_name(name: str) -> str:
    """Key for entity identity: NFKC, case-folded, whitespace removed."""

    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name or "")).casefold()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        # SQLite may already have rolled back; never let that hide the
        # original error or leave the connection inside a transaction.
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise


class _Graph:
    """One open graph database. All access goes through ``use()``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.closed = False
        self.conn = _connect(path)
        self.conn.executescript(_GRAPH_SCHEMA)

    @contextmanager
    def use(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            if self.closed:
                raise GraphNotFoundError(
                    f"graph was deleted: {os.path.basename(self.path)}"
                )
            yield self.conn

    def current_dim(self) -> int | None:
        # Read every time: a rolled-back transaction or another process may
        # have changed it, so an in-memory copy can go stale.
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_dim'"
        ).fetchone()
        return int(row[0]) if row else None

    def stored_probe(self) -> list[float] | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_probe'"
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_probe(self, vector: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('embedding_probe', ?)", (json.dumps(vector),)
        )

    def drop_vectors(self) -> None:
        for table in ("vec_nodes", "vec_edges"):
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.execute(
            "DELETE FROM meta WHERE key IN ('embedding_dim', 'embedding_probe', 'vectors_incomplete')"
        )

    def ensure_vec_tables(self, dim: int) -> None:
        current = self.current_dim()
        if current is None:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('embedding_dim', ?)", (str(dim),)
            )
            for table in ("vec_nodes", "vec_edges"):
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
                    f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
                )
        elif dim != current:
            raise ValueError(f"embedding dimension changed from {current} to {dim}")

    def close(self) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                self.conn.close()
