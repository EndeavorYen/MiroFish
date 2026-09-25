"""Local ``GraphStore``: one SQLite file per graph, FTS5 + sqlite-vec search.

Layout under ``data_dir``::

    registry.sqlite        graphs, node -> graph and episode -> graph index
    <graph_id>.sqlite      nodes, edges, episodes, ontology, FTS5 and vec0

Labels follow Zep: every node carries ``"Entity"`` plus its ontology type, so
``ZepEntityReader.filter_defined_entities`` works unchanged.

Search runs BM25 (FTS5 over CJK-bigram text) and cosine KNN (sqlite-vec)
and merges the two rankings with reciprocal rank fusion (k=60). Both
``ranking`` values use that fusion; there is no cross-encoder locally.

Ingestion is synchronous: ``add_text_episodes`` extracts and writes each
episode in its own transaction, so ``wait_until_processed`` only verifies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid as uuidlib
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal

import sqlite_vec

from .embedding import Embedder, make_embedder
from .extractor import ExtractedEntity, ExtractedRelation, Extraction, Extractor, StubExtractor
from .store import (
    EpisodeHandle,
    GraphEdge,
    GraphNode,
    GraphNotFoundError,
    IngestionHandle,
    ProgressCallback,
    SearchResult,
    TextEpisode,
)
from .text_index import index_text, match_query

RRF_K = 60
MAX_SUMMARY_CHARS = 1200
_GRAPH_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

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


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class _Graph:
    """One open graph database. All access is serialised by ``lock``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.conn = _connect(path)
        self.conn.executescript(_GRAPH_SCHEMA)
        self.dim = self._meta_int("embedding_dim")

    def _meta_int(self, key: str) -> int | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return int(row[0]) if row else None

    def ensure_vec_tables(self, dim: int) -> None:
        if self.dim is None:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_dim', ?)", (str(dim),))
            for table in ("vec_nodes", "vec_edges"):
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
                    f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
                )
            self.dim = dim
        elif dim != self.dim:
            raise ValueError(f"embedding dimension changed from {self.dim} to {dim}")

    def close(self) -> None:
        with self.lock:
            self.conn.close()


class LocalGraphStore:
    def __init__(
        self,
        data_dir: str,
        *,
        embedder: Embedder,
        extractor: Extractor,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.embedder = embedder
        self.extractor = extractor
        self._registry_lock = threading.RLock()
        self._registry = _connect(os.path.join(self.data_dir, "registry.sqlite"))
        self._registry.executescript(_REGISTRY_SCHEMA)
        self._graphs: dict[str, _Graph] = {}
        self._graphs_lock = threading.Lock()

    # ------------------------------------------------------------------ graphs

    def _graph_path(self, graph_id: str) -> str:
        if not _GRAPH_ID_RE.match(graph_id or ""):
            raise ValueError(f"invalid graph_id: {graph_id!r}")
        return os.path.join(self.data_dir, f"{graph_id}.sqlite")

    def _exists(self, graph_id: str) -> bool:
        with self._registry_lock:
            row = self._registry.execute(
                "SELECT 1 FROM graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
        return row is not None

    def _graph(self, graph_id: str) -> _Graph:
        if not self._exists(graph_id):
            raise GraphNotFoundError(f"graph not found: {graph_id}")
        with self._graphs_lock:
            graph = self._graphs.get(graph_id)
            if graph is None:
                graph = _Graph(self._graph_path(graph_id))
                self._graphs[graph_id] = graph
            return graph

    def create_graph(
        self,
        name: str,
        *,
        graph_id: str | None = None,
        graph_id_callback: Callable[[str], None] | None = None,
    ) -> str:
        graph_id = graph_id or f"mirofish_{uuidlib.uuid4().hex[:16]}"
        self._graph_path(graph_id)
        if graph_id_callback:
            graph_id_callback(graph_id)
        with self._registry_lock:
            self._registry.execute(
                "INSERT OR IGNORE INTO graphs VALUES (?, ?, ?)", (graph_id, name, _now())
            )
        self._graph(graph_id)
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        if not self._exists(graph_id):
            raise GraphNotFoundError(f"graph not found: {graph_id}")
        with self._graphs_lock:
            graph = self._graphs.pop(graph_id, None)
        if graph is not None:
            graph.close()
        with self._registry_lock:
            self._registry.execute("BEGIN")
            self._registry.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
            self._registry.execute("DELETE FROM node_index WHERE graph_id = ?", (graph_id,))
            self._registry.execute("DELETE FROM episode_index WHERE graph_id = ?", (graph_id,))
            self._registry.execute("COMMIT")
        path = self._graph_path(graph_id)
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except FileNotFoundError:
                pass

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        graph = self._graph(graph_id)
        with graph.lock:
            graph.conn.execute(
                "INSERT OR REPLACE INTO ontology (id, body) VALUES (1, ?)",
                (json.dumps(ontology, ensure_ascii=False),),
            )

    def get_ontology(self, graph_id: str) -> dict[str, Any] | None:
        graph = self._graph(graph_id)
        with graph.lock:
            row = graph.conn.execute("SELECT body FROM ontology WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else None

    # --------------------------------------------------------------- ingestion

    def add_text_episodes(
        self,
        graph_id: str,
        episodes: list[TextEpisode],
        *,
        durable: bool,
        batch_size: int = 350,
        on_submitted: Callable[[str | None, str], None] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionHandle:
        """Extract and write each episode now.

        Every local write is durable (one SQLite transaction per episode), so
        ``durable`` does not change behaviour. ``on_submitted`` is never
        called: there is no remote batch identity to journal.
        """

        graph = self._graph(graph_id)
        ontology = self.get_ontology(graph_id)
        episode_ids: list[str] = []
        total = len(episodes)
        for index, episode in enumerate(episodes, 1):
            episode_id = uuidlib.uuid4().hex
            extraction = self.extractor.extract(episode.text, ontology)
            with graph.lock:
                self._write_episode(graph, graph_id, episode_id, episode, extraction)
            episode_ids.append(episode_id)
            if on_progress:
                on_progress(f"processed episode {index}/{total}", index / total)
        return EpisodeHandle(episode_ids)

    def _write_episode(
        self,
        graph: _Graph,
        graph_id: str,
        episode_id: str,
        episode: TextEpisode,
        extraction: Extraction,
    ) -> None:
        conn = graph.conn
        created_at = episode.created_at or _now()
        node_texts: dict[int, str] = {}
        edge_texts: dict[int, str] = {}
        new_nodes: list[str] = []
        conn.execute("BEGIN")
        try:
            conn.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, 0)",
                (
                    episode_id,
                    episode.text,
                    created_at,
                    episode.source,
                    json.dumps(episode.metadata or {}, ensure_ascii=False),
                ),
            )
            uuids_by_key: dict[str, str] = {}
            for entity in extraction.entities:
                node_uuid, rowid, text, is_new = self._upsert_node(conn, entity, created_at)
                uuids_by_key[normalize_name(entity.name)] = node_uuid
                node_texts[rowid] = text
                if is_new:
                    new_nodes.append(node_uuid)
                self._link(conn, episode_id, "node", node_uuid)
            for relation in extraction.relations:
                source = uuids_by_key.get(normalize_name(relation.source))
                target = uuids_by_key.get(normalize_name(relation.target))
                if source is None:
                    source = self._node_uuid_by_name(conn, relation.source)
                if target is None:
                    target = self._node_uuid_by_name(conn, relation.target)
                if source is None or target is None:
                    continue  # relation endpoints must be extracted entities
                edge_uuid, rowid, text = self._upsert_edge(
                    conn, relation, source, target, episode_id, created_at
                )
                edge_texts[rowid] = text
                self._link(conn, episode_id, "edge", edge_uuid)
            self._index_embeddings(graph, node_texts, edge_texts)
            conn.execute("UPDATE episodes SET processed = 1 WHERE uuid = ?", (episode_id,))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        with self._registry_lock:
            self._registry.execute("BEGIN")
            self._registry.execute(
                "INSERT OR REPLACE INTO episode_index VALUES (?, ?)", (episode_id, graph_id)
            )
            self._registry.executemany(
                "INSERT OR REPLACE INTO node_index VALUES (?, ?)",
                [(node_uuid, graph_id) for node_uuid in new_nodes],
            )
            self._registry.execute("COMMIT")

    @staticmethod
    def _link(conn: sqlite3.Connection, episode_id: str, kind: str, target: str) -> None:
        conn.execute("INSERT OR IGNORE INTO episode_links VALUES (?, ?, ?)", (episode_id, kind, target))

    @staticmethod
    def _node_uuid_by_name(conn: sqlite3.Connection, name: str) -> str | None:
        row = conn.execute(
            "SELECT uuid FROM nodes WHERE name_key = ?", (normalize_name(name),)
        ).fetchone()
        return row[0] if row else None

    def _upsert_node(
        self, conn: sqlite3.Connection, entity: ExtractedEntity, created_at: str
    ) -> tuple[str, int, str, bool]:
        key = normalize_name(entity.name)
        row = conn.execute(
            "SELECT rowid, uuid, labels, summary, attributes FROM nodes WHERE name_key = ?", (key,)
        ).fetchone()
        type_labels = [entity.entity_type] if entity.entity_type else []
        if row is None:
            node_uuid = uuidlib.uuid4().hex
            labels = ["Entity", *[t for t in type_labels if t != "Entity"]]
            summary = entity.summary[:MAX_SUMMARY_CHARS]
            attributes = dict(entity.attributes)
            cursor = conn.execute(
                "INSERT INTO nodes (uuid, name, name_key, labels, summary, attributes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    node_uuid,
                    entity.name,
                    key,
                    json.dumps(labels, ensure_ascii=False),
                    summary,
                    json.dumps(attributes, ensure_ascii=False),
                    created_at,
                ),
            )
            rowid = cursor.lastrowid
            is_new = True
        else:
            rowid, node_uuid = row[0], row[1]
            labels = json.loads(row[2])
            for label in type_labels:
                if label not in labels:
                    labels.append(label)
            summary = row[3]
            if entity.summary and entity.summary not in summary:
                summary = f"{summary} {entity.summary}".strip()[:MAX_SUMMARY_CHARS]
            attributes = json.loads(row[4])
            for k, v in entity.attributes.items():
                attributes.setdefault(k, v)
            conn.execute(
                "UPDATE nodes SET labels = ?, summary = ?, attributes = ? WHERE rowid = ?",
                (
                    json.dumps(labels, ensure_ascii=False),
                    summary,
                    json.dumps(attributes, ensure_ascii=False),
                    rowid,
                ),
            )
            is_new = False
        text = f"{entity.name} {' '.join(labels[1:])} {summary}".strip()
        conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (rowid,))
        conn.execute("INSERT INTO nodes_fts (rowid, body) VALUES (?, ?)", (rowid, index_text(text)))
        return node_uuid, rowid, text, is_new

    def _upsert_edge(
        self,
        conn: sqlite3.Connection,
        relation: ExtractedRelation,
        source: str,
        target: str,
        episode_id: str,
        created_at: str,
    ) -> tuple[str, int, str]:
        dedupe_key = hashlib.sha256(
            "\0".join([source, relation.name, target, relation.fact]).encode("utf-8")
        ).hexdigest()
        row = conn.execute(
            "SELECT rowid, uuid, episodes FROM edges WHERE dedupe_key = ?", (dedupe_key,)
        ).fetchone()
        names = conn.execute(
            "SELECT uuid, name FROM nodes WHERE uuid IN (?, ?)", (source, target)
        ).fetchall()
        name_of = dict(names)
        text = f"{name_of.get(source, '')} {relation.name} {name_of.get(target, '')} {relation.fact}"
        if row is None:
            edge_uuid = uuidlib.uuid4().hex
            cursor = conn.execute(
                "INSERT INTO edges (uuid, name, fact, source_uuid, target_uuid, attributes, "
                "created_at, valid_at, episodes, fact_type, dedupe_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge_uuid,
                    relation.name,
                    relation.fact,
                    source,
                    target,
                    json.dumps(relation.attributes, ensure_ascii=False),
                    created_at,
                    created_at,
                    json.dumps([episode_id]),
                    relation.name,
                    dedupe_key,
                ),
            )
            rowid = cursor.lastrowid
            conn.execute(
                "INSERT INTO edges_fts (rowid, body) VALUES (?, ?)", (rowid, index_text(text))
            )
            return edge_uuid, rowid, text
        rowid, edge_uuid = row[0], row[1]
        episodes = json.loads(row[2])
        if episode_id not in episodes:
            episodes.append(episode_id)
            conn.execute(
                "UPDATE edges SET episodes = ? WHERE rowid = ?", (json.dumps(episodes), rowid)
            )
        return edge_uuid, rowid, text

    def _index_embeddings(
        self, graph: _Graph, node_texts: dict[int, str], edge_texts: dict[int, str]
    ) -> None:
        for table, texts in (("vec_nodes", node_texts), ("vec_edges", edge_texts)):
            if not texts:
                continue
            rowids = list(texts)
            vectors = self.embedder.embed_documents([texts[r] for r in rowids])
            graph.ensure_vec_tables(len(vectors[0]))
            for rowid, vector in zip(rowids, vectors):
                graph.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
                graph.conn.execute(
                    f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )

    def wait_until_processed(
        self,
        handle: IngestionHandle,
        *,
        deadline: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[str]:
        ids = list(handle.episode_ids)
        pending = self._unprocessed(ids)
        if pending:
            # Local ingestion is synchronous; unprocessed ids mean a failed
            # or unknown episode, which will not finish by waiting.
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"{len(pending)} local episode(s) not processed")
            raise RuntimeError(f"local episodes not processed: {pending[:5]}")
        if on_progress:
            on_progress(f"processed {len(ids)} episode(s)", 1.0)
        return ids

    def _unprocessed(self, episode_ids: list[str]) -> list[str]:
        if not episode_ids:
            return []
        with self._registry_lock:
            rows = self._registry.execute(
                f"SELECT episode_uuid, graph_id FROM episode_index WHERE episode_uuid IN "
                f"({','.join('?' * len(episode_ids))})",
                episode_ids,
            ).fetchall()
        graph_of = dict(rows)
        pending = []
        for episode_id in episode_ids:
            graph_id = graph_of.get(episode_id)
            if graph_id is None:
                pending.append(episode_id)
                continue
            graph = self._graph(graph_id)
            with graph.lock:
                row = graph.conn.execute(
                    "SELECT processed FROM episodes WHERE uuid = ?", (episode_id,)
                ).fetchone()
            if not row or not row[0]:
                pending.append(episode_id)
        return pending

    # ------------------------------------------------------------------- reads

    @staticmethod
    def _node(row: sqlite3.Row | tuple) -> GraphNode:
        return GraphNode(
            uuid=row[0],
            name=row[1],
            labels=json.loads(row[2]),
            summary=row[3],
            attributes=json.loads(row[4]),
            created_at=row[5],
        )

    @staticmethod
    def _edge(row: sqlite3.Row | tuple) -> GraphEdge:
        return GraphEdge(
            uuid=row[0],
            name=row[1],
            fact=row[2],
            source_node_uuid=row[3],
            target_node_uuid=row[4],
            attributes=json.loads(row[5]),
            created_at=row[6],
            valid_at=row[7],
            invalid_at=row[8],
            expired_at=row[9],
            episodes=json.loads(row[10]),
            fact_type=row[11],
        )

    _NODE_COLS = "uuid, name, labels, summary, attributes, created_at"
    _EDGE_COLS = (
        "uuid, name, fact, source_uuid, target_uuid, attributes, created_at, "
        "valid_at, invalid_at, expired_at, episodes, fact_type"
    )

    def list_nodes(self, graph_id: str) -> list[GraphNode]:
        graph = self._graph(graph_id)
        with graph.lock:
            rows = graph.conn.execute(
                f"SELECT {self._NODE_COLS} FROM nodes ORDER BY rowid"
            ).fetchall()
        return [self._node(r) for r in rows]

    def list_edges(self, graph_id: str) -> list[GraphEdge]:
        graph = self._graph(graph_id)
        with graph.lock:
            rows = graph.conn.execute(
                f"SELECT {self._EDGE_COLS} FROM edges ORDER BY rowid"
            ).fetchall()
        return [self._edge(r) for r in rows]

    def _graph_of_node(self, node_uuid: str) -> _Graph:
        with self._registry_lock:
            row = self._registry.execute(
                "SELECT graph_id FROM node_index WHERE node_uuid = ?", (node_uuid,)
            ).fetchone()
        if row is None:
            raise GraphNotFoundError(f"node not found: {node_uuid}")
        return self._graph(row[0])

    def get_node(self, node_uuid: str) -> GraphNode:
        graph = self._graph_of_node(node_uuid)
        with graph.lock:
            row = graph.conn.execute(
                f"SELECT {self._NODE_COLS} FROM nodes WHERE uuid = ?", (node_uuid,)
            ).fetchone()
        if row is None:
            raise GraphNotFoundError(f"node not found: {node_uuid}")
        return self._node(row)

    def get_node_edges(self, node_uuid: str) -> list[GraphEdge]:
        graph = self._graph_of_node(node_uuid)
        with graph.lock:
            rows = graph.conn.execute(
                f"SELECT {self._EDGE_COLS} FROM edges "
                "WHERE source_uuid = ? OR target_uuid = ? ORDER BY rowid",
                (node_uuid, node_uuid),
            ).fetchall()
        return [self._edge(r) for r in rows]

    def search(
        self,
        graph_id: str,
        query: str,
        scope: Literal["edges", "nodes"],
        limit: int,
        *,
        ranking: Literal["relevance", "fusion"] = "relevance",
    ) -> SearchResult:
        if scope not in ("edges", "nodes"):
            raise ValueError(f"scope must be edges or nodes, got {scope!r}")
        graph = self._graph(graph_id)
        limit = max(1, int(limit))
        pool = max(limit * 4, 20)
        fts, vec, table, cols = (
            ("nodes_fts", "vec_nodes", "nodes", self._NODE_COLS)
            if scope == "nodes"
            else ("edges_fts", "vec_edges", "edges", self._EDGE_COLS)
        )
        query_vector = self.embedder.embed_query(query) if query.strip() else None
        expression = match_query(query)
        scores: dict[int, float] = {}
        with graph.lock:
            if expression:
                lexical = graph.conn.execute(
                    f"SELECT rowid FROM {fts} WHERE {fts} MATCH ? ORDER BY bm25({fts}) LIMIT ?",
                    (expression, pool),
                ).fetchall()
                for rank, (rowid,) in enumerate(lexical):
                    scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if query_vector is not None and graph.dim is not None:
                semantic = graph.conn.execute(
                    f"SELECT rowid FROM {vec} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (sqlite_vec.serialize_float32(query_vector), pool),
                ).fetchall()
                for rank, (rowid,) in enumerate(semantic):
                    scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank + 1)
            best = sorted(scores, key=lambda r: (-scores[r], r))[:limit]
            if not best:
                return SearchResult(edges=[], nodes=[])
            rows = graph.conn.execute(
                f"SELECT rowid, {cols} FROM {table} WHERE rowid IN "
                f"({','.join('?' * len(best))})",
                best,
            ).fetchall()
        by_rowid = {row[0]: row[1:] for row in rows}
        ordered = [by_rowid[r] for r in best if r in by_rowid]
        if scope == "nodes":
            return SearchResult(edges=[], nodes=[self._node(r) for r in ordered])
        return SearchResult(edges=[self._edge(r) for r in ordered], nodes=[])

    def close(self) -> None:
        with self._graphs_lock:
            graphs = list(self._graphs.values())
            self._graphs.clear()
        for graph in graphs:
            graph.close()
        with self._registry_lock:
            self._registry.close()


def make_extractor() -> Extractor:
    from ..config import Config

    kind = Config.GRAPH_EXTRACTOR
    if kind == "stub":
        return StubExtractor()
    raise ValueError(f"unknown GRAPH_EXTRACTOR: {kind!r}")


def build_local_graph_store() -> LocalGraphStore:
    from ..config import Config

    return LocalGraphStore(
        Config.GRAPH_DATA_DIR,
        embedder=make_embedder(),
        extractor=make_extractor(),
    )
