"""Local ``GraphStore``: one SQLite file per graph, FTS5 + sqlite-vec search.

Layout under ``data_dir``::

    registry.sqlite        graphs, node -> graph and episode -> graph index
    <graph_id>.sqlite      nodes, edges, episodes, ontology, FTS5 and vec0

Labels follow Zep: every node carries ``"Entity"`` plus its ontology type, so
``EntityReader.filter_defined_entities`` works unchanged.

Search runs BM25 (FTS5 over CJK-bigram text) and cosine KNN (sqlite-vec)
and merges the two rankings with reciprocal rank fusion (k=60). Both
``ranking`` values use that fusion; there is no cross-encoder locally.

Ingestion is synchronous: ``add_text_episodes`` extracts and writes each
episode in its own transaction, so ``wait_until_processed`` only verifies.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid as uuidlib
from collections.abc import Callable
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

logger = logging.getLogger("mirofish.graph")

# Split in #68; these names stay importable from here.
from .local_db import (  # noqa: F401
    PROBE_TEXT,
    PROBE_MATCH,
    RRF_K,
    MAX_SUMMARY_CHARS,
    MAX_ALIASES,
    _GRAPH_ID_RE,
    _ALIAS_KEY,
    ROLE_EXCLUDED,
    _GRAPH_SCHEMA,
    _REGISTRY_SCHEMA,
    _now,
    normalize_name,
    _cosine,
    _connect,
    _transaction,
    _Graph,
)
from .local_vectors import VectorsMixin
from .local_facts import StructuredFactsMixin
from .local_roles import RoleEditingMixin


class LocalGraphStore(VectorsMixin, StructuredFactsMixin, RoleEditingMixin):
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
        self._probe_vector: list[float] | None = None
        self._warned: set[str] = set()
        self._registry_lock = threading.RLock()
        self._registry = _connect(os.path.join(self.data_dir, "registry.sqlite"))
        self._registry.executescript(_REGISTRY_SCHEMA)
        # Lock order: _graphs_lock -> graph.lock -> _registry_lock.
        self._graphs: dict[str, _Graph] = {}
        self._graphs_lock = threading.RLock()
        self.closed = False

    def _graph_path(self, graph_id: str) -> str:
        if not _GRAPH_ID_RE.match(graph_id or ""):
            raise ValueError(f"invalid graph_id: {graph_id!r}")
        return os.path.join(self.data_dir, f"{graph_id}.sqlite")

    def graph_ids(self) -> list[str]:
        with self._registry_lock:
            rows = self._registry.execute("SELECT graph_id FROM graphs ORDER BY graph_id").fetchall()
        return [row[0] for row in rows]

    def _exists(self, graph_id: str) -> bool:
        with self._registry_lock:
            row = self._registry.execute(
                "SELECT 1 FROM graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
        return row is not None

    def _graph(self, graph_id: str) -> _Graph:
        with self._graphs_lock:
            if not self._exists(graph_id):
                raise GraphNotFoundError(f"graph not found: {graph_id}")
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
        with self._graphs_lock:
            with self._registry_lock, _transaction(self._registry):
                self._registry.execute(
                    "INSERT OR IGNORE INTO graphs VALUES (?, ?, ?)", (graph_id, name, _now())
                )
            self._graph(graph_id)
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        with self._graphs_lock:
            if not self._exists(graph_id):
                raise GraphNotFoundError(f"graph not found: {graph_id}")
            graph = self._graphs.pop(graph_id, None)
            if graph is not None:
                graph.close()  # waits for in-flight operations on this graph
            path = self._graph_path(graph_id)
            # Files first: if Windows refuses (file held elsewhere) the
            # registry still lists the graph, so the delete can be retried.
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except FileNotFoundError:
                    pass
            with self._registry_lock, _transaction(self._registry):
                self._registry.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
                self._registry.execute("DELETE FROM node_index WHERE graph_id = ?", (graph_id,))
                self._registry.execute(
                    "DELETE FROM episode_index WHERE graph_id = ?", (graph_id,)
                )

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        graph = self._graph(graph_id)
        with graph.use() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ontology (id, body) VALUES (1, ?)",
                (json.dumps(ontology, ensure_ascii=False),),
            )

    def get_ontology(self, graph_id: str) -> dict[str, Any] | None:
        graph = self._graph(graph_id)
        with graph.use() as conn:
            row = conn.execute("SELECT body FROM ontology WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else None

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

        Every local write is durable (SQLite transactions), so ``durable``
        does not change behaviour. ``on_submitted`` is never called: there is
        no remote batch identity to journal.
        """

        graph = self._graph(graph_id)
        ontology = self.get_ontology(graph_id)
        episode_ids: list[str] = []
        total = len(episodes)
        for index, episode in enumerate(episodes, 1):
            episode_id = uuidlib.uuid4().hex
            extraction = self.extractor.extract(
                episode.text, ontology, known_entities=self._known_entities(graph)
            )
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
        """Rows, then embeddings outside any lock, then vectors + processed.

        The episode is marked processed only after its vectors are stored, so
        a failed embedding call leaves it unprocessed (BM25 still finds the
        rows) and ``wait_until_processed`` reports it.
        """

        created_at = episode.created_at or _now()
        node_texts: dict[int, str] = {}
        edge_texts: dict[int, str] = {}
        new_nodes: list[str] = []
        with graph.use() as conn, _transaction(conn):
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
        with self._registry_lock, _transaction(self._registry):
            self._registry.execute(
                "INSERT OR REPLACE INTO episode_index VALUES (?, ?)", (episode_id, graph_id)
            )
            self._registry.executemany(
                "INSERT OR REPLACE INTO node_index VALUES (?, ?)",
                [(node_uuid, graph_id) for node_uuid in new_nodes],
            )
        vectors = self._embed(node_texts, edge_texts)
        with graph.use() as conn, _transaction(conn):
            self._store_vectors(graph, vectors)
            conn.execute("UPDATE episodes SET processed = 1 WHERE uuid = ?", (episode_id,))

    @staticmethod
    def _known_entities(graph: _Graph) -> list[tuple[str, str]]:
        with graph.use() as conn:
            rows = conn.execute("SELECT name, labels FROM nodes ORDER BY rowid").fetchall()
        known = []
        for name, labels in rows:
            labels = json.loads(labels)
            if "Entity" not in labels:
                continue  # simulation posts, comments and agents (#8)
            types = [label for label in labels if label not in ("Entity", "Node")]
            known.append((name, types[0] if types else "Entity"))
        return known

    @staticmethod
    def _link(conn: sqlite3.Connection, episode_id: str, kind: str, target: str) -> None:
        conn.execute("INSERT OR IGNORE INTO episode_links VALUES (?, ?, ?)", (episode_id, kind, target))

    @classmethod
    def _node_uuid_by_name_or_alias(cls, conn: sqlite3.Connection, name: str) -> str | None:
        found = cls._node_uuid_by_name(conn, name)
        if found:
            return found
        needle = json.dumps(name, ensure_ascii=False)
        for uuid_, attributes in conn.execute(
            "SELECT uuid, attributes FROM nodes WHERE attributes LIKE ?", (f"%{needle}%",)
        ):
            if name in (json.loads(attributes).get("aliases") or []):
                return uuid_
        return None

    @staticmethod
    def _node_uuid_by_name(conn: sqlite3.Connection, name: str) -> str | None:
        row = conn.execute(
            "SELECT uuid FROM nodes WHERE name_key = ?", (normalize_name(name),)
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _node_text(name: str, labels: list[str], summary: str) -> str:
        return f"{name} {' '.join(labels[1:])} {summary}".strip()

    def _current_node_text(self, conn: sqlite3.Connection, rowid: int) -> str | None:
        row = conn.execute(
            "SELECT name, labels, summary FROM nodes WHERE rowid = ?", (rowid,)
        ).fetchone()
        return None if row is None else self._node_text(row[0], json.loads(row[1]), row[2])

    def _upsert_node(
        self, conn: sqlite3.Connection, entity: ExtractedEntity, created_at: str
    ) -> tuple[str, int, str, bool]:
        key = normalize_name(entity.name)
        row = conn.execute(
            "SELECT rowid, uuid, labels, summary, attributes, name FROM nodes WHERE name_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            # A name the user renamed or merged away (#64).
            row = conn.execute(
                "SELECT n.rowid, n.uuid, n.labels, n.summary, n.attributes, n.name FROM node_keys k "
                "JOIN nodes n ON n.uuid = k.node_uuid WHERE k.key = ?",
                (_ALIAS_KEY + key,),
            ).fetchone()
        type_labels = [entity.entity_type] if entity.entity_type else []
        if row is not None and json.loads(row[4]).get(ROLE_EXCLUDED):
            type_labels = []  # the user excluded this role
        if row is None:
            node_uuid = uuidlib.uuid4().hex
            labels = ["Entity", *[t for t in type_labels if t != "Entity"]]
            summary = entity.summary[:MAX_SUMMARY_CHARS]
            attributes = dict(entity.attributes)
            if isinstance(attributes.get("aliases"), list):
                attributes["aliases"] = attributes["aliases"][:MAX_ALIASES]
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
                if isinstance(v, list) and isinstance(attributes.get(k), list):
                    # Union list attributes such as aliases across episodes
                    # (values may be unhashable).
                    merged = list(attributes[k])
                    for item in v:
                        if item not in merged:
                            merged.append(item)
                    attributes[k] = merged[:MAX_ALIASES] if k == "aliases" else merged
                else:
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
        text = self._node_text(entity.name if row is None else row[5], labels, summary)
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
            with graph.use():
                row = graph.conn.execute(
                    "SELECT processed FROM episodes WHERE uuid = ?", (episode_id,)
                ).fetchone()
            if not row or not row[0]:
                pending.append(episode_id)
        return pending

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
        with graph.use():
            rows = graph.conn.execute(
                f"SELECT {self._NODE_COLS} FROM nodes ORDER BY rowid"
            ).fetchall()
        return [self._node(r) for r in rows]

    def list_edges(self, graph_id: str) -> list[GraphEdge]:
        graph = self._graph(graph_id)
        with graph.use():
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
        with graph.use():
            row = graph.conn.execute(
                f"SELECT {self._NODE_COLS} FROM nodes WHERE uuid = ?", (node_uuid,)
            ).fetchone()
        if row is None:
            raise GraphNotFoundError(f"node not found: {node_uuid}")
        return self._node(row)

    def get_node_edges(self, node_uuid: str) -> list[GraphEdge]:
        graph = self._graph_of_node(node_uuid)
        with graph.use():
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
        query_vector = (
            self.embedder.embed_query(query)
            if query.strip() and self.vectors_usable(graph_id)
            else None
        )
        expression = match_query(query)
        scores: dict[int, float] = {}
        with graph.use():
            if expression:
                lexical = graph.conn.execute(
                    f"SELECT rowid FROM {fts} WHERE {fts} MATCH ? ORDER BY bm25({fts}) LIMIT ?",
                    (expression, pool),
                ).fetchall()
                for rank, (rowid,) in enumerate(lexical):
                    scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if query_vector is not None and graph.current_dim() is not None:
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
            self.closed = True
        for graph in graphs:
            graph.close()
        with self._registry_lock:
            self._registry.close()


def make_extractor() -> Extractor:
    from ..config import Config

    kind = Config.GRAPH_EXTRACTOR
    if kind == "stub":
        return StubExtractor()
    if kind == "local":
        from ..system_one.client import get_system_one_client
        from .local_extractor import LocalExtractor, llm_decode_fn, llm_summary_fn

        if Config.LOCAL_NER not in ("candidates", "gliner", "decode"):
            raise ValueError(
                f"LOCAL_NER must be candidates, gliner or decode, got {Config.LOCAL_NER!r}"
            )
        return LocalExtractor(
            get_system_one_client(),
            embedder=make_embedder(),
            ner=Config.LOCAL_NER,
            gliner_model=Config.LOCAL_NER_GLINER_MODEL,
            summary_fn=llm_summary_fn() if Config.EXTRACT_SUMMARY_LLM else None,
            decode_fn=llm_decode_fn() if Config.LOCAL_NER == "decode" else None,
        )
    raise ValueError(f"unknown GRAPH_EXTRACTOR: {kind!r}")


_SHARED: dict[str, LocalGraphStore] = {}


_SHARED_LOCK = threading.Lock()


def build_local_graph_store() -> LocalGraphStore:
    """One shared store per data directory.

    Services each call ``get_graph_store()``; sharing one instance keeps one
    connection and lock per graph, so writers, readers and ``delete_graph``
    coordinate instead of racing on separate handles.
    """

    from ..config import Config

    key = os.path.abspath(Config.GRAPH_DATA_DIR)
    with _SHARED_LOCK:
        store = _SHARED.get(key)
        if store is None or store.closed:
            store = LocalGraphStore(key, embedder=make_embedder(), extractor=make_extractor())
            _SHARED[key] = store
        return store
