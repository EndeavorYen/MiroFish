"""
NetworkX + SQLite backed implementation of GraphStore and TextStore.

Uses NetworkX DiGraph for in-memory graph algorithms (community detection,
traversal) and SQLite for persistence and full-text search (FTS5).
"""

import json
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

import networkx as nx

from app.services.graph_store import (
    EntityNode,
    RelationEdge,
    SearchResult,
    Community,
    generate_uuid,
    normalize_name,
)


# Keep module-private alias for backward compatibility within this file
_normalize_name = normalize_name


class NetworkXGraphStore:
    """Combined GraphStore + TextStore backed by NetworkX and SQLite."""

    def __init__(self, db_path: str = ":memory:"):
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._graph_meta: Dict[str, dict] = {}
        self._db_path = db_path
        self._db_lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    # ── Schema setup ────────────────────────────────────────────────

    def _init_tables(self):
        with self._db_lock:
            c = self._conn
            c.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    uuid TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    attributes TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            c.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_dedup
                ON entities(graph_id, name_normalized, entity_type)
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_name
                ON entities(graph_id, name_normalized)
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    uuid TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    source_uuid TEXT NOT NULL,
                    target_uuid TEXT NOT NULL,
                    name TEXT NOT NULL,
                    fact TEXT NOT NULL DEFAULT '',
                    attributes TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    valid_at TEXT,
                    invalid_at TEXT,
                    expired_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS texts (
                    uuid TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    tick INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            # FTS5 virtual table for full-text search on texts
            try:
                c.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS texts_fts
                    USING fts5(uuid, graph_id, content, content=texts,
                               content_rowid='rowid')
                """)
            except sqlite3.OperationalError:
                pass  # already exists or unsupported
            # Trigger to keep FTS in sync
            try:
                c.execute("""
                    CREATE TRIGGER IF NOT EXISTS texts_ai AFTER INSERT ON texts BEGIN
                        INSERT INTO texts_fts(rowid, uuid, graph_id, content)
                        VALUES (new.rowid, new.uuid, new.graph_id, new.content);
                    END
                """)
            except sqlite3.OperationalError:
                pass
            c.commit()

    # ── Graph lifecycle ─────────────────────────────────────────────

    def create_graph(self, graph_id: str, name: str, description: str) -> str:
        if graph_id in self._graphs:
            raise ValueError(f"Graph '{graph_id}' already exists")
        self._graphs[graph_id] = nx.DiGraph()
        self._graph_meta[graph_id] = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        self._graphs.pop(graph_id, None)
        self._graph_meta.pop(graph_id, None)
        with self._db_lock:
            self._conn.execute("DELETE FROM entities WHERE graph_id = ?", (graph_id,))
            self._conn.execute("DELETE FROM relations WHERE graph_id = ?", (graph_id,))
            self._conn.execute("DELETE FROM texts WHERE graph_id = ?", (graph_id,))
            self._conn.commit()

    # ── Entity operations ───────────────────────────────────────────

    def add_entity(
        self,
        graph_id: str,
        name: str,
        entity_type: str,
        summary: str,
        attributes: dict,
    ) -> str:
        norm = _normalize_name(name)
        g = self._graphs.get(graph_id)
        if g is None:
            raise ValueError(f"Graph '{graph_id}' does not exist")

        with self._db_lock:
            row = self._conn.execute(
                "SELECT uuid, summary, attributes FROM entities "
                "WHERE graph_id = ? AND name_normalized = ? AND entity_type = ?",
                (graph_id, norm, entity_type),
            ).fetchone()

            if row:
                # Dedup: update with richer data
                existing_uuid = row["uuid"]
                existing_attrs = json.loads(row["attributes"])
                new_summary = summary if len(summary) > len(row["summary"]) else row["summary"]
                merged_attrs = {**existing_attrs, **attributes}
                self._conn.execute(
                    "UPDATE entities SET summary = ?, attributes = ? WHERE uuid = ?",
                    (new_summary, json.dumps(merged_attrs), existing_uuid),
                )
                self._conn.commit()
                # Update in-memory graph node
                g.nodes[existing_uuid]["summary"] = new_summary
                g.nodes[existing_uuid]["attributes"] = merged_attrs
                return existing_uuid
            else:
                entity_uuid = generate_uuid()
                now = datetime.now().isoformat()
                self._conn.execute(
                    "INSERT INTO entities (uuid, graph_id, name, name_normalized, "
                    "entity_type, summary, attributes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (entity_uuid, graph_id, name, norm, entity_type,
                     summary, json.dumps(attributes), now),
                )
                self._conn.commit()
                g.add_node(
                    entity_uuid,
                    name=name,
                    entity_type=entity_type,
                    labels=[entity_type],
                    summary=summary,
                    attributes=attributes,
                )
                return entity_uuid

    def get_entity(self, graph_id: str, uuid: str) -> EntityNode:
        g = self._graphs.get(graph_id)
        if g is None or uuid not in g:
            raise ValueError(f"Entity '{uuid}' not found in graph '{graph_id}'")
        data = g.nodes[uuid]
        return EntityNode(
            uuid=uuid,
            name=data["name"],
            labels=data.get("labels", [data.get("entity_type", "Unknown")]),
            summary=data.get("summary", ""),
            attributes=data.get("attributes", {}),
        )

    def find_entity_by_name(
        self, graph_id: str, name: str
    ) -> Optional[str]:
        """Find entity UUID by name (case-insensitive). Returns None if not found."""
        norm = _normalize_name(name)
        with self._db_lock:
            row = self._conn.execute(
                "SELECT uuid FROM entities WHERE graph_id = ? AND name_normalized = ? LIMIT 1",
                (graph_id, norm),
            ).fetchone()
            return row["uuid"] if row else None

    def list_entities(
        self, graph_id: str, limit: int = 100, cursor: Optional[str] = None
    ) -> List[EntityNode]:
        g = self._graphs.get(graph_id)
        if g is None:
            return []
        nodes = []
        for nid, data in g.nodes(data=True):
            nodes.append(
                EntityNode(
                    uuid=nid,
                    name=data["name"],
                    labels=data.get("labels", [data.get("entity_type", "Unknown")]),
                    summary=data.get("summary", ""),
                    attributes=data.get("attributes", {}),
                )
            )
            if len(nodes) >= limit:
                break
        return nodes

    def get_entity_edges(
        self, graph_id: str, uuid: str
    ) -> List[RelationEdge]:
        g = self._graphs.get(graph_id)
        if g is None:
            return []
        edges = []
        # Outgoing
        for _, tgt, data in g.out_edges(uuid, data=True):
            edges.append(self._edge_data_to_relation(uuid, tgt, data))
        # Incoming
        for src, _, data in g.in_edges(uuid, data=True):
            edges.append(self._edge_data_to_relation(src, uuid, data))
        return edges

    # ── Relation operations ─────────────────────────────────────────

    def add_relation(
        self,
        graph_id: str,
        source_uuid: str,
        target_uuid: str,
        relation_type: str,
        fact: str,
        attributes: Optional[dict] = None,
        temporal: Optional[dict] = None,
    ) -> str:
        g = self._graphs.get(graph_id)
        if g is None:
            raise ValueError(f"Graph '{graph_id}' does not exist")

        attributes = attributes or {}
        temporal = temporal or {}
        rel_uuid = generate_uuid()
        now = datetime.now()

        valid_at = self._parse_dt(temporal.get("valid_at"))
        invalid_at = self._parse_dt(temporal.get("invalid_at"))
        expired_at = self._parse_dt(temporal.get("expired_at"))

        with self._db_lock:
            self._conn.execute(
                "INSERT INTO relations (uuid, graph_id, source_uuid, target_uuid, "
                "name, fact, attributes, created_at, valid_at, invalid_at, expired_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rel_uuid, graph_id, source_uuid, target_uuid,
                    relation_type, fact, json.dumps(attributes),
                    now.isoformat(),
                    valid_at.isoformat() if valid_at else None,
                    invalid_at.isoformat() if invalid_at else None,
                    expired_at.isoformat() if expired_at else None,
                ),
            )
            self._conn.commit()

        weight = attributes.get("weight", 1.0)
        g.add_edge(
            source_uuid,
            target_uuid,
            uuid=rel_uuid,
            name=relation_type,
            fact=fact,
            attributes=attributes,
            created_at=now,
            valid_at=valid_at,
            invalid_at=invalid_at,
            expired_at=expired_at,
            weight=weight,
        )
        return rel_uuid

    def list_relations(
        self, graph_id: str, limit: int = 100, cursor: Optional[str] = None
    ) -> List[RelationEdge]:
        g = self._graphs.get(graph_id)
        if g is None:
            return []
        relations = []
        for src, tgt, data in g.edges(data=True):
            relations.append(self._edge_data_to_relation(src, tgt, data))
            if len(relations) >= limit:
                break
        return relations

    # ── Search ──────────────────────────────────────────────────────

    def search(
        self, graph_id: str, query: str, scope: str = "all", limit: int = 10
    ) -> SearchResult:
        nodes: List[EntityNode] = []
        edges: List[RelationEdge] = []
        facts: List[str] = []
        q_lower = query.lower()

        g = self._graphs.get(graph_id)

        if scope in ("nodes", "all") and g is not None:
            for nid, data in g.nodes(data=True):
                name = data.get("name", "")
                summary = data.get("summary", "")
                if q_lower in name.lower() or q_lower in summary.lower():
                    nodes.append(
                        EntityNode(
                            uuid=nid,
                            name=name,
                            labels=data.get("labels", []),
                            summary=summary,
                            attributes=data.get("attributes", {}),
                        )
                    )
                    facts.append(f"{name}: {summary}")
                    if len(nodes) >= limit:
                        break

        if scope in ("edges", "all") and g is not None:
            for src, tgt, data in g.edges(data=True):
                fact_text = data.get("fact", "")
                rel_name = data.get("name", "")
                if q_lower in fact_text.lower() or q_lower in rel_name.lower():
                    edges.append(self._edge_data_to_relation(src, tgt, data))
                    if fact_text:
                        facts.append(fact_text)
                    if len(edges) >= limit:
                        break

        if scope in ("text", "all"):
            text_results = self.search_text(graph_id, query, limit=limit)
            for r in text_results:
                facts.append(r["content"])

        return SearchResult(nodes=nodes, edges=edges, facts=facts)

    # ── Text store ──────────────────────────────────────────────────

    def add_text(
        self,
        graph_id: str,
        agent_id: str,
        content: str,
        action_type: str,
        metadata: dict,
        tick: int,
        timestamp: datetime,
    ) -> str:
        text_uuid = generate_uuid()
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO texts (uuid, graph_id, agent_id, content, action_type, "
                "metadata, tick, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    text_uuid, graph_id, agent_id, content, action_type,
                    json.dumps(metadata), tick, timestamp.isoformat(),
                ),
            )
            self._conn.commit()
        return text_uuid

    def search_text(
        self, graph_id: str, query: str, limit: int = 20
    ) -> List[dict]:
        with self._db_lock:
            # Try FTS5 first; fall back to LIKE for CJK / short queries
            try:
                rows = self._conn.execute(
                    "SELECT t.* FROM texts t JOIN texts_fts f ON t.uuid = f.uuid "
                    "WHERE f.content MATCH ? AND t.graph_id = ? LIMIT ?",
                    (query, graph_id, limit),
                ).fetchall()
                if rows:
                    return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass
            # Fallback: LIKE
            rows = self._conn.execute(
                "SELECT * FROM texts WHERE graph_id = ? AND content LIKE ? LIMIT ?",
                (graph_id, f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_agent(
        self, graph_id: str, agent_id: str, limit: int = 50
    ) -> List[dict]:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM texts WHERE graph_id = ? AND agent_id = ? "
                "ORDER BY tick LIMIT ?",
                (graph_id, agent_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_time_range(
        self, graph_id: str, start: datetime, end: datetime
    ) -> List[dict]:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM texts WHERE graph_id = ? "
                "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                (graph_id, start.isoformat(), end.isoformat()),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_active_facts(self, graph_id: str) -> List[dict]:
        """Return relations that are currently valid (not expired)."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM relations WHERE graph_id = ? "
                "AND (expired_at IS NULL OR expired_at = '')",
                (graph_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_historical_facts(self, graph_id: str) -> List[dict]:
        """Return relations that have expired."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM relations WHERE graph_id = ? "
                "AND expired_at IS NOT NULL AND expired_at != ''",
                (graph_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Community detection ─────────────────────────────────────────

    def detect_communities(
        self, graph_id: str, algorithm: str = "louvain"
    ) -> List[Community]:
        g = self._graphs.get(graph_id)
        if g is None or len(g) == 0:
            return []
        # Louvain requires undirected graph
        ug = g.to_undirected()
        communities_sets = nx.community.louvain_communities(
            ug, weight="weight", seed=42
        )
        result = []
        for idx, members in enumerate(communities_sets):
            result.append(Community(id=idx, members=list(members)))
        return result

    # ── Snapshot / restore ──────────────────────────────────────────

    def snapshot(self, graph_id: str) -> bytes:
        g = self._graphs.get(graph_id)
        if g is None:
            raise ValueError(f"Graph '{graph_id}' does not exist")
        meta = self._graph_meta.get(graph_id, {})
        data = {
            "graph_id": graph_id,
            "meta": meta,
            "graph": nx.node_link_data(g),
        }
        return json.dumps(data, default=str).encode("utf-8")

    def restore(self, data: bytes) -> str:
        payload = json.loads(data.decode("utf-8"))
        graph_id = payload["graph_id"]
        meta = payload.get("meta", {})
        graph_data = payload["graph"]
        g = nx.node_link_graph(graph_data, directed=True)

        # Use a new graph_id if the original already exists
        restored_id = graph_id
        if restored_id in self._graphs:
            restored_id = f"{graph_id}-restored-{generate_uuid()[:8]}"

        self._graphs[restored_id] = g
        self._graph_meta[restored_id] = meta

        # Re-insert entities and relations into SQLite
        with self._db_lock:
            now = datetime.now().isoformat()
            for nid, ndata in g.nodes(data=True):
                self._conn.execute(
                    "INSERT OR REPLACE INTO entities "
                    "(uuid, graph_id, name, name_normalized, entity_type, "
                    "summary, attributes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        nid, restored_id, ndata.get("name", ""),
                        _normalize_name(ndata.get("name", "")),
                        ndata.get("entity_type", "Unknown"),
                        ndata.get("summary", ""),
                        json.dumps(ndata.get("attributes", {})),
                        now,
                    ),
                )
            for src, tgt, edata in g.edges(data=True):
                self._conn.execute(
                    "INSERT OR REPLACE INTO relations "
                    "(uuid, graph_id, source_uuid, target_uuid, name, fact, "
                    "attributes, created_at, valid_at, invalid_at, expired_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        edata.get("uuid", generate_uuid()),
                        restored_id, src, tgt,
                        edata.get("name", ""),
                        edata.get("fact", ""),
                        json.dumps(edata.get("attributes", {})),
                        now, None, None, None,
                    ),
                )
            self._conn.commit()
        return restored_id

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _edge_data_to_relation(src: str, tgt: str, data: dict) -> RelationEdge:
        return RelationEdge(
            uuid=data.get("uuid", ""),
            name=data.get("name", ""),
            fact=data.get("fact", ""),
            source_node_uuid=src,
            target_node_uuid=tgt,
            attributes=data.get("attributes", {}),
            created_at=data.get("created_at", datetime.now()),
            valid_at=data.get("valid_at"),
            invalid_at=data.get("invalid_at"),
            expired_at=data.get("expired_at"),
        )

    @staticmethod
    def _parse_dt(value) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return None
