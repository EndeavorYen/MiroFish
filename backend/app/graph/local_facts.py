"""Structured facts written without extraction (#8)."""

from __future__ import annotations
import json
import sqlite3
import uuid as uuidlib
from typing import Any
from .store import (
    FactNode,
    StructuredFact,
)
from .text_index import index_text


from .local_db import (
    _now,
    normalize_name,
    _transaction,
)


class StructuredFactsMixin:
    """Part of LocalGraphStore (#68)."""

    def add_structured_facts(self, graph_id: str, facts: list[StructuredFact]) -> list[str]:
        """Write facts as nodes and edges without any model call (#8).

        Idempotent: a fact whose ``key`` was already written changes nothing,
        so replaying an action log keeps node and edge counts unchanged.
        Simulation nodes are labelled ``["Node"]`` (kind in attributes) so the
        entity filter never mistakes a post for an ontology entity; a
        ``FactNode`` whose name matches an existing entity attaches to it.
        """

        graph = self._graph(graph_id)
        node_texts: dict[int, str] = {}
        edge_texts: dict[int, str] = {}
        new_nodes: list[str] = []
        edge_ids: list[str] = []
        with graph.use() as conn, _transaction(conn):
            for fact in facts:
                created_at = fact.created_at or _now()
                keyed = {}
                for node in (fact.source, fact.target, *fact.extra_nodes):
                    keyed[node.key] = self._upsert_fact_node(
                        conn, node, created_at, node_texts, new_nodes
                    )
                edge_ids.append(
                    self._upsert_fact_edge(
                        conn,
                        f"fact:{fact.key}",
                        fact.relation,
                        keyed[fact.source.key],
                        keyed[fact.target.key],
                        fact.fact,
                        fact.attributes,
                        created_at,
                        edge_texts,
                    )
                )
                for source_key, relation, target_key in fact.extra_edges:
                    self._upsert_fact_edge(
                        conn,
                        f"fact:{fact.key}:{source_key}:{relation}:{target_key}",
                        relation,
                        keyed[source_key],
                        keyed[target_key],
                        fact.fact,
                        fact.attributes,
                        created_at,
                        edge_texts,
                    )
                for name in fact.mentions:
                    entity = self._node_uuid_by_name_or_alias(conn, name)
                    if entity is None or entity == keyed[fact.target.key]:
                        continue
                    self._upsert_fact_edge(
                        conn,
                        # Keyed by entity: a name and its alias give one edge.
                        f"fact:{fact.key}:mentions:{entity}",
                        "MENTIONS",
                        keyed[fact.target.key],
                        entity,
                        f"{fact.target.name} 提到 {name}",
                        fact.attributes,
                        created_at,
                        edge_texts,
                    )
        if new_nodes:
            with self._registry_lock, _transaction(self._registry):
                self._registry.executemany(
                    "INSERT OR REPLACE INTO node_index VALUES (?, ?)",
                    [(node_uuid, graph_id) for node_uuid in new_nodes],
                )
        vectors = self._embed(node_texts, edge_texts)
        if vectors:
            with graph.use() as conn, _transaction(conn):
                self._store_vectors(graph, vectors)
        return edge_ids

    def _upsert_fact_node(
        self,
        conn: sqlite3.Connection,
        node: FactNode,
        created_at: str,
        node_texts: dict[int, str],
        new_nodes: list[str],
    ) -> str:
        row = conn.execute("SELECT node_uuid FROM node_keys WHERE key = ?", (node.key,)).fetchone()
        if row:
            self._requeue_missing_node_vector(conn, row[0], node_texts)
            return row[0]
        # A simulated agent attaches to the ontology entity of the same name;
        # otherwise agents are identified per simulation scope and name, so
        # runs never share agent nodes. Posts and comments never attach by name.
        existing = None
        if node.label == "SimAgent":
            row = conn.execute(
                "SELECT uuid, labels FROM nodes WHERE name_key = ?", (normalize_name(node.name),)
            ).fetchone()
            if row and "Entity" in json.loads(row[1]):
                existing = row
            else:
                existing = conn.execute(
                    "SELECT uuid FROM nodes WHERE name_key = ?", (self._fact_name_key(node),)
                ).fetchone()
        if existing:
            node_uuid = existing[0]
        else:
            node_uuid = uuidlib.uuid4().hex
            attributes = {"kind": node.label, **node.attributes}
            # name_key must stay unique; simulation objects (posts) can share
            # display names, so their key includes the caller key.
            cursor = conn.execute(
                "INSERT INTO nodes (uuid, name, name_key, labels, summary, attributes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    node_uuid,
                    node.name,
                    self._fact_name_key(node),
                    json.dumps(["Node"]),
                    node.summary,
                    json.dumps(attributes, ensure_ascii=False),
                    created_at,
                ),
            )
            text = self._node_text(node.name, ["Node"], node.summary)
            conn.execute(
                "INSERT INTO nodes_fts (rowid, body) VALUES (?, ?)",
                (cursor.lastrowid, index_text(text)),
            )
            node_texts[cursor.lastrowid] = text
            new_nodes.append(node_uuid)
        conn.execute("INSERT INTO node_keys VALUES (?, ?)", (node.key, node_uuid))
        return node_uuid

    @staticmethod
    def _fact_name_key(node: FactNode) -> str:
        if node.label == "SimAgent":
            return f"sim:{node.attributes.get('scope', '')}:{normalize_name(node.name)}"
        return f"key:{node.key}"

    def _requeue_missing_node_vector(
        self, conn: sqlite3.Connection, node_uuid: str, node_texts: dict[int, str]
    ) -> None:
        """A replay re-embeds nodes whose vector write failed earlier."""

        row = conn.execute(
            "SELECT rowid, name, labels, summary FROM nodes WHERE uuid = ?", (node_uuid,)
        ).fetchone()
        if row and not self._has_vector(conn, "vec_nodes", row[0]):
            node_texts[row[0]] = self._node_text(row[1], json.loads(row[2]), row[3])

    @staticmethod
    def _has_vector(conn: sqlite3.Connection, table: str, rowid: int) -> bool:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)
        ).fetchone()
        if not exists:
            return False
        return conn.execute(f"SELECT 1 FROM {table} WHERE rowid = ?", (rowid,)).fetchone() is not None

    @classmethod
    def _upsert_fact_edge(
        cls,
        conn: sqlite3.Connection,
        dedupe_key: str,
        relation: str,
        source: str,
        target: str,
        fact: str,
        attributes: dict[str, Any],
        created_at: str,
        edge_texts: dict[int, str],
    ) -> str:
        row = conn.execute(
            "SELECT uuid, rowid, name, fact FROM edges WHERE dedupe_key = ?", (dedupe_key,)
        ).fetchone()
        if row:
            if not cls._has_vector(conn, "vec_edges", row[1]):
                edge_texts[row[1]] = f"{row[2]} {row[3]}"  # replay restores the vector
            return row[0]
        edge_uuid = uuidlib.uuid4().hex
        cursor = conn.execute(
            "INSERT INTO edges (uuid, name, fact, source_uuid, target_uuid, attributes, "
            "created_at, valid_at, episodes, fact_type, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?)",
            (
                edge_uuid,
                relation,
                fact,
                source,
                target,
                json.dumps(attributes, ensure_ascii=False),
                created_at,
                created_at,
                relation,
                dedupe_key,
            ),
        )
        text = f"{relation} {fact}"
        conn.execute(
            "INSERT INTO edges_fts (rowid, body) VALUES (?, ?)", (cursor.lastrowid, index_text(text))
        )
        edge_texts[cursor.lastrowid] = text
        return edge_uuid
