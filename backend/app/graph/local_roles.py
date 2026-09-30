"""Role confirmation: rename, merge, exclude, include, add (#64)."""

from __future__ import annotations
import hashlib
import json
import logging
import sqlite3
import uuid as uuidlib
from datetime import datetime, timezone
from .store import (
    GraphNotFoundError,
)
from .text_index import index_text

logger = logging.getLogger("mirofish.graph")

from .local_db import (
    MAX_SUMMARY_CHARS,
    MAX_ALIASES,
    _ALIAS_KEY,
    ROLE_EXCLUDED,
    normalize_name,
    _transaction,
    _Graph,
)


class RoleEditingMixin:
    """Part of LocalGraphStore (#68)."""

    def _entity_types(self, conn: sqlite3.Connection) -> set[str]:
        row = conn.execute("SELECT body FROM ontology WHERE id = 1").fetchone()
        ontology = json.loads(row[0]) if row else {}
        return {t.get("name") for t in ontology.get("entity_types", [])}

    def _role_row(self, conn: sqlite3.Connection, node_uuid: str) -> tuple:
        row = conn.execute(
            "SELECT rowid, name, labels, summary, attributes FROM nodes WHERE uuid = ?", (node_uuid,)
        ).fetchone()
        if row is None or "Entity" not in json.loads(row[2]):
            raise GraphNotFoundError(f"role not found: {node_uuid}")
        return row

    @staticmethod
    def _add_alias(conn: sqlite3.Connection, attributes: dict, alias: str, node_uuid: str) -> None:
        aliases = [a for a in attributes.get("aliases", []) if a != alias]
        attributes["aliases"] = [alias, *aliases][:MAX_ALIASES]
        conn.execute(
            "INSERT OR REPLACE INTO node_keys VALUES (?, ?)", (_ALIAS_KEY + normalize_name(alias), node_uuid)
        )

    def _refresh_role_texts(self, graph: _Graph, node_uuids: list[str]) -> None:
        """FTS rows now, vectors of the nodes and their edges after (#61 formats)."""

        with graph.use() as conn:
            nodes = conn.execute(
                f"SELECT rowid, name, labels, summary FROM nodes WHERE uuid IN ({','.join('?' * len(node_uuids))})",
                node_uuids,
            ).fetchall()
            edges = conn.execute(
                "SELECT e.rowid, e.name, e.fact, e.dedupe_key, s.name, t.name FROM edges e "
                "LEFT JOIN nodes s ON s.uuid = e.source_uuid LEFT JOIN nodes t ON t.uuid = e.target_uuid "
                f"WHERE e.source_uuid IN ({','.join('?' * len(node_uuids))}) "
                f"OR e.target_uuid IN ({','.join('?' * len(node_uuids))})",
                [*node_uuids, *node_uuids],
            ).fetchall()
        node_texts = {row[0]: self._node_text(row[1], json.loads(row[2]), row[3]) for row in nodes}
        edge_texts = {
            row[0]: f"{row[1]} {row[2]}" if row[3].startswith("fact:")
            else f"{row[4] or ''} {row[1]} {row[5] or ''} {row[2]}"
            for row in edges
        }
        with graph.use() as conn, _transaction(conn):
            for rowid, text in node_texts.items():
                conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (rowid,))
                conn.execute("INSERT INTO nodes_fts (rowid, body) VALUES (?, ?)", (rowid, index_text(text)))
            for rowid, text in edge_texts.items():
                conn.execute("DELETE FROM edges_fts WHERE rowid = ?", (rowid,))
                conn.execute("INSERT INTO edges_fts (rowid, body) VALUES (?, ?)", (rowid, index_text(text)))
        # The edit is committed; a vector failure must not read as a failed
        # edit. Keyword search already sees the change.
        try:
            vectors = self._embed(node_texts, edge_texts)
            with graph.use() as conn, _transaction(conn):
                self._store_vectors(graph, vectors)
        except Exception as error:
            logger.warning("role edit: vectors not refreshed (%s); run reembed_graphs.py", error)
            try:  # so embedding_state() and --check report it
                with graph.use() as conn, _transaction(conn):
                    conn.execute("INSERT OR REPLACE INTO meta VALUES ('vectors_incomplete', '1')")
            except Exception as flag_error:
                logger.warning("role edit: could not flag vectors_incomplete (%s)", flag_error)

    @staticmethod
    def _delete_edge(conn: sqlite3.Connection, graph: _Graph, rowid: int) -> None:
        conn.execute("DELETE FROM edges WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM edges_fts WHERE rowid = ?", (rowid,))
        if graph.current_dim() is not None:
            conn.execute("DELETE FROM vec_edges WHERE rowid = ?", (rowid,))

    def _rekey_moved_edges(self, conn: sqlite3.Connection, graph: _Graph, rowids: list[int]) -> None:
        """Extracted edges key on their endpoints (_upsert_edge); after a merge
        the key must use the kept node, and an edge that now repeats another
        one folds into it (its episodes join the survivor's)."""

        for rowid in rowids:
            row = conn.execute(
                "SELECT name, fact, source_uuid, target_uuid, dedupe_key, episodes FROM edges WHERE rowid = ?",
                (rowid,),
            ).fetchone()
            if row is None or row[4].startswith("fact:"):
                continue
            key = hashlib.sha256("\0".join([row[2], row[0], row[3], row[1]]).encode("utf-8")).hexdigest()
            if key == row[4]:
                continue
            twin = conn.execute("SELECT rowid, episodes FROM edges WHERE dedupe_key = ?", (key,)).fetchone()
            if twin is None:
                conn.execute("UPDATE edges SET dedupe_key = ? WHERE rowid = ?", (key, rowid))
                continue
            episodes = json.loads(twin[1])
            episodes += [e for e in json.loads(row[5]) if e not in episodes]
            conn.execute("UPDATE edges SET episodes = ? WHERE rowid = ?", (json.dumps(episodes), twin[0]))
            self._delete_edge(conn, graph, rowid)

    def rename_node(self, graph_id: str, node_uuid: str, new_name: str) -> None:
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("name is empty")
        graph = self._graph(graph_id)
        with graph.use() as conn, _transaction(conn):
            rowid, old_name, _, _, attributes = self._role_row(conn, node_uuid)
            key = normalize_name(new_name)
            clash = conn.execute("SELECT uuid FROM nodes WHERE name_key = ?", (key,)).fetchone()
            if clash and clash[0] != node_uuid:
                raise ValueError(f"{new_name!r} is another role; merge the two instead")
            attributes = json.loads(attributes)
            if normalize_name(old_name) != key:
                self._add_alias(conn, attributes, old_name, node_uuid)
            conn.execute("DELETE FROM node_keys WHERE key = ?", (_ALIAS_KEY + key,))
            conn.execute(
                "UPDATE nodes SET name = ?, name_key = ?, attributes = ? WHERE rowid = ?",
                (new_name, key, json.dumps(attributes, ensure_ascii=False), rowid),
            )
        self._refresh_role_texts(graph, [node_uuid])

    def merge_nodes(self, graph_id: str, keep_uuid: str, drop_uuid: str) -> None:
        if keep_uuid == drop_uuid:
            raise ValueError("cannot merge a role with itself")
        graph = self._graph(graph_id)
        with graph.use() as conn, _transaction(conn):
            keep_rowid, _, keep_labels, keep_summary, keep_attrs = self._role_row(conn, keep_uuid)
            drop_rowid, drop_name, _, drop_summary, drop_attrs = self._role_row(conn, drop_uuid)
            attributes = json.loads(keep_attrs)
            for alias in [drop_name, *json.loads(drop_attrs).get("aliases", [])]:
                self._add_alias(conn, attributes, alias, keep_uuid)
            summary = keep_summary
            if drop_summary and drop_summary not in summary:
                summary = f"{summary} {drop_summary}".strip()[:MAX_SUMMARY_CHARS]
            conn.execute(
                "UPDATE nodes SET summary = ?, attributes = ? WHERE rowid = ?",
                (summary, json.dumps(attributes, ensure_ascii=False), keep_rowid),
            )
            moved = conn.execute(
                "SELECT rowid FROM edges WHERE source_uuid = ? OR target_uuid = ?", (drop_uuid, drop_uuid)
            ).fetchall()
            conn.execute("UPDATE edges SET source_uuid = ? WHERE source_uuid = ?", (keep_uuid, drop_uuid))
            conn.execute("UPDATE edges SET target_uuid = ? WHERE target_uuid = ?", (keep_uuid, drop_uuid))
            self._rekey_moved_edges(conn, graph, [rowid for (rowid,) in moved])
            # An edge between the two became a loop: drop it.
            for (rowid,) in conn.execute(
                "SELECT rowid FROM edges WHERE source_uuid = ? AND target_uuid = ?", (keep_uuid, keep_uuid)
            ).fetchall():
                self._delete_edge(conn, graph, rowid)
            conn.execute(
                "UPDATE OR IGNORE episode_links SET target_uuid = ? WHERE kind = 'node' AND target_uuid = ?",
                (keep_uuid, drop_uuid),
            )
            conn.execute("DELETE FROM episode_links WHERE kind = 'node' AND target_uuid = ?", (drop_uuid,))
            conn.execute("UPDATE node_keys SET node_uuid = ? WHERE node_uuid = ?", (keep_uuid, drop_uuid))
            conn.execute("DELETE FROM nodes WHERE rowid = ?", (drop_rowid,))
            conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (drop_rowid,))
            if graph.current_dim() is not None:
                conn.execute("DELETE FROM vec_nodes WHERE rowid = ?", (drop_rowid,))
        with self._registry_lock, _transaction(self._registry):
            self._registry.execute("DELETE FROM node_index WHERE node_uuid = ?", (drop_uuid,))
        self._refresh_role_texts(graph, [keep_uuid])

    def set_node_role(self, graph_id: str, node_uuid: str, entity_type: str | None) -> None:
        """``None`` excludes the role from preparation; a type includes it."""

        graph = self._graph(graph_id)
        with graph.use() as conn, _transaction(conn):
            rowid, _, labels, _, attributes = self._role_row(conn, node_uuid)
            if entity_type is not None and entity_type not in self._entity_types(conn):
                raise ValueError(f"{entity_type!r} is not an entity type of this graph")
            attributes = json.loads(attributes)
            if entity_type is None:
                attributes[ROLE_EXCLUDED] = True
            else:
                attributes.pop(ROLE_EXCLUDED, None)
            labels = ["Entity"] + ([entity_type] if entity_type else [])
            conn.execute(
                "UPDATE nodes SET labels = ?, attributes = ? WHERE rowid = ?",
                (json.dumps(labels, ensure_ascii=False), json.dumps(attributes, ensure_ascii=False), rowid),
            )
        self._refresh_role_texts(graph, [node_uuid])

    def add_role_node(self, graph_id: str, name: str, entity_type: str, summary: str = "") -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("name is empty")
        graph = self._graph(graph_id)
        node_uuid = uuidlib.uuid4().hex
        with graph.use() as conn, _transaction(conn):
            if entity_type not in self._entity_types(conn):
                raise ValueError(f"{entity_type!r} is not an entity type of this graph")
            key = normalize_name(name)
            if conn.execute("SELECT 1 FROM nodes WHERE name_key = ?", (key,)).fetchone():
                raise ValueError(f"{name!r} is already a role")
            conn.execute(
                "INSERT INTO nodes (uuid, name, name_key, labels, summary, attributes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    node_uuid, name, key, json.dumps(["Entity", entity_type], ensure_ascii=False),
                    (summary or "")[:MAX_SUMMARY_CHARS],
                    json.dumps({"added_by": "user"}, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        with self._registry_lock, _transaction(self._registry):
            self._registry.execute("INSERT OR REPLACE INTO node_index VALUES (?, ?)", (node_uuid, graph_id))
        self._refresh_role_texts(graph, [node_uuid])
        return node_uuid
