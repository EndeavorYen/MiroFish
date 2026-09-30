"""Embedding fingerprint, re-embedding and vector storage of the local graph."""

from __future__ import annotations
import json
import logging
import os
from typing import Any
import sqlite_vec

logger = logging.getLogger("mirofish.graph")

from .local_db import (
    PROBE_TEXT,
    PROBE_MATCH,
    _cosine,
    _transaction,
    _Graph,
)


class VectorsMixin:
    """Part of LocalGraphStore (#68)."""

    def _probe(self) -> list[float]:
        if self._probe_vector is None:
            self._probe_vector = self.embedder.embed_documents([PROBE_TEXT])[0]
        return self._probe_vector

    def _warn_once(self, graph_id: str, message: str, *args: Any) -> None:
        if graph_id not in self._warned:
            self._warned.add(graph_id)
            logger.warning(message, *args)

    def embedding_state(self, graph_id: str) -> str:
        """``matches``, ``other embedder``, ``no fingerprint``, ``no vectors`` or
        ``unknown`` (the embedder did not answer), or ``incomplete`` (this
        embedder's vectors, but rows written while another one ran have none) (#61)."""

        graph = self._graph(graph_id)
        with graph.use():
            dim, stored = graph.current_dim(), graph.stored_probe()
        if dim is None:
            return "no vectors"
        if stored is None:
            return "no fingerprint"
        try:
            current = self._probe()
        except Exception as error:
            logger.warning("embedding fingerprint check failed: %s", error)
            return "unknown"
        if not self._same_probe(current, stored):
            return "other embedder"
        with graph.use():
            incomplete = graph.conn.execute(
                "SELECT 1 FROM meta WHERE key = 'vectors_incomplete'"
            ).fetchone()
        return "incomplete" if incomplete else "matches"

    @staticmethod
    def _same_probe(current: list[float], stored: list[float]) -> bool:
        return len(current) == len(stored) and _cosine(current, stored) >= PROBE_MATCH

    def vectors_usable(self, graph_id: str) -> bool:
        """Whether this store's embedder wrote the graph's vectors."""

        state = self.embedding_state(graph_id)
        if state == "no fingerprint":
            # Built before fingerprints: keep using its vectors, but say so.
            self._warn_once(
                graph_id,
                "graph %s has no embedding fingerprint; if the embedding model (or its GGUF) "
                "changed since it was built, run backend/scripts/reembed_graphs.py",
                graph_id,
            )
        elif state == "other embedder":
            self._warn_once(
                graph_id,
                "graph %s was embedded by another embedder; searching it by keywords only "
                "until backend/scripts/reembed_graphs.py re-embeds it",
                graph_id,
            )
        elif state == "incomplete":
            self._warn_once(
                graph_id,
                "graph %s has rows without vectors (written by another embedder); "
                "run backend/scripts/reembed_graphs.py",
                graph_id,
            )
        elif state == "unknown":
            # The query embedding would fail the same way: answer by keywords.
            self._warn_once(
                graph_id, "embedding service unavailable; searching graph %s by keywords only", graph_id
            )
        return state in ("matches", "no fingerprint", "incomplete")

    def reembed(self, graph_id: str) -> dict[str, int]:
        """Recompute every vector with this store's embedder.

        A maintenance step: run it while nothing else writes to the graph.
        """

        graph = self._graph(graph_id)
        with graph.use() as conn:
            nodes = conn.execute("SELECT rowid, name, labels, summary FROM nodes").fetchall()
            edges = conn.execute(
                "SELECT e.rowid, e.name, e.fact, e.dedupe_key, s.name, t.name FROM edges e "
                "LEFT JOIN nodes s ON s.uuid = e.source_uuid "
                "LEFT JOIN nodes t ON t.uuid = e.target_uuid"
            ).fetchall()
        node_texts = {row[0]: self._node_text(row[1], json.loads(row[2]), row[3]) for row in nodes}
        # The texts ingestion embeds: structured facts (_upsert_fact_edge) use
        # "relation fact", extracted relations (_upsert_edge) name both ends.
        edge_texts = {
            row[0]: f"{row[1]} {row[2]}" if row[3].startswith("fact:")
            else f"{row[4] or ''} {row[1]} {row[5] or ''} {row[2]}"
            for row in edges
        }
        vectors = self._embed(node_texts, edge_texts)
        with graph.use() as conn, _transaction(conn):
            graph.drop_vectors()
            if vectors:
                self._store_vectors(graph, vectors)
        self._warned.discard(graph_id)
        return {"nodes": len(node_texts), "edges": len(edge_texts)}

    def _embed(
        self, node_texts: dict[int, str], edge_texts: dict[int, str]
    ) -> dict[str, dict[int, tuple[str, list[float]]]]:
        result: dict[str, dict[int, tuple[str, list[float]]]] = {}
        if node_texts or edge_texts:
            self._probe()  # outside the graph lock; _store_vectors records it
        for table, texts in (("vec_nodes", node_texts), ("vec_edges", edge_texts)):
            if texts:
                rowids = list(texts)
                vectors = self.embedder.embed_documents([texts[r] for r in rowids])
                result[table] = {r: (texts[r], v) for r, v in zip(rowids, vectors)}
        return result

    def _store_vectors(
        self, graph: _Graph, vectors: dict[str, dict[int, tuple[str, list[float]]]]
    ) -> None:
        stored = graph.stored_probe()
        if stored is not None and vectors and not self._same_probe(self._probe(), stored):
            # Keyword search still finds the new rows; reembed restores vectors,
            # and --check reports the graph as incomplete until then.
            graph.conn.execute("INSERT OR REPLACE INTO meta VALUES ('vectors_incomplete', '1')")
            logger.warning(
                "not storing vectors in %s: another embedder wrote its vectors; "
                "run backend/scripts/reembed_graphs.py",
                os.path.basename(graph.path),
            )
            return
        for table, by_rowid in vectors.items():
            new = graph.current_dim() is None
            graph.ensure_vec_tables(len(next(iter(by_rowid.values()))[1]))
            if new:
                graph.set_probe(self._probe())
            for rowid, (text, vector) in by_rowid.items():
                if (
                    table == "vec_nodes"
                    and self._current_node_text(graph.conn, rowid) != text
                    and graph.conn.execute(
                        "SELECT 1 FROM vec_nodes WHERE rowid = ?", (rowid,)
                    ).fetchone()
                ):
                    # A concurrent episode merged newer text into this node
                    # and already stored a vector; keep that one. With no
                    # vector yet, an older vector beats none.
                    continue
                graph.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
                graph.conn.execute(
                    f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )
