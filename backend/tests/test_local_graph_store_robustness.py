"""Failure and concurrency behaviour of LocalGraphStore."""

import threading

import pytest

from app.config import Config
from app.graph.embedding import HashEmbedder
from app.graph.extractor import StubExtractor
from app.graph.local_store import LocalGraphStore, build_local_graph_store
from app.graph.store import EpisodeHandle, GraphNotFoundError, TextEpisode

LEXICON = {"Alice": "Person", "Bob": "Person", "Carol": "Person"}


class FlakyEmbedder(HashEmbedder):
    """Fails on the first ``fail_calls`` document batches."""

    def __init__(self, fail_calls: int) -> None:
        super().__init__()
        self.fail_calls = fail_calls

    def embed_documents(self, texts):
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise ConnectionError("embedding service down")
        return super().embed_documents(texts)


def test_failed_embedding_keeps_store_usable(tmp_path):
    store = LocalGraphStore(
        str(tmp_path), embedder=FlakyEmbedder(fail_calls=1), extractor=StubExtractor(LEXICON)
    )
    store.create_graph("g", graph_id="g1")
    with pytest.raises(ConnectionError):
        store.add_text_episodes("g1", [TextEpisode("Alice met Bob.")], durable=True)

    # Rows were committed without vectors: lexical search still works and
    # the episode is reported as not processed.
    assert store.search("g1", "Alice", "nodes", 3).nodes[0].name == "Alice"
    episode_rows = store._graph("g1").conn.execute(
        "SELECT uuid, processed FROM episodes"
    ).fetchall()
    assert [row[1] for row in episode_rows] == [0]
    with pytest.raises(RuntimeError):
        store.wait_until_processed(EpisodeHandle([episode_rows[0][0]]))

    handle = store.add_text_episodes("g1", [TextEpisode("Carol met Alice.")], durable=True)
    assert store.wait_until_processed(handle) == handle.episode_ids
    assert store.search("g1", "Carol", "nodes", 3).nodes[0].name == "Carol"
    store.close()


def test_services_share_one_store_per_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "GRAPH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "GRAPH_EMBEDDER", "hash")
    first = build_local_graph_store()
    second = build_local_graph_store()
    assert first is second

    first.create_graph("g", graph_id="g1")
    first.add_text_episodes("g1", [TextEpisode("Alice.")], durable=True)
    second.delete_graph("g1")
    with pytest.raises(GraphNotFoundError):
        first.list_nodes("g1")
    first.close()
    assert build_local_graph_store() is not first
    build_local_graph_store().close()


def test_handle_used_after_delete_raises_not_found(tmp_path):
    store = LocalGraphStore(str(tmp_path), embedder=HashEmbedder(), extractor=StubExtractor(LEXICON))
    store.create_graph("g", graph_id="g1")
    graph = store._graph("g1")
    store.delete_graph("g1")
    with pytest.raises(GraphNotFoundError):
        with graph.use():
            pass
    assert not (tmp_path / "g1.sqlite").exists()
    store.close()


def test_concurrent_writers_and_readers(tmp_path):
    store = LocalGraphStore(str(tmp_path), embedder=HashEmbedder(), extractor=StubExtractor(LEXICON))
    store.create_graph("g", graph_id="g1")
    errors = []

    def write(i):
        try:
            store.add_text_episodes("g1", [TextEpisode(f"Alice met Bob {i}.")], durable=True)
        except Exception as error:  # pragma: no cover - surfaced below
            errors.append(error)

    def read():
        try:
            for _ in range(20):
                store.search("g1", "Alice", "edges", 5)
                store.list_nodes("g1")
        except Exception as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    threads += [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert {n.name for n in store.list_nodes("g1")} == {"Alice", "Bob"}
    assert len(store.list_edges("g1")) == 8
    store.close()


def test_stale_node_vector_is_not_written_over_newer_text(tmp_path):
    import sqlite_vec

    store = LocalGraphStore(str(tmp_path), embedder=HashEmbedder(), extractor=StubExtractor(LEXICON))
    store.create_graph("g", graph_id="g1")
    store.add_text_episodes("g1", [TextEpisode("Alice.")], durable=True)
    graph = store._graph("g1")
    rowid = graph.conn.execute("SELECT rowid FROM nodes WHERE name = 'Alice'").fetchone()[0]
    before = graph.conn.execute("SELECT embedding FROM vec_nodes WHERE rowid = ?", (rowid,)).fetchone()

    stale = {"vec_nodes": {rowid: ("Alice Person some older summary", [1.0] + [0.0] * 255)}}
    with graph.use():
        store._store_vectors(graph, stale)
    after = graph.conn.execute("SELECT embedding FROM vec_nodes WHERE rowid = ?", (rowid,)).fetchone()
    assert after == before
    assert after[0] != sqlite_vec.serialize_float32([1.0] + [0.0] * 255)
    store.close()
