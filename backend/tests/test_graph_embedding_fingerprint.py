"""A graph remembers which embedder wrote its vectors (#61).

The broken e5-small GGUF and a correct one both give 384-dimensional
vectors, so the dimension alone cannot tell them apart.
"""

import httpx
import pytest

from app.graph.embedding import HashEmbedder
from app.graph.extractor import StubExtractor
from app.graph import local_store
from app.graph.local_store import LocalGraphStore
from app.graph.store import TextEpisode

LEXICON = {"凌雲飛行": "Company", "東海市交通運輸委員會": "Agency"}
ONTOLOGY = {"entity_types": [{"name": "Company", "description": ""}, {"name": "Agency", "description": ""}],
            "edge_types": []}


class ShiftedEmbedder(HashEmbedder):
    """Same dimension, different vectors: another model behind the same name."""

    def _embed(self, text):
        return super()._embed("shifted " + text[::-1])


@pytest.fixture
def warnings(monkeypatch):
    """The "mirofish" logger does not propagate, so caplog cannot see it."""

    seen = []
    monkeypatch.setattr(local_store.logger, "warning", lambda msg, *args: seen.append(msg % args))
    return seen


def _store(path, embedder):
    return LocalGraphStore(str(path), embedder=embedder, extractor=StubExtractor(LEXICON))


def _seed(path, embedder):
    store = _store(path, embedder)
    store.create_graph("g", graph_id="g1")
    store.set_ontology("g1", ONTOLOGY)
    store.add_text_episodes("g1", [TextEpisode("東海市交通運輸委員會今天批准凌雲飛行開始試營運。")], durable=True)
    return store


def test_vectors_are_used_by_the_embedder_that_wrote_them(tmp_path):
    store = _seed(tmp_path, HashEmbedder())
    assert store.vectors_usable("g1")
    assert store.search("g1", "凌雲飛行", scope="nodes", limit=5).nodes
    store.close()


def test_another_embedder_falls_back_to_keywords_until_reembedded(tmp_path, warnings):
    _seed(tmp_path, HashEmbedder()).close()
    store = _store(tmp_path, ShiftedEmbedder())
    assert not store.vectors_usable("g1")
    assert store.search("g1", "凌雲飛行", scope="nodes", limit=5).nodes  # BM25 still answers
    assert len(warnings) == 1 and "reembed_graphs.py" in warnings[0]  # once per graph

    counts = store.reembed("g1")
    assert counts["nodes"] == 2 and counts["edges"] >= 0
    assert store.vectors_usable("g1")
    store.close()


def test_a_graph_without_a_fingerprint_keeps_its_vectors(tmp_path, warnings):
    store = _seed(tmp_path, HashEmbedder())
    with store._graph("g1").use() as conn:
        conn.execute("DELETE FROM meta WHERE key = 'embedding_probe'")
    assert store.vectors_usable("g1")
    assert "fingerprint" in warnings[0]
    store.close()


def test_reembed_handles_a_new_dimension(tmp_path):
    _seed(tmp_path, HashEmbedder(dim=256)).close()
    store = _store(tmp_path, HashEmbedder(dim=128))
    store.reembed("g1")
    assert store._graph("g1").current_dim() == 128
    assert store.vectors_usable("g1")
    assert store.search("g1", "凌雲飛行", scope="nodes", limit=5).nodes
    store.close()


def test_tokenizer_check_warns_on_unknown_tokens(monkeypatch):
    from app.graph import embedding as model_health

    class Response:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

    broken = {"tokens": [{"id": 3, "piece": "<unk>"}, {"id": 70, "piece": " the"}]}
    monkeypatch.setattr(model_health.httpx, "post", lambda url, **kw: Response(200, broken))
    warning = model_health.check_embedding_tokenizer("http://127.0.0.1:8001/v1")
    assert warning and "convert_e5_gguf.py" in warning

    good = {"tokens": [{"id": 1, "piece": " Taxi"}, {"id": 2, "piece": " driver"}]}
    monkeypatch.setattr(model_health.httpx, "post", lambda url, **kw: Response(200, good))
    assert model_health.check_embedding_tokenizer("http://127.0.0.1:8001/v1") is None

    monkeypatch.setattr(model_health.httpx, "post", lambda url, **kw: Response(404, {}))
    assert model_health.check_embedding_tokenizer("http://127.0.0.1:8001/v1") is None  # not llama.cpp (TEI...)

    def refused(url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(model_health.httpx, "post", refused)
    assert model_health.check_embedding_tokenizer("http://127.0.0.1:8001/v1") is None
    assert model_health.check_embedding_tokenizer("https://api.example.com/v1") is None
