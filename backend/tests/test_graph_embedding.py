import math

from app.config import Config
from app.graph.embedding import HashEmbedder, HttpEmbedder, make_embedder


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dim=64)
    a = embedder.embed_query("凌雲飛行")
    assert a == embedder.embed_documents(["凌雲飛行"])[0]
    assert math.isclose(sum(v * v for v in a), 1.0, rel_tol=1e-9)
    near = embedder.embed_query("凌雲飛行公司")
    far = embedder.embed_query("banana bread")
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731
    assert dot(a, near) > dot(a, far)


def test_http_embedder_prefixes_truncates_and_batches(monkeypatch):
    embedder = HttpEmbedder(
        base_url="http://embed/v1",
        model="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
        batch_size=2,
        max_input_chars=5,
    )
    calls = []

    def fake_post(inputs):
        calls.append(inputs)
        return [[3.0, 4.0] for _ in inputs]

    monkeypatch.setattr(embedder, "_post", fake_post)
    vectors = embedder.embed_documents(["abcdefgh", "b", "c"])
    assert calls == [["passage: abcde", "passage: b"], ["passage: c"]]
    assert len(vectors) == 3
    embedder.embed_query("0123456789")
    assert calls[-1] == ["query: 01234"]


def test_make_embedder_uses_e5_prefixes_by_default(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_EMBEDDER", "http")
    monkeypatch.setattr(Config, "EMBED_MODEL_NAME", "intfloat/multilingual-e5-small")
    monkeypatch.setattr(Config, "EMBED_QUERY_PREFIX", None)
    monkeypatch.setattr(Config, "EMBED_PASSAGE_PREFIX", None)
    embedder = make_embedder()
    assert (embedder.query_prefix, embedder.passage_prefix) == ("query: ", "passage: ")
    monkeypatch.setattr(Config, "GRAPH_EMBEDDER", "hash")
    assert isinstance(make_embedder(), HashEmbedder)
