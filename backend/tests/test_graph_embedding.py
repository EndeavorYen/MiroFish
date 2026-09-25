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


def test_http_embedder_retries_dropped_connections_but_not_read_timeouts(monkeypatch):
    import httpx

    import app.graph.embedding as embedding

    monkeypatch.setattr(embedding.time, "sleep", lambda s: None)
    embedder = HttpEmbedder(base_url="http://embed/v1", model="m")
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    embedder._client = httpx.Client(transport=httpx.MockTransport(flaky))
    assert embedder.embed_query("x") == [0.6, 0.8]
    assert calls["n"] == 3

    def stalled(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("stalled", request=request)

    calls["n"] = 0
    embedder._client = httpx.Client(transport=httpx.MockTransport(stalled))
    try:
        embedder.embed_query("x")
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError("expected ReadTimeout")
    assert calls["n"] == 1
