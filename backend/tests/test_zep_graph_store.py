from graph_store_contract import *  # noqa: F401,F403

import time

import pytest

from app.graph.store import GraphEdge, SearchResult, TextEpisode
from app.graph.zep_store import BatchSubmission, ZepGraphStore
from fake_zep_client import FakeZepClient


@pytest.fixture
def graph_store():
    return ZepGraphStore(FakeZepClient())


def test_search_ranking_maps_to_zep_reranker():
    client = FakeZepClient()
    store = ZepGraphStore(client)

    relevance = store.search("g1", "q", "edges", 10, ranking="relevance")
    fusion = store.search("g1", "q", "edges", 10, ranking="fusion")

    assert [call["reranker"] for call in client.graph.search_calls] == [
        "cross_encoder",
        "rrf",
    ]
    for result in (relevance, fusion):
        assert isinstance(result, SearchResult)
        assert result.edges
        assert all(isinstance(edge, GraphEdge) for edge in result.edges)


def test_durable_flag_selects_zep_write_path():
    client = FakeZepClient()
    store = ZepGraphStore(client)
    episode = TextEpisode("a", "2026-01-01T00:00:00Z", "s", {"k": "v"})

    immediate = store.add_text_episodes("g1", [episode], durable=False)

    assert client.graph.add_calls == [{
        "graph_id": "g1",
        "type": "text",
        "data": "a",
        "created_at": "2026-01-01T00:00:00Z",
        "source_description": "s",
        "metadata": {"k": "v"},
    }]
    assert immediate.batch_id is None

    durable = store.add_text_episodes("g1", [TextEpisode("b")], durable=True)

    assert client.batch.create_calls
    assert durable.batch_id is not None
    assert durable.operation_id is not None


def test_expired_batch_deadline_raises_without_polling():
    client = FakeZepClient()
    store = ZepGraphStore(client)
    submitted = store.add_text_episodes("g1", [TextEpisode("b")], durable=True)
    gets = {"count": 0}
    original_get = client.batch.get

    def counting_get(batch_id):
        gets["count"] += 1
        return original_get(batch_id)

    client.batch.get = counting_get

    with pytest.raises(TimeoutError, match="batch-"):
        store.wait_until_processed(submitted, deadline=time.time() - 1)
    assert gets["count"] == 0


def test_batch_wait_returns_processed_episode_ids():
    client = FakeZepClient()
    store = ZepGraphStore(client)
    submitted = store.add_text_episodes("g1", [TextEpisode("b")], durable=True)
    resumed = BatchSubmission(
        submitted.batch_id,
        submitted.operation_id,
        [],
        submitted.item_count,
    )

    processed = store.wait_until_processed(resumed, deadline=time.time() + 5)

    assert processed == submitted.episode_ids
    assert processed
