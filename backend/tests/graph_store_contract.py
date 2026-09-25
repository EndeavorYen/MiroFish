"""GraphStore contract tests.

Imported by an implementation suite that provides the ``graph_store`` fixture.
This module is not named ``test_*.py``, so pytest does not collect it directly.
"""

from __future__ import annotations

import time

import pytest

from app.graph.store import EpisodeHandle, GraphNotFoundError, TextEpisode


def test_graph_lifecycle_roundtrip(graph_store):
    assert graph_store.create_graph("g", graph_id="g1") == "g1"
    assert graph_store.list_nodes("g1") == []
    assert graph_store.list_edges("g1") == []
    with pytest.raises(GraphNotFoundError):
        graph_store.get_node("missing")
    graph_store.delete_graph("g1")
    with pytest.raises(GraphNotFoundError):
        graph_store.delete_graph("g1")


def test_episode_ingestion_is_waitable(graph_store):
    graph_store.create_graph("g", graph_id="g1")
    handle = graph_store.add_text_episodes(
        "g1",
        [TextEpisode("Alice met Bob")],
        durable=False,
    )
    assert len(handle.episode_ids) == 1
    processed = graph_store.wait_until_processed(handle, deadline=time.time() + 5)
    assert processed == handle.episode_ids


def test_wait_accepts_known_episode_ids(graph_store):
    graph_store.create_graph("g", graph_id="g1")
    handle = graph_store.add_text_episodes(
        "g1",
        [TextEpisode("Alice met Bob")],
        durable=False,
    )
    known = EpisodeHandle(handle.episode_ids)
    assert graph_store.wait_until_processed(known, deadline=time.time() + 5) == (
        handle.episode_ids
    )
