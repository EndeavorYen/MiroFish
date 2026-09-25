"""Callers must work against any GraphStore, not only ZepGraphStore."""

from __future__ import annotations

import time

import pytest

from app.config import Config
from app.graph.store import GraphNode, GraphNotFoundError, SearchResult
from app.services import zep_graph_memory_updater as updater_module
from app.services.graph_builder import GraphBuilderService
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import ZepEntityReader
from app.services.zep_graph_memory_updater import AgentActivity, ZepGraphMemoryUpdater
from app.services.zep_tools import ZepToolsService


class _Handle:
    def __init__(self, episode_ids):
        self._episode_ids = list(episode_ids)

    @property
    def episode_ids(self):
        return list(self._episode_ids)


class ProtocolOnlyStore:
    """In-memory store exposing only GraphStore Protocol methods."""

    def __init__(self):
        self.calls = []
        self._next_id = 0

    def create_graph(self, name, *, graph_id=None, graph_id_callback=None):
        graph_id = graph_id or "g-local"
        if graph_id_callback:
            graph_id_callback(graph_id)
        self.calls.append(("create_graph", graph_id))
        return graph_id

    def delete_graph(self, graph_id):
        self.calls.append(("delete_graph", graph_id))

    def set_ontology(self, graph_id, ontology):
        self.calls.append(("set_ontology", graph_id))

    def add_text_episodes(
        self,
        graph_id,
        episodes,
        *,
        durable,
        batch_size=350,
        on_submitted=None,
        on_progress=None,
    ):
        ids = []
        for _episode in episodes:
            self._next_id += 1
            ids.append(f"ep-{self._next_id}")
        self.calls.append(("add_text_episodes", durable, len(episodes)))
        return _Handle(ids)

    def wait_until_processed(self, handle, *, deadline=None, on_progress=None):
        self.calls.append(("wait_until_processed", handle.episode_ids))
        return handle.episode_ids

    def list_nodes(self, graph_id):
        return [GraphNode("n1", "Alice", ["Entity", "Person"], "", {})]

    def list_edges(self, graph_id):
        return []

    def get_node(self, node_uuid):
        raise GraphNotFoundError(node_uuid)

    def get_node_edges(self, node_uuid):
        return []

    def search(self, graph_id, query, scope, limit, *, ranking="relevance"):
        return SearchResult(edges=[], nodes=[])


class _Tasks:
    def __init__(self):
        self.completed = None
        self.failed = None

    def update_task(self, *_args, **_kwargs):
        pass

    def complete_task(self, task_id, result):
        self.completed = result

    def fail_task(self, task_id, error):
        self.failed = error


def test_graph_build_worker_uses_only_protocol_methods():
    store = ProtocolOnlyStore()
    builder = object.__new__(GraphBuilderService)
    builder.store = store
    builder.task_manager = _Tasks()

    builder._build_graph_worker(
        "task-1",
        "Alice met Bob in Paris. " * 10,
        {"entity_types": [], "edge_types": []},
        "graph",
        500,
        50,
        350,
    )

    assert builder.task_manager.failed is None
    assert builder.task_manager.completed["graph_info"]["node_count"] == 1
    names = [call[0] for call in store.calls]
    assert names == [
        "create_graph",
        "set_ontology",
        "add_text_episodes",
        "wait_until_processed",
    ]
    assert store.calls[2][1] is True


def test_memory_updater_uses_only_protocol_methods(monkeypatch):
    store = ProtocolOnlyStore()
    monkeypatch.setattr(updater_module, "get_graph_store", lambda **_kwargs: store)
    updater = ZepGraphMemoryUpdater("graph-1", api_key="k", simulation_id="sim-1")
    activity = AgentActivity(
        platform="twitter",
        agent_id=1,
        agent_name="Agent 1",
        action_type="CREATE_POST",
        action_args={"content": "hello"},
        round_num=1,
        timestamp="2026-07-22T12:00:00+08:00",
    )

    updater._send_batch_activities([activity], "twitter")
    updater._wait_for_pending_episodes(deadline=time.time() + 5)

    assert store.calls[0] == ("add_text_episodes", False, 1)
    assert store.calls[1] == ("wait_until_processed", ["ep-1"])
    assert updater.get_stats()["pending_episode_count"] == 0


_SERVICES = [
    ("GraphBuilderService", lambda: GraphBuilderService()),
    ("ZepEntityReader", lambda: ZepEntityReader()),
    ("ZepToolsService", lambda: ZepToolsService()),
    ("ZepGraphMemoryUpdater", lambda: ZepGraphMemoryUpdater("graph-1")),
]


@pytest.mark.parametrize("name,build", _SERVICES, ids=[name for name, _ in _SERVICES])
def test_local_backend_fails_clearly_without_zep_key(monkeypatch, name, build):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "local")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)

    with pytest.raises(NotImplementedError, match="GRAPH_BACKEND=local"):
        build()


def test_profile_generator_fails_clearly_for_local_backend(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "local")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)

    with pytest.raises(NotImplementedError, match="GRAPH_BACKEND=local"):
        OasisProfileGenerator(api_key="llm-key", base_url="http://127.0.0.1:9")


@pytest.mark.parametrize("name,build", _SERVICES, ids=[name for name, _ in _SERVICES])
def test_zep_backend_still_requires_zep_key(monkeypatch, name, build):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)

    with pytest.raises(ValueError, match="ZEP_API_KEY 未配置"):
        build()
