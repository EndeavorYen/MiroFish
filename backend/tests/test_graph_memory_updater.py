import pytest
from datetime import datetime


@pytest.fixture
def store(tmp_db_path):
    from app.services.networkx_graph_store import NetworkXGraphStore
    store = NetworkXGraphStore(db_path=tmp_db_path)
    store.create_graph("sim-1", "Sim", "")
    store.add_entity("sim-1", "agent_1", "agent", "User 1", {})
    store.add_entity("sim-1", "agent_2", "agent", "User 2", {})
    return store


@pytest.fixture
def updater(store):
    from app.services.graph_memory_updater import GraphMemoryUpdater
    return GraphMemoryUpdater(graph_id="sim-1", store=store)


def test_create_post_adds_edge_and_text(updater, store):
    updater.add_activity({
        "platform": "twitter", "agent_id": "agent_1", "agent_name": "User1",
        "action_type": "CREATE_POST",
        "action_args": {"content": "Hello world!", "post_id": "post_1"},
        "round_num": 1, "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert any(e.name == "POSTED" for e in edges)
    texts = store.search_text("sim-1", "Hello world")
    assert len(texts) >= 1


def test_like_post_adds_edge(updater, store):
    store.add_entity("sim-1", "post_1", "post", "A post", {})
    updater.add_activity({
        "platform": "twitter", "agent_id": "agent_1", "agent_name": "User1",
        "action_type": "LIKE_POST",
        "action_args": {"post_id": "post_1", "author": "User2", "content": "Nice"},
        "round_num": 1, "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert any(e.name == "LIKED" for e in edges)


def test_follow_adds_agent_edge(updater, store):
    updater.add_activity({
        "platform": "twitter", "agent_id": "agent_1", "agent_name": "User1",
        "action_type": "FOLLOW",
        "action_args": {"target_user_name": "agent_2"},
        "round_num": 1, "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert any(e.name == "FOLLOWS" for e in edges)


def test_do_nothing_is_skipped(updater, store):
    updater.add_activity({
        "platform": "twitter", "agent_id": "agent_1", "agent_name": "User1",
        "action_type": "DO_NOTHING", "action_args": {},
        "round_num": 1, "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert len(edges) == 0


def test_get_stats(updater):
    updater.add_activity({
        "platform": "twitter", "agent_id": "agent_1", "agent_name": "User1",
        "action_type": "CREATE_POST",
        "action_args": {"content": "Test", "post_id": "p1"},
        "round_num": 1, "timestamp": datetime.now().isoformat(),
    })
    stats = updater.get_stats()
    assert stats["total_processed"] == 1


def _find_uuid(store, name):
    entities = store.list_entities("sim-1")
    return next(e.uuid for e in entities if e.name == name)
