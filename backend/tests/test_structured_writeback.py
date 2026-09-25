"""Structured simulation writeback on the local graph (#8)."""

import json

import pytest

from app.config import Config
from app.graph.embedding import HashEmbedder
from app.graph.extractor import StubExtractor
from app.graph.local_store import LocalGraphStore
from app.graph.store import StructuredFactStore, TextEpisode
from app.services.simulation_facts import (
    REDDIT_ACTIONS,
    TWITTER_ACTIONS,
    activity_to_fact,
    fact_key,
)
from app.services.zep_graph_memory_updater import AgentActivity, ZepGraphMemoryUpdater

ARGS = {
    "CREATE_POST": {"content": "凌雲飛行的試飛很成功", "post_id": 11},
    "LIKE_POST": {"post_id": 5, "post_content": "票價太貴", "post_author_name": "Bob"},
    "DISLIKE_POST": {"post_id": 5, "post_content": "票價太貴", "post_author_name": "Bob"},
    "REPOST": {
        "new_post_id": 12,
        "original_post_id": 5,
        "original_content": "票價太貴",
        "original_author_name": "Bob",
    },
    "QUOTE_POST": {
        "quoted_id": 5,
        "new_post_id": 13,
        "original_content": "票價太貴",
        "original_author_name": "Bob",
        "quote_content": "同意，凌雲飛行應該降價",
    },
    "FOLLOW": {"follow_id": 3, "target_user_name": "Bob"},
    "MUTE": {"user_id": 4, "target_user_name": "Carol"},
    "CREATE_COMMENT": {
        "content": "我也想搭",
        "comment_id": 21,
        "post_id": 5,
        "post_content": "票價太貴",
        "post_author_name": "Bob",
    },
    "LIKE_COMMENT": {"comment_id": 21, "comment_content": "我也想搭", "comment_author_name": "Carol"},
    "DISLIKE_COMMENT": {"comment_id": 21, "comment_content": "我也想搭", "comment_author_name": "Carol"},
    "SEARCH_POSTS": {"query": "空中計程車"},
    "SEARCH_USER": {"query": "Bob"},
    "TREND": {},
    "REFRESH": {},
    "DO_NOTHING": {},
}
EXPECTED_KIND = {
    "CREATE_POST": "Post",
    "LIKE_POST": "Post",
    "DISLIKE_POST": "Post",
    "REPOST": "Post",
    "QUOTE_POST": "Post",
    "FOLLOW": "SimAgent",
    "MUTE": "SimAgent",
    "CREATE_COMMENT": "Comment",
    "LIKE_COMMENT": "Comment",
    "DISLIKE_COMMENT": "Comment",
    "SEARCH_POSTS": "SearchQuery",
    "SEARCH_USER": "SearchQuery",
    "TREND": "Feed",
    "REFRESH": "Feed",
}


def _activity(action, platform="twitter", agent_id=1, name="Alice", round_num=3, seq=0):
    return AgentActivity(
        platform=platform,
        agent_id=agent_id,
        agent_name=name,
        action_type=action,
        action_args=dict(ARGS[action]),
        round_num=round_num,
        timestamp="2026-09-25T10:00:00",
        action_seq=seq,
    )


def _store(tmp_path, lexicon=None):
    return LocalGraphStore(
        str(tmp_path), embedder=HashEmbedder(), extractor=StubExtractor(lexicon or {})
    )


@pytest.mark.parametrize(
    "platform,action",
    [("twitter", a) for a in TWITTER_ACTIONS] + [("reddit", a) for a in REDDIT_ACTIONS],
)
def test_every_action_type_maps_to_a_fact(platform, action):
    fact = activity_to_fact(_activity(action, platform=platform), 0, ["凌雲飛行"], simulation_id="sim_1")
    if action == "DO_NOTHING":
        assert fact is None
        return
    assert fact.key == fact_key(platform, 3, 1, 0, "sim_1") == f"sim_1:{platform}:3:1:0"
    assert fact.relation == action
    assert fact.source.key == "agent:sim_1:1" and fact.source.name == "Alice"
    assert fact.target.label == EXPECTED_KIND[action]
    assert fact.attributes == {
        "platform": platform,
        "round": 3,
        "simulated_time": "2026-09-25T10:00:00",
        "agent_id": 1,
        "action_seq": 0,
    }
    assert "Alice" in fact.fact
    keys = {fact.source.key, fact.target.key, *(n.key for n in fact.extra_nodes)}
    for source_key, _, target_key in fact.extra_edges:
        assert source_key in keys and target_key in keys


def test_posts_carry_text_and_link_mentions():
    fact = activity_to_fact(_activity("QUOTE_POST"), 0, ["凌雲飛行", "林建東"], simulation_id="s")
    assert fact.target.key == "post:s:twitter:5" and fact.target.summary == "票價太貴"
    quote = next(n for n in fact.extra_nodes if n.key == "post:s:twitter:13")
    assert quote.summary == "同意，凌雲飛行應該降價"
    assert ("post:s:twitter:13", "QUOTES", "post:s:twitter:5") in fact.extra_edges
    assert fact.mentions == ("凌雲飛行",)


def test_replaying_actions_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.create_graph("g", graph_id="g1")
    facts = [
        activity_to_fact(_activity(action, platform="reddit", seq=i), i)
        for i, action in enumerate(REDDIT_ACTIONS)
    ]
    facts = [f for f in facts if f]
    store.add_structured_facts("g1", facts)
    counts = (len(store.list_nodes("g1")), len(store.list_edges("g1")))
    store.add_structured_facts("g1", facts)
    store.add_structured_facts("g1", list(reversed(facts)))
    assert (len(store.list_nodes("g1")), len(store.list_edges("g1"))) == counts
    store.close()


def test_agent_attaches_to_existing_entity_and_posts_stay_out_of_entity_filter(tmp_path):
    store = _store(tmp_path, {"Alice": "Person", "凌雲飛行": "Company"})
    store.create_graph("g", graph_id="g1")
    store.set_ontology(
        "g1",
        {"entity_types": [{"name": "Person"}, {"name": "Company"}], "edge_types": []},
    )
    store.add_text_episodes("g1", [TextEpisode("Alice 看好凌雲飛行。")], durable=True)
    alice = next(n for n in store.list_nodes("g1") if n.name == "Alice")

    store.add_structured_facts("g1", [activity_to_fact(_activity("CREATE_POST"), 0, ["凌雲飛行"])])
    nodes = store.list_nodes("g1")
    assert [n.uuid for n in nodes if n.name == "Alice"] == [alice.uuid]
    post = next(n for n in nodes if n.attributes.get("kind") == "Post")
    assert post.labels == ["Node"]
    edges = store.list_edges("g1")
    create = next(e for e in edges if e.name == "CREATE_POST")
    assert create.source_node_uuid == alice.uuid and create.target_node_uuid == post.uuid
    company = next(n for n in nodes if n.name == "凌雲飛行")
    assert any(
        e.name == "MENTIONS" and e.source_node_uuid == post.uuid and e.target_node_uuid == company.uuid
        for e in edges
    )
    store.close()


class _NoModelExtractor:
    def extract(self, *args, **kwargs):
        raise AssertionError("structured writeback must not run text extraction")


@pytest.fixture
def local_updater(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "local")
    monkeypatch.setattr(Config, "GRAPH_DATA_DIR", str(tmp_path / "graphs"))
    monkeypatch.setattr(Config, "GRAPH_EMBEDDER", "hash")
    monkeypatch.setattr("app.graph.local_store.make_extractor", lambda: _NoModelExtractor())
    updater = ZepGraphMemoryUpdater("sim_graph", simulation_id="sim_1")
    updater.store.create_graph("g", graph_id="sim_graph")
    yield updater
    updater.store.close()


def test_updater_writes_structured_facts_without_model_calls(local_updater, tmp_path):
    from app.utils.llm_usage import usage_stage

    assert isinstance(local_updater.store, StructuredFactStore)
    activities = [
        _activity("CREATE_POST", agent_id=2, name="Bob", seq=0),
        _activity("REPOST", agent_id=1, name="Alice", seq=0),
        _activity("LIKE_POST", agent_id=1, name="Alice", seq=1),
    ]
    metrics = tmp_path / "metrics"
    with usage_stage("simulation", str(metrics)):
        processed = local_updater._send_batch_activities(activities, "twitter")
        local_updater._wait_for_pending_episodes()
    assert processed == 3
    assert not (metrics / "llm_usage.jsonl").exists()
    stats = local_updater.get_stats()
    assert stats["failed_count"] == 0 and stats["pending_episode_count"] == 0

    edges = local_updater.store.search("sim_graph", "Alice 转发了什么", "edges", 5).edges
    assert edges and edges[0].name == "REPOST"
    assert "Bob" in edges[0].fact and "票價太貴" in edges[0].fact


def test_updater_assigns_action_seq_in_log_order(local_updater):
    local_updater._running = True
    for action in ("LIKE_POST", "REPOST", "LIKE_POST"):
        local_updater.add_activity(_activity(action))
    local_updater.add_activity(_activity("LIKE_POST", round_num=4))
    local_updater._running = False
    queued = [local_updater._activity_queue.get_nowait() for _ in range(4)]
    assert [a.action_seq for a in queued] == [0, 1, 2, 0]


def test_zep_store_keeps_text_episode_path():
    from app.graph.zep_store import ZepGraphStore
    from fake_zep_client import FakeZepClient

    assert not isinstance(ZepGraphStore(FakeZepClient()), StructuredFactStore)


def test_a_second_simulation_on_the_same_graph_is_not_dropped(tmp_path):
    store = _store(tmp_path)
    store.create_graph("g", graph_id="g1")
    first = activity_to_fact(_activity("CREATE_POST"), 0, simulation_id="sim_a")
    second = activity_to_fact(_activity("CREATE_POST"), 0, simulation_id="sim_b")
    store.add_structured_facts("g1", [first])
    store.add_structured_facts("g1", [second])
    posts = [n for n in store.list_nodes("g1") if n.attributes.get("kind") == "Post"]
    assert len(posts) == 2
    assert len([e for e in store.list_edges("g1") if e.name == "CREATE_POST"]) == 2
    store.close()


def test_follow_and_mute_target_agents_by_id():
    follow = _activity("FOLLOW")
    follow.action_args["followee_id"] = 7
    mute = _activity("MUTE")
    mute.action_args = {"mutee_id": 7}
    assert activity_to_fact(follow, 0, simulation_id="s").target.key == "agent:s:7"
    assert activity_to_fact(mute, 0, simulation_id="s").target.key == "agent:s:7"
    unknown = _activity("MUTE")
    unknown.action_args = {}
    assert activity_to_fact(unknown, 0, simulation_id="s").target.key.startswith("user:s:unknown:")


def test_replay_restores_vectors_lost_to_an_embedding_failure(tmp_path):
    class Flaky(HashEmbedder):
        fail = True

        def embed_documents(self, texts):
            if Flaky.fail:
                Flaky.fail = False
                raise ConnectionError("down")
            return super().embed_documents(texts)

    store = LocalGraphStore(str(tmp_path), embedder=Flaky(), extractor=StubExtractor({}))
    store.create_graph("g", graph_id="g1")
    fact = activity_to_fact(_activity("CREATE_POST"), 0, simulation_id="s")
    with pytest.raises(ConnectionError):
        store.add_structured_facts("g1", [fact])
    store.add_structured_facts("g1", [fact])
    graph = store._graph("g1")
    edges = graph.conn.execute("SELECT rowid FROM edges").fetchall()
    for (rowid,) in edges:
        assert graph.conn.execute("SELECT 1 FROM vec_edges WHERE rowid = ?", (rowid,)).fetchone()
    store.close()


def test_mentions_resolve_entity_aliases(tmp_path):
    from app.graph.extractor import ExtractedEntity, Extraction

    class Scripted:
        def extract(self, text, ontology, known_entities=None):
            return Extraction([ExtractedEntity("凌雲飛行智能公司", "Company", "x", {"aliases": ["凌雲科技"]})])

    store = LocalGraphStore(str(tmp_path), embedder=HashEmbedder(), extractor=Scripted())
    store.create_graph("g", graph_id="g1")
    store.add_text_episodes("g1", [TextEpisode("x")], durable=True)
    post = _activity("CREATE_POST")
    post.action_args = {"content": "凌雲科技又上新聞", "post_id": 9}
    store.add_structured_facts("g1", [activity_to_fact(post, 0, ["凌雲科技"], simulation_id="s")])
    company = next(n for n in store.list_nodes("g1") if n.name == "凌雲飛行智能公司")
    assert any(e.name == "MENTIONS" and e.target_node_uuid == company.uuid for e in store.list_edges("g1"))
    store.close()


def test_unknown_targets_and_same_named_agents_stay_separate(tmp_path):
    store = _store(tmp_path)
    store.create_graph("g", graph_id="g1")
    facts = []
    for seq in range(2):
        mute = _activity("MUTE", seq=seq)
        mute.action_args = {}
        facts.append(activity_to_fact(mute, seq, simulation_id="s"))
    store.add_structured_facts("g1", facts)
    unknown = [n for n in store.list_nodes("g1") if n.name.startswith("未知用户")]
    assert len(unknown) == 2

    other = activity_to_fact(_activity("CREATE_POST"), 0, simulation_id="t")
    store.add_structured_facts("g1", [other])
    alices = [n for n in store.list_nodes("g1") if n.name == "Alice"]
    assert len(alices) == 2  # one per simulation, no ontology entity to attach to
    store.close()
