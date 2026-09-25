from graph_store_contract import *  # noqa: F401,F403

import os
import time

import pytest

from app.graph.embedding import HashEmbedder
from app.graph.extractor import StubExtractor
from app.graph.local_store import LocalGraphStore, normalize_name
from app.graph.store import GraphNotFoundError, TextEpisode
from app.graph.text_index import match_query, tokenize

ONTOLOGY = {
    "entity_types": [
        {"name": "Company", "description": "a company"},
        {"name": "Agency", "description": "a government agency"},
        {"name": "Person", "description": "a person"},
    ],
    "edge_types": [
        {
            "name": "REGULATES",
            "source_targets": [{"source": "Agency", "target": "Company"}],
        }
    ],
}

LEXICON = {
    "凌雲飛行": "Company",
    "東海市交通運輸委員會": "Agency",
    "陳志遠": "Person",
    "Alice": "Person",
    "Bob": "Person",
    "計程車工會": "Union",  # not in the ontology -> dropped
}


def _store(path, lexicon=None):
    return LocalGraphStore(
        str(path),
        embedder=HashEmbedder(),
        extractor=StubExtractor(LEXICON if lexicon is None else lexicon),
    )


@pytest.fixture
def graph_store(tmp_path):
    store = _store(tmp_path)
    yield store
    store.close()


@pytest.fixture
def seeded(tmp_path):
    store = _store(tmp_path)
    store.create_graph("g", graph_id="g1")
    store.set_ontology("g1", ONTOLOGY)
    store.add_text_episodes(
        "g1",
        [
            TextEpisode("東海市交通運輸委員會今天批准凌雲飛行開始試營運。計程車工會表示反對。"),
            TextEpisode("凌雲飛行執行長陳志遠說，首批六架航空器將在下月載客。"),
        ],
        durable=True,
    )
    yield store
    store.close()


def test_tokenize_splits_cjk_into_bigrams_and_lowercases_latin():
    assert tokenize("凌雲飛行 eVTOL 2026") == ["凌雲", "雲飛", "飛行", "evtol", "2026"]
    assert tokenize("市") == ["市"]
    assert match_query("") is None
    assert match_query('he said "hi"') == '"he" OR "said" OR "hi"'


def test_labels_follow_zep_convention_and_ontology_filter(seeded):
    nodes = {node.name: node for node in seeded.list_nodes("g1")}
    assert set(nodes) == {"凌雲飛行", "東海市交通運輸委員會", "陳志遠"}
    assert nodes["凌雲飛行"].labels == ["Entity", "Company"]
    assert "Union" not in {label for node in nodes.values() for label in node.labels}


def test_nodes_merge_by_normalized_name_across_episodes(seeded):
    company = [n for n in seeded.list_nodes("g1") if n.name == "凌雲飛行"]
    assert len(company) == 1
    assert "批准" in company[0].summary and "執行長" in company[0].summary
    assert normalize_name(" Ｌing Yun ") == normalize_name("lingyun")


def test_relations_use_ontology_edge_type_and_link_episodes(seeded):
    edges = seeded.list_edges("g1")
    regulates = [e for e in edges if e.name == "REGULATES"]
    assert len(regulates) == 1
    assert "批准" in regulates[0].fact
    assert len(regulates[0].episodes) == 1
    agency = next(n for n in seeded.list_nodes("g1") if n.name == "東海市交通運輸委員會")
    assert regulates[0].source_node_uuid == agency.uuid
    assert seeded.get_node_edges(agency.uuid)[0].uuid == regulates[0].uuid


def test_replaying_the_same_episode_does_not_duplicate_edges(seeded):
    before = (len(seeded.list_nodes("g1")), len(seeded.list_edges("g1")))
    seeded.add_text_episodes(
        "g1",
        [TextEpisode("東海市交通運輸委員會今天批准凌雲飛行開始試營運。計程車工會表示反對。")],
        durable=True,
    )
    after = (len(seeded.list_nodes("g1")), len(seeded.list_edges("g1")))
    assert before == after
    regulates = next(e for e in seeded.list_edges("g1") if e.name == "REGULATES")
    assert len(regulates.episodes) == 2


def test_search_returns_edges_with_fact_and_nodes_with_summary(seeded):
    edges = seeded.search("g1", "誰批准了試營運", "edges", 5).edges
    assert edges and "批准" in edges[0].fact
    nodes = seeded.search("g1", "執行長", "nodes", 5, ranking="fusion").nodes
    assert nodes[0].name in {"陳志遠", "凌雲飛行"}
    assert all(node.summary for node in nodes)
    assert seeded.search("g1", "   ", "nodes", 5).nodes == []
    with pytest.raises(ValueError):
        seeded.search("g1", "x", "episodes", 5)


def test_get_node_and_missing_lookups(seeded):
    node = seeded.list_nodes("g1")[0]
    assert seeded.get_node(node.uuid) == node
    with pytest.raises(GraphNotFoundError):
        seeded.get_node("nope")
    with pytest.raises(GraphNotFoundError):
        seeded.list_nodes("missing-graph")
    with pytest.raises(ValueError):
        seeded.create_graph("bad", graph_id="../escape")


def test_data_persists_across_store_instances(tmp_path):
    store = _store(tmp_path)
    store.create_graph("g", graph_id="g1")
    store.set_ontology("g1", ONTOLOGY)
    store.add_text_episodes("g1", [TextEpisode("Alice met Bob.")], durable=False)
    store.close()

    reopened = _store(tmp_path)
    try:
        assert {n.name for n in reopened.list_nodes("g1")} == {"Alice", "Bob"}
        assert reopened.get_ontology("g1") == ONTOLOGY
        assert reopened.search("g1", "Alice", "nodes", 3).nodes[0].name == "Alice"
    finally:
        reopened.close()


def test_delete_graph_removes_files_and_index(tmp_path):
    store = _store(tmp_path)
    store.create_graph("g", graph_id="g1")
    store.add_text_episodes("g1", [TextEpisode("Alice met Bob.")], durable=False)
    node = store.list_nodes("g1")[0]
    store.delete_graph("g1")
    assert not os.path.exists(tmp_path / "g1.sqlite")
    with pytest.raises(GraphNotFoundError):
        store.get_node(node.uuid)
    store.close()


def test_wait_rejects_unknown_episode_ids(graph_store):
    from app.graph.store import EpisodeHandle

    with pytest.raises(RuntimeError):
        graph_store.wait_until_processed(EpisodeHandle(["unknown"]))
    with pytest.raises(TimeoutError):
        graph_store.wait_until_processed(EpisodeHandle(["unknown"]), deadline=time.time() - 1)


def test_progress_callback_reports_each_episode(graph_store):
    graph_store.create_graph("g", graph_id="g1")
    seen = []
    graph_store.add_text_episodes(
        "g1",
        [TextEpisode("Alice."), TextEpisode("Bob.")],
        durable=True,
        on_progress=lambda message, ratio: seen.append(ratio),
    )
    assert seen == [0.5, 1.0]
