"""Users confirm the roles a graph found before preparation (#64)."""

import pytest

from app.graph.embedding import HashEmbedder
from app.graph.extractor import StubExtractor
from app.graph.local_store import LocalGraphStore
from app.graph.store import TextEpisode

LEXICON = {"雲梯科技": "Company", "雲梯": "Company", "師林書瑤": "Person", "林書瑤": "Person", "數位前線": "Company"}
ONTOLOGY = {
    "entity_types": [{"name": "Company", "description": ""}, {"name": "Person", "description": ""}],
    "edge_types": [{"name": "WORKS_FOR", "description": "",
                    "source_targets": [{"source": "Person", "target": "Company"}]}],
}


@pytest.fixture
def store(tmp_path):
    store = LocalGraphStore(str(tmp_path), embedder=HashEmbedder(), extractor=StubExtractor(LEXICON))
    store.create_graph("g", graph_id="g1")
    store.set_ontology("g1", ONTOLOGY)
    store.add_text_episodes("g1", [TextEpisode("分析師林書瑤任職於雲梯科技。雲梯表示歡迎。數位前線報導此事。")], durable=True)
    yield store
    store.close()


def _roles(store):
    return {n.name: n for n in store.list_nodes("g1") if "Entity" in n.labels}


def _typed(store):
    return {name for name, n in _roles(store).items() if len(n.labels) > 1}


def test_rename_keeps_the_old_name_as_an_alias(store):
    node = _roles(store)["師林書瑤"]
    store.rename_node("g1", node.uuid, "林書瑤x")
    renamed = _roles(store)["林書瑤x"]
    assert renamed.uuid == node.uuid and "師林書瑤" in renamed.attributes["aliases"]
    assert store.search("g1", "林書瑤x", scope="nodes", limit=3).nodes[0].uuid == node.uuid
    # A later mention of the old name lands on the same node.
    store.add_text_episodes("g1", [TextEpisode("師林書瑤再度發言。")], durable=True)
    assert "師林書瑤" not in _roles(store)


def test_rename_onto_an_existing_name_is_refused(store):
    with pytest.raises(ValueError, match="merge"):
        store.rename_node("g1", _roles(store)["雲梯"].uuid, "雲梯科技")


def test_merge_moves_edges_and_aliases(store):
    roles = _roles(store)
    keep, drop = roles["雲梯科技"], roles["雲梯"]
    edges_before = len(store.list_edges("g1"))
    store.merge_nodes("g1", keep.uuid, drop.uuid)
    roles = _roles(store)
    assert "雲梯" not in roles and "雲梯" in roles["雲梯科技"].attributes["aliases"]
    assert all(drop.uuid not in (e.source_node_uuid, e.target_node_uuid) for e in store.list_edges("g1"))
    assert len(store.list_edges("g1")) <= edges_before
    store.add_text_episodes("g1", [TextEpisode("雲梯今天發表新產品。")], durable=True)
    assert "雲梯" not in _roles(store)  # resolves to 雲梯科技


def test_excluded_roles_are_not_prepared_and_stay_excluded(store):
    node = _roles(store)["數位前線"]
    store.set_node_role("g1", node.uuid, None)
    assert "數位前線" not in _typed(store)
    store.add_text_episodes("g1", [TextEpisode("數位前線刊出評論。")], durable=True)
    assert "數位前線" not in _typed(store)  # a later mention does not bring the type back
    store.set_node_role("g1", node.uuid, "Company")
    assert "數位前線" in _typed(store)


def test_add_a_role_the_extractor_missed(store):
    uuid = store.add_role_node("g1", "OpenAI", "Company", "宣布新的訂閱方案")
    assert _roles(store)["OpenAI"].uuid == uuid and "OpenAI" in _typed(store)
    with pytest.raises(ValueError):
        store.add_role_node("g1", "openai", "Company", "")  # same name key
    with pytest.raises(ValueError):
        store.set_node_role("g1", uuid, "Planet")  # not an ontology type


@pytest.fixture
def client(store, monkeypatch):
    from app import create_app
    from app.graph import store as store_module

    monkeypatch.setattr(store_module, "get_graph_store", lambda **kw: store)
    return create_app().test_client()


def test_the_api_edits_roles_and_preparation_sees_them(client, store, monkeypatch):
    from app.services import entity_reader

    roles = {r["name"]: r for r in client.get("/api/graph/g1/roles").get_json()["data"]["roles"]}
    ops = [
        {"op": "rename", "uuid": roles["師林書瑤"]["uuid"], "name": "林書瑤二"},
        {"op": "merge", "keep": roles["雲梯科技"]["uuid"], "drop": roles["雲梯"]["uuid"]},
        {"op": "exclude", "uuid": roles["數位前線"]["uuid"]},
        {"op": "add", "name": "OpenAI", "type": "Company"},
    ]
    body = client.post("/api/graph/g1/roles", json={"ops": ops}).get_json()
    assert body["success"], body
    after = {r["name"]: r for r in body["data"]["roles"]}
    assert after["數位前線"]["excluded"] and after["數位前線"]["type"] is None
    assert "雲梯" not in after and "OpenAI" in after

    monkeypatch.setattr(entity_reader, "get_graph_store", lambda **kw: store)
    prepared = entity_reader.EntityReader().filter_defined_entities("g1", enrich_with_edges=False)
    names = {e.name for e in prepared.entities}
    assert {"林書瑤二", "雲梯科技", "OpenAI"} <= names
    assert not names & {"師林書瑤", "雲梯", "數位前線"}


def test_the_api_reports_the_op_that_failed(client):
    body = client.post("/api/graph/g1/roles", json={"ops": [{"op": "add", "name": "X", "type": "Planet"}]})
    assert body.status_code == 400 and body.get_json()["failed_op"] == 0
    assert client.post("/api/graph/g1/roles", json={"ops": [{"op": "fly"}]}).status_code == 400
    assert client.post("/api/graph/g1/roles", json={}).status_code == 400
    assert client.get("/api/graph/nope/roles").status_code == 404


def test_the_zep_backend_is_not_supported(monkeypatch):
    from app import create_app
    from app.graph import store as store_module

    monkeypatch.setattr(store_module, "get_graph_store", lambda **kw: object())
    assert create_app().test_client().get("/api/graph/g1/roles").status_code == 501


def test_merging_folds_edges_that_now_repeat(store):
    store.add_text_episodes("g1", [TextEpisode("林書瑤任職於雲梯。")], durable=True)
    roles = _roles(store)
    store.merge_nodes("g1", roles["雲梯科技"].uuid, roles["雲梯"].uuid)
    count = len(store.list_edges("g1"))
    store.add_text_episodes("g1", [TextEpisode("林書瑤任職於雲梯。")], durable=True)
    assert len(store.list_edges("g1")) == count  # the same fact is not added twice
    triples = [(e.source_node_uuid, e.name, e.target_node_uuid, e.fact) for e in store.list_edges("g1")]
    assert len(triples) == len(set(triples))


def test_a_vector_failure_does_not_undo_or_fail_an_edit(store, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("embedding dimension changed from 256 to 8")

    monkeypatch.setattr(store, "_store_vectors", broken)
    node = _roles(store)["師林書瑤"]
    store.rename_node("g1", node.uuid, "林書瑤二")  # does not raise
    assert "林書瑤二" in _roles(store)
    assert store.embedding_state("g1") == "incomplete"  # --check says to re-embed


def test_edits_wait_while_a_simulation_owns_the_graph(client, monkeypatch):
    from app.services.graph_memory_updater import GraphMemoryManager

    monkeypatch.setattr(GraphMemoryManager, "get_simulation_ids_for_graph", classmethod(lambda cls, g: ["sim_1"]))
    response = client.post("/api/graph/g1/roles", json={"ops": [{"op": "add", "name": "X", "type": "Company"}]})
    assert response.status_code == 409 and "sim_1" in response.get_json()["error"]


def test_the_api_rejects_non_string_fields(client):
    roles = {r["name"]: r for r in client.get("/api/graph/g1/roles").get_json()["data"]["roles"]}
    response = client.post("/api/graph/g1/roles", json={"ops": [{"op": "rename", "uuid": roles["雲梯"]["uuid"], "name": 123}]})
    assert response.status_code == 400 and response.get_json()["failed_op"] == 0
