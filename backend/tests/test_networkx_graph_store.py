import pytest
from datetime import datetime
from app.services.networkx_graph_store import NetworkXGraphStore


@pytest.fixture
def store(tmp_db_path):
    from app.services.networkx_graph_store import NetworkXGraphStore
    return NetworkXGraphStore(db_path=tmp_db_path)


@pytest.fixture
def populated_store(store, graph_id):
    """Store with a graph containing entities and relations."""
    store.create_graph(graph_id, "Test Graph", "For testing")
    store.add_entity(graph_id, "Alice", "Person", "A researcher", {"age": 30})
    store.add_entity(graph_id, "Bob", "Person", "An engineer", {"age": 25})
    store.add_entity(graph_id, "ACME", "Organization", "A company", {})
    return store


class TestGraphLifecycle:
    def test_create_graph(self, store, graph_id):
        result = store.create_graph(graph_id, "My Graph", "A test graph")
        assert result == graph_id

    def test_create_duplicate_graph_raises(self, store, graph_id):
        store.create_graph(graph_id, "G1", "")
        with pytest.raises(ValueError, match="already exists"):
            store.create_graph(graph_id, "G2", "")

    def test_delete_graph(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        store.add_entity(graph_id, "Alice", "Person", "", {})
        store.delete_graph(graph_id)
        assert store.list_entities(graph_id) == []


class TestEntityOperations:
    def test_add_and_get_entity(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        uuid = store.add_entity(graph_id, "Alice", "Person", "A researcher", {"age": 30})
        entity = store.get_entity(graph_id, uuid)
        assert entity.name == "Alice"
        assert entity.labels == ["Person"]
        assert entity.attributes["age"] == 30

    def test_add_duplicate_entity_deduplicates(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        uuid1 = store.add_entity(graph_id, "Alice", "Person", "Short bio", {})
        uuid2 = store.add_entity(graph_id, "Alice", "Person", "Longer, richer bio", {"age": 30})
        assert uuid1 == uuid2
        entity = store.get_entity(graph_id, uuid1)
        assert entity.summary == "Longer, richer bio"
        assert entity.attributes["age"] == 30

    def test_list_entities(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        assert len(entities) == 3
        names = {e.name for e in entities}
        assert names == {"Alice", "Bob", "ACME"}

    def test_list_entities_with_limit(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id, limit=2)
        assert len(entities) == 2

    def test_get_entity_edges(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        alice = next(e for e in entities if e.name == "Alice")
        bob = next(e for e in entities if e.name == "Bob")
        populated_store.add_relation(
            graph_id, alice.uuid, bob.uuid,
            "KNOWS", "Alice knows Bob"
        )
        edges = populated_store.get_entity_edges(graph_id, alice.uuid)
        assert len(edges) == 1
        assert edges[0].fact == "Alice knows Bob"


class TestRelationOperations:
    def test_add_and_list_relations(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        alice = next(e for e in entities if e.name == "Alice")
        acme = next(e for e in entities if e.name == "ACME")
        populated_store.add_relation(
            graph_id, alice.uuid, acme.uuid,
            "WORKS_AT", "Alice works at ACME",
            attributes={"role": "lead"},
            temporal={"valid_at": datetime(2025, 1, 1).isoformat()}
        )
        relations = populated_store.list_relations(graph_id)
        assert len(relations) == 1
        assert relations[0].name == "WORKS_AT"
        assert relations[0].valid_at is not None


class TestSearch:
    def test_search_nodes_by_name(self, populated_store, graph_id):
        result = populated_store.search(graph_id, "Alice", scope="nodes")
        assert len(result.nodes) >= 1
        assert result.nodes[0].name == "Alice"

    def test_search_edges_by_fact(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        alice = next(e for e in entities if e.name == "Alice")
        bob = next(e for e in entities if e.name == "Bob")
        populated_store.add_relation(
            graph_id, alice.uuid, bob.uuid,
            "KNOWS", "Alice knows Bob from university"
        )
        result = populated_store.search(graph_id, "university", scope="edges")
        assert len(result.edges) >= 1

    def test_search_all(self, populated_store, graph_id):
        result = populated_store.search(graph_id, "researcher", scope="all")
        assert len(result.facts) >= 1

    def test_search_tokenized_mixed_query(self, populated_store, graph_id):
        """Mixed CJK + Latin queries should match on individual tokens."""
        # "Alice研究" should match Alice (latin token "alice")
        result = populated_store.search(graph_id, "Alice研究", scope="nodes")
        assert len(result.nodes) >= 1
        assert result.nodes[0].name == "Alice"

    def test_tokenize_query(self):
        tokens = NetworkXGraphStore._tokenize_query("OPEC石油价格")
        assert "opec" in tokens
        assert "石油价格" in tokens

        tokens2 = NetworkXGraphStore._tokenize_query("Alice Bob")
        assert "alice" in tokens2
        assert "bob" in tokens2


class TestTextStore:
    def test_add_and_search_text(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        store.add_text(graph_id, "agent1", "央行升息對房市造成衝擊",
                       "CREATE_POST", {}, tick=1, timestamp=datetime.now())
        results = store.search_text(graph_id, "房市")
        assert len(results) >= 1

    def test_get_by_agent(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        store.add_text(graph_id, "agent1", "Post 1", "CREATE_POST", {}, 1, datetime.now())
        store.add_text(graph_id, "agent2", "Post 2", "CREATE_POST", {}, 2, datetime.now())
        results = store.get_by_agent(graph_id, "agent1")
        assert len(results) == 1


class TestCommunityDetection:
    def test_louvain_with_interactions(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        agents = {}
        for name in ["A", "B", "C", "D", "E", "F"]:
            agents[name] = store.add_entity(graph_id, name, "agent", "", {})
        for src, tgt in [("A","B"),("B","C"),("A","C")]:
            store.add_relation(graph_id, agents[src], agents[tgt], "INTERACTS", "",
                               attributes={"weight": 5.0})
        for src, tgt in [("D","E"),("E","F"),("D","F")]:
            store.add_relation(graph_id, agents[src], agents[tgt], "INTERACTS", "",
                               attributes={"weight": 5.0})
        store.add_relation(graph_id, agents["C"], agents["D"], "INTERACTS", "",
                           attributes={"weight": 1.0})
        communities = store.detect_communities(graph_id)
        assert len(communities) >= 2


class TestSerialization:
    def test_snapshot_and_restore(self, populated_store, graph_id):
        data = populated_store.snapshot(graph_id)
        assert isinstance(data, bytes)
        restored_id = populated_store.restore(data)
        entities = populated_store.list_entities(restored_id)
        assert len(entities) == 3
