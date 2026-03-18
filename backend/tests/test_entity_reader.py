import pytest


@pytest.fixture
def store_with_entities(tmp_db_path):
    from app.services.networkx_graph_store import NetworkXGraphStore
    store = NetworkXGraphStore(db_path=tmp_db_path)
    gid = "test-graph"
    store.create_graph(gid, "Test", "")
    store.add_entity(gid, "Alice", "Person", "A researcher", {"role": "lead"})
    store.add_entity(gid, "ACME", "Organization", "A tech company", {})
    store.add_entity(gid, "GenericThing", "Entity", "Untyped thing", {})
    entities = store.list_entities(gid)
    alice = next(e for e in entities if e.name == "Alice")
    acme = next(e for e in entities if e.name == "ACME")
    store.add_relation(gid, alice.uuid, acme.uuid, "WORKS_AT", "Alice works at ACME")
    return store, gid


def test_filter_defined_entities_excludes_generic(store_with_entities):
    from app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid)
    names = {e.name for e in result.entities}
    assert "Alice" in names
    assert "ACME" in names
    assert "GenericThing" not in names


def test_filter_defined_entities_by_type(store_with_entities):
    from app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid, defined_entity_types=["Person"])
    assert all(e.get_entity_type() == "Person" for e in result.entities)


def test_filter_enriches_with_edges(store_with_entities):
    from app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid, enrich_with_edges=True)
    alice = next(e for e in result.entities if e.name == "Alice")
    assert len(alice.related_edges) > 0


def test_filtered_entities_counts(store_with_entities):
    from app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid)
    assert result.total_count == 3
    assert result.filtered_count == 2
