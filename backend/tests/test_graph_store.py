from datetime import datetime


def test_entity_node_creation():
    from app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Person"],
        summary="A researcher", attributes={"age": 30},
        related_edges=[], related_nodes=[]
    )
    assert node.name == "Alice"
    assert node.labels == ["Person"]


def test_entity_node_to_dict():
    from app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Person"],
        summary="A researcher", attributes={},
        related_edges=[], related_nodes=[]
    )
    d = node.to_dict()
    assert d["uuid"] == "n1"
    assert d["name"] == "Alice"


def test_entity_node_get_entity_type():
    from app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Entity", "Person"],
        summary="", attributes={}, related_edges=[], related_nodes=[]
    )
    assert node.get_entity_type() == "Person"


def test_entity_node_get_entity_type_generic_only():
    from app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Entity"],
        summary="", attributes={}, related_edges=[], related_nodes=[]
    )
    assert node.get_entity_type() == "Entity"


def test_relation_edge_creation():
    from app.services.graph_store import RelationEdge
    edge = RelationEdge(
        uuid="e1", name="KNOWS", fact="Alice knows Bob",
        source_node_uuid="n1", target_node_uuid="n2",
        attributes={}, created_at=datetime.now()
    )
    assert edge.fact == "Alice knows Bob"
    assert edge.valid_at is None


def test_search_result_creation():
    from app.services.graph_store import SearchResult
    result = SearchResult(nodes=[], edges=[], facts=["fact1"])
    assert len(result.facts) == 1


def test_community_creation():
    from app.services.graph_store import Community
    c = Community(id=0, members=["n1", "n2"])
    assert c.summary is None
    assert len(c.members) == 2
