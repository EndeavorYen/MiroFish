import pytest
from unittest.mock import MagicMock


@pytest.fixture
def store(tmp_db_path):
    from app.services.networkx_graph_store import NetworkXGraphStore
    return NetworkXGraphStore(db_path=tmp_db_path)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "entities": [
            {"name": "Elon Musk", "type": "Person", "description": "CEO of Tesla"},
            {"name": "Tesla", "type": "Organization", "description": "Electric car company"},
        ],
        "relations": [
            {"source": "Elon Musk", "target": "Tesla", "type": "CEO_OF", "fact": "Elon Musk is CEO of Tesla"},
        ],
    }
    return llm


def test_build_graph_from_text(store, mock_llm):
    from app.services.graph_builder import GraphBuilderService
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Test Graph")
    builder.add_text_and_extract(graph_id, "Elon Musk is the CEO of Tesla, an electric car company.")
    entities = store.list_entities(graph_id)
    names = {e.name for e in entities}
    assert "Elon Musk" in names
    assert "Tesla" in names
    relations = store.list_relations(graph_id)
    assert len(relations) >= 1


def test_build_graph_deduplicates_entities(store, mock_llm):
    from app.services.graph_builder import GraphBuilderService
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Dedup Test")
    builder.add_text_and_extract(graph_id, "Chunk 1 about Elon Musk.")
    builder.add_text_and_extract(graph_id, "Chunk 2 about Elon Musk.")
    entities = store.list_entities(graph_id)
    elon_count = sum(1 for e in entities if e.name == "Elon Musk")
    assert elon_count == 1  # deduplicated
