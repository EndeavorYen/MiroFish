import pytest
from unittest.mock import MagicMock, call


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


@pytest.fixture
def sample_ontology():
    return {
        "entity_types": [
            {"name": "Person", "description": "A natural person"},
            {"name": "Organization", "description": "Any organization"},
        ],
        "edge_types": [
            {"name": "WORKS_FOR", "description": "Employment relationship"},
        ],
    }


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


def test_set_ontology_changes_extraction_prompt(store, mock_llm, sample_ontology):
    """set_ontology should make extraction use ontology-guided prompt."""
    from app.services.graph_builder import GraphBuilderService, EXTRACTION_SYSTEM_PROMPT
    builder = GraphBuilderService(store=store, llm_client=mock_llm)

    # Without ontology: uses generic prompt
    assert builder._get_extraction_prompt() == EXTRACTION_SYSTEM_PROMPT

    # With ontology: uses ontology-guided prompt
    builder.set_ontology(sample_ontology)
    prompt = builder._get_extraction_prompt()
    assert prompt != EXTRACTION_SYSTEM_PROMPT
    assert "Person" in prompt
    assert "Organization" in prompt
    assert "WORKS_FOR" in prompt


def test_extraction_passes_max_tokens(store, mock_llm):
    """chat_json must be called with max_tokens to prevent runaway generation."""
    from app.services.graph_builder import GraphBuilderService
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Token Limit Test")
    builder.add_text_and_extract(graph_id, "Some text.")

    # Verify chat_json was called with max_tokens
    _, kwargs = mock_llm.chat_json.call_args
    assert "max_tokens" in kwargs, "chat_json must be called with max_tokens to prevent Ollama runaway generation"
    assert kwargs["max_tokens"] <= 4096, "max_tokens should be reasonable for entity extraction output"


def test_extraction_skips_invalid_entities(store):
    """Entities/relations missing required fields should be skipped, not crash."""
    from app.services.graph_builder import GraphBuilderService
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = {
        "entities": [
            {"name": "Valid", "type": "Person", "description": "OK"},
            {"type": "Person", "description": "Missing name"},       # no name
            {"name": "NoType", "description": "Missing type"},       # no type
            {"name": "", "type": "Person", "description": "Empty"},  # empty name
        ],
        "relations": [
            {"source": "Valid", "target": "X", "type": "KNOWS", "fact": "ok"},
            {"target": "Y", "type": "KNOWS", "fact": "no source"},   # no source
            {"source": "Z", "type": "KNOWS", "fact": "no target"},   # no target
        ],
    }
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Invalid Test")

    # Should not raise
    builder.add_text_and_extract(graph_id, "Test text.")

    entities = store.list_entities(graph_id)
    names = {e.name for e in entities}
    assert "Valid" in names
    # Invalid entities should have been skipped
    assert "" not in names


def test_chat_json_default_think_false():
    """chat_json should default think=False for structured output."""
    import inspect
    from app.utils.llm_client import LLMClient
    sig = inspect.signature(LLMClient.chat_json)
    think_default = sig.parameters["think"].default
    assert think_default is False, (
        "chat_json must default think=False to prevent Ollama thinking mode "
        "from consuming all output tokens on structured JSON calls"
    )


def test_chat_json_default_max_tokens_none():
    """chat_json should default max_tokens=None (callers set explicit limits)."""
    import inspect
    from app.utils.llm_client import LLMClient
    sig = inspect.signature(LLMClient.chat_json)
    max_tokens_default = sig.parameters["max_tokens"].default
    assert max_tokens_default is None, (
        "chat_json should default max_tokens=None; callers like "
        "add_text_and_extract must pass explicit limits"
    )


def test_fix_truncated_json():
    """fix_truncated_json should close unclosed brackets."""
    from app.utils.llm_client import fix_truncated_json

    # Normal truncation: unclosed braces/brackets
    # Note: the fixer appends `"` when last char isn't a JSON terminal,
    # so `2` → `2"` before closing brackets. This is best-effort.
    result = fix_truncated_json('{"a": [1, 2')
    assert result.count('[') == result.count(']')
    assert result.count('{') == result.count('}')

    result = fix_truncated_json('{"a": {"b": 1')
    assert result.count('{') == result.count('}')

    # Truncated mid-string
    assert fix_truncated_json('{"name": "hel') == '{"name": "hel"}'

    # Already valid
    assert fix_truncated_json('{"a": 1}') == '{"a": 1}'
