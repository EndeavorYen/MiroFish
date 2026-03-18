import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


@pytest.fixture
def store(tmp_db_path):
    from app.services.networkx_graph_store import NetworkXGraphStore
    store = NetworkXGraphStore(db_path=tmp_db_path)
    gid = "sim-test"
    store.create_graph(gid, "Test Sim", "")
    # Create entities
    a1 = store.add_entity(gid, "Alice", "Person", "政策分析師", {})
    a2 = store.add_entity(gid, "Bob", "Person", "社群媒體用戶", {})
    store.add_relation(gid, a1, a2, "KNOWS", "Alice和Bob是同事")
    # Add searchable text
    store.add_text(gid, "Alice", "央行升息對房市造成衝擊", "CREATE_POST",
                   {}, tick=1, timestamp=datetime.now())
    store.add_text(gid, "Bob", "房價下跌已成趨勢", "CREATE_POST",
                   {}, tick=2, timestamp=datetime.now())
    return store


@pytest.fixture
def tools(store):
    from app.services.search_tools import SearchTools
    mock_llm = MagicMock()
    # Mock LLM to return sub-questions as JSON
    # Use substrings that actually match the test data
    mock_llm.chat_json.return_value = {
        "sub_questions": ["升息", "房市", "央行"]
    }
    return SearchTools(graph_id="sim-test", store=store, llm_client=mock_llm)


def test_quick_search_returns_results(tools):
    result = tools.quick_search("房市")
    assert len(result.facts) > 0


def test_panorama_search_returns_all(tools):
    result = tools.panorama_search("概況")
    assert len(result.nodes) > 0 or len(result.edges) > 0


def test_insight_forge_decomposes_query(tools):
    result = tools.insight_forge("央行升息對社會的整體影響是什麼?")
    assert len(result.facts) > 0
    # Verify LLM was called to decompose
    tools.llm_client.chat_json.assert_called_once()


def test_search_result_to_text(tools):
    result = tools.quick_search("房")
    text = result.to_text(query="房市")
    assert "搜索查询" in text
