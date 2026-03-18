"""
End-to-end integration test for the de-Zep'd pipeline.
Tests the full flow: graph building → entity reading → memory updates → search.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock


@pytest.fixture
def store(tmp_db_path):
    from app.services.networkx_graph_store import NetworkXGraphStore
    return NetworkXGraphStore(db_path=tmp_db_path)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mock for entity extraction
    llm.chat_json.side_effect = [
        # First call: entity extraction from document
        {
            "entities": [
                {"name": "央行", "type": "Organization", "description": "中央銀行"},
                {"name": "升息政策", "type": "Policy", "description": "央行升息決策"},
                {"name": "房市", "type": "Topic", "description": "房地產市場"},
            ],
            "relations": [
                {"source": "央行", "target": "升息政策", "type": "IMPLEMENTS", "fact": "央行實施升息政策"},
                {"source": "升息政策", "target": "房市", "type": "IMPACTS", "fact": "升息政策影響房市"},
            ],
        },
        # Second call: InsightForge sub-question decomposition
        {"sub_questions": ["升息影響", "房市變化", "央行決策"]},
    ]
    return llm


def test_full_pipeline(store, mock_llm):
    from app.services.graph_builder import GraphBuilderService
    from app.services.entity_reader import EntityReader
    from app.services.graph_memory_updater import GraphMemoryUpdater
    from app.services.search_tools import SearchTools

    # Step 1: Build graph from document
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Integration Test")
    builder.add_text_and_extract(graph_id, "央行宣布升息，預計將對房市造成衝擊。")

    # Step 2: Read entities
    reader = EntityReader(store)
    filtered = reader.filter_defined_entities(graph_id)
    assert filtered.filtered_count >= 3
    assert "Organization" in filtered.entity_types

    # Step 3: Simulate agent memory updates
    updater = GraphMemoryUpdater(graph_id=graph_id, store=store)
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "analyst_1",
        "agent_name": "分析師小王",
        "action_type": "CREATE_POST",
        "action_args": {"content": "央行升息後房價恐下跌20%", "post_id": "p1"},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "citizen_1",
        "agent_name": "市民小李",
        "action_type": "LIKE_POST",
        "action_args": {"post_id": "p1", "author": "分析師小王", "content": "房價恐下跌"},
        "round_num": 2,
        "timestamp": datetime.now().isoformat(),
    })

    # Step 4: Search
    tools = SearchTools(graph_id=graph_id, store=store, llm_client=mock_llm)
    result = tools.quick_search("房價")
    assert len(result.facts) > 0

    # Step 5: Community detection (need at least 2 connected nodes)
    communities = store.detect_communities(graph_id)
    assert len(communities) >= 1  # at least one community

    # Step 6: Snapshot and restore
    snapshot = store.snapshot(graph_id)
    restored_id = store.restore(snapshot)
    restored_entities = store.list_entities(restored_id)
    assert len(restored_entities) >= 3
