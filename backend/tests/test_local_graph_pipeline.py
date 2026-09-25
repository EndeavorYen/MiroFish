"""GRAPH_BACKEND=local end to end: build graph -> read entities -> profile.

No ZEP_API_KEY, no network: HashEmbedder + StubExtractor.
"""

import pytest

from app.config import Config
from app.graph.extractor import StubExtractor

ONTOLOGY = {
    "entity_types": [
        {"name": "Company", "description": "a company", "attributes": []},
        {"name": "Agency", "description": "a government agency", "attributes": []},
        {"name": "Person", "description": "a person", "attributes": []},
    ],
    "edge_types": [
        {"name": "REGULATES", "source_targets": [{"source": "Agency", "target": "Company"}]},
        {"name": "WORKS_FOR", "source_targets": [{"source": "Person", "target": "Company"}]},
    ],
}

TEXT = (
    "東海市交通運輸委員會今天批准凌雲飛行開始試營運。"
    "凌雲飛行執行長陳志遠表示，首批六架航空器將在下月載客。"
    "市民林小美說她很期待搭乘。"
)


@pytest.fixture
def local_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "local")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    monkeypatch.delenv("ZEP_API_KEY", raising=False)
    monkeypatch.setattr(Config, "GRAPH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "GRAPH_EMBEDDER", "hash")
    monkeypatch.setattr(
        "app.graph.local_store.make_extractor",
        lambda: StubExtractor(
            {"東海市交通運輸委員會": "Agency", "凌雲飛行": "Company", "陳志遠": "Person"}
        ),
    )

    def no_zep(*_args, **_kwargs):
        raise AssertionError("Zep must not be used by the local backend")

    monkeypatch.setattr("app.utils.zep.get_zep_client", no_zep)


def test_build_read_entities_and_generate_profile(local_backend):
    from app.services.graph_builder import GraphBuilderService
    from app.services.oasis_profile_generator import OasisProfileGenerator
    from app.services.text_processor import TextProcessor
    from app.services.zep_entity_reader import ZepEntityReader

    builder = GraphBuilderService()
    remembered = []
    graph_id = builder.create_graph("golden", graph_id_callback=remembered.append)
    assert remembered == [graph_id]
    builder.set_ontology(graph_id, ONTOLOGY)
    chunks = TextProcessor.split_text(TEXT, chunk_size=40, overlap=0)
    submission = builder.add_text_batches(graph_id, chunks)
    builder._wait_for_batch(submission)
    assert getattr(submission, "batch_id", None) is None

    data = builder.get_graph_data(graph_id)
    assert data["node_count"] == 3
    assert data["edge_count"] >= 2

    reader = ZepEntityReader()
    filtered = reader.filter_defined_entities(graph_id)
    by_name = {entity.name: entity for entity in filtered.entities}
    assert set(by_name) == {"東海市交通運輸委員會", "凌雲飛行", "陳志遠"}
    assert by_name["凌雲飛行"].get_entity_type() == "Company"
    assert by_name["凌雲飛行"].related_edges

    generator = OasisProfileGenerator(
        api_key="llm-key", base_url="http://127.0.0.1:9", graph_id=graph_id
    )
    context = generator._search_zep_for_entity(by_name["陳志遠"])
    assert context["facts"] or context["node_summaries"]

    profile = generator.generate_profile_from_entity(by_name["陳志遠"], user_id=1, use_llm=False)
    assert profile.name == "陳志遠"
    assert profile.user_name
