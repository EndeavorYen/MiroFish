import json
import re
from pathlib import Path

import pytest

from app.graph.candidates import find_candidates
from app.graph.embedding import HashEmbedder
from app.graph.local_extractor import LocalExtractor
from app.system_one.models import ChoiceAnswer, NoulAnswer

FIXTURES = Path(__file__).parent / "fixtures" / "golden_scenario"
ONTOLOGY = json.loads((FIXTURES / "ontology.json").read_text(encoding="utf-8"))
GOLD = json.loads((FIXTURES / "gold_graph.json").read_text(encoding="utf-8"))
SEED = (FIXTURES / "news_seed.txt").read_text(encoding="utf-8")

GOLD_TYPES = {}
for entity in GOLD["entities"]:
    for name in [entity["name"], *entity["aliases"]]:
        GOLD_TYPES[name] = entity["type"]
GOLD_GROUP = {
    name: entity["name"]
    for entity in GOLD["entities"]
    for name in [entity["name"], *entity["aliases"]]
}


class FakeSystemOne:
    """Answers from the gold annotation; counts calls."""

    def __init__(self, bogus_type: str | None = None):
        self.calls = {"choice": 0, "noul": 0}
        self.bogus_type = bogus_type

    def choice(self, state, instructions, criteria):
        self.calls["choice"] += 1
        if "邊界正確" in instructions:
            pick = next((k for k, v in criteria.items() if v in GOLD_TYPES), "none")
        elif "候選名稱" in state:
            name = re.search(r"候選名稱：(.+)", state).group(1).strip()
            pick = self.bogus_type or GOLD_TYPES.get(name, "none")
            if pick not in criteria:
                pick = next(iter(criteria))  # model answers are always an option key
        else:
            a = re.search(r"實體A：(.+?)（", state).group(1)
            b = re.search(r"實體B：(.+?)（", state).group(1)
            pick = "none"
            for key, text in criteria.items():
                for rel in GOLD["relations"]:
                    if {GOLD_GROUP.get(a), GOLD_GROUP.get(b)} == {rel["source"], rel["target"]} and (
                        f" {rel['type']}（" in text
                        and text.startswith(
                            next(n for n in (a, b) if GOLD_GROUP.get(n) == rel["source"])
                        )
                    ):
                        pick = key
        probs = {k: (0.9 if k == pick else 0.1 / max(len(criteria) - 1, 1)) for k in criteria}
        return ChoiceAnswer(choice=pick, probabilities=probs, confidence=probs[pick])

    def noul(self, state, instructions):
        self.calls["noul"] += 1
        a = re.search(r"名稱一：(.+?)（", state).group(1)
        b = re.search(r"名稱二：(.+?)（", state).group(1)
        same = GOLD_GROUP.get(a) is not None and GOLD_GROUP.get(a) == GOLD_GROUP.get(b)
        return NoulAnswer(noul=0.95 if same else 0.05)


def test_candidates_cover_every_gold_entity():
    names = {option for c in find_candidates(SEED) for option in c.options()}
    for entity in GOLD["entities"]:
        assert names & {entity["name"], *entity["aliases"]}, entity["name"]


def test_extracted_types_always_belong_to_the_ontology():
    allowed = {e["name"] for e in ONTOLOGY["entity_types"]}
    edges = {e["name"] for e in ONTOLOGY["edge_types"]}
    # Even a model that answers a non-ontology label cannot leak it.
    for fake in (FakeSystemOne(), FakeSystemOne(bogus_type="Spaceship")):
        extraction = LocalExtractor(fake, embedder=HashEmbedder()).extract(SEED, ONTOLOGY)
        assert extraction.entities
        assert {e.entity_type for e in extraction.entities} <= allowed
        assert {r.name for r in extraction.relations} <= edges


def test_extraction_matches_gold_with_a_perfect_oracle():
    fake = FakeSystemOne()
    extraction = LocalExtractor(fake, embedder=HashEmbedder()).extract(SEED, ONTOLOGY)
    groups = {GOLD_GROUP.get(e.name) for e in extraction.entities}
    assert groups == {e["name"] for e in GOLD["entities"]}
    # Aliases fold into one node per gold entity.
    assert len(extraction.entities) == len(GOLD["entities"])
    company = next(e for e in extraction.entities if GOLD_GROUP[e.name] == "凌雲飛行智能公司")
    assert "凌雲科技" in [company.name, *company.attributes.get("aliases", [])]
    pairs = {(GOLD_GROUP[r.source], GOLD_GROUP[r.target]) for r in extraction.relations}
    for rel in GOLD["relations"]:
        assert (rel["source"], rel["target"]) in pairs, rel
    assert all(r.fact in SEED for r in extraction.relations)
    assert all(e.summary for e in extraction.entities)


def test_known_graph_names_win_as_canonical():
    fake = FakeSystemOne()
    extractor = LocalExtractor(fake, embedder=HashEmbedder())
    text = "市交通委發言人林建東表示，試點將在下月啟動。"
    extraction = extractor.extract(
        text, ONTOLOGY, known_entities=[("交通運輸委員會", "GovernmentAgency")]
    )
    names = {e.name for e in extraction.entities}
    assert "交通運輸委員會" in names and "市交通委" not in names
    assert any(r.target == "交通運輸委員會" and r.source == "林建東" for r in extraction.relations)


def test_no_ontology_means_no_extraction():
    fake = FakeSystemOne()
    assert LocalExtractor(fake).extract(SEED, None).entities == []
    assert fake.calls == {"choice": 0, "noul": 0}


def test_relations_only_use_allowed_directions():
    fake = FakeSystemOne()
    extraction = LocalExtractor(fake, embedder=HashEmbedder()).extract(SEED, ONTOLOGY)
    type_of = {e.name: e.entity_type for e in extraction.entities}
    allowed = {
        (edge["name"], pair["source"], pair["target"])
        for edge in ONTOLOGY["edge_types"]
        for pair in edge["source_targets"]
    }
    for rel in extraction.relations:
        assert (rel.name, type_of[rel.source], type_of[rel.target]) in allowed


def test_summary_fn_is_optional_and_used_when_given():
    fake = FakeSystemOne()
    calls = []

    def summary(name, etype, mentions):
        calls.append(name)
        return f"{name} summary"

    extraction = LocalExtractor(fake, embedder=HashEmbedder(), summary_fn=summary).extract(
        "東海綠色能源研究所給予高度評價。", ONTOLOGY
    )
    assert calls == ["東海綠色能源研究所"]
    assert extraction.entities[0].summary == "東海綠色能源研究所 summary"


def test_too_many_entity_types_is_rejected():
    ontology = {"entity_types": [{"name": f"T{i}"} for i in range(26)]}
    with pytest.raises(ValueError):
        LocalExtractor(FakeSystemOne()).extract("x", ontology)


def test_candidate_variants_fix_common_boundary_errors():
    text = "衛生局今日宣布，本地食品大廠星河食品股份有限公司已委託綠源檢測研究院檢驗。執行長高明哲致歉，教授羅建國受訪。"
    options = {c.text: c.options() for c in find_candidates(text)}
    flat = {o for opts in options.values() for o in opts}
    assert "星河食品股份有限公司" in flat
    assert "綠源檢測研究院" in flat and "託綠源檢測研究院" not in flat
    assert "高明哲" in flat and "羅建國" in flat
    assert not any(name.startswith("託") for name in flat)


def test_single_character_stop_words_inside_names_do_not_cut_them():
    cases = {
        "國際和平基金會今日發表聲明。": "國際和平基金會",
        "台灣經濟研究院指出成長放緩。": "台灣經濟研究院",
        "同濟大學教授表示。": "同濟大學",
        "立法院財經委員會召開會議。": "立法院財經委員會",
    }
    for text, name in cases.items():
        assert name in {o for c in find_candidates(text) for o in c.options()}, text


def test_nested_names_do_not_co_occur_with_themselves():
    fake = FakeSystemOne()
    extractor = LocalExtractor(fake)
    surfaces = {"東海市政府": ["東海市政府"], "市政府": ["市政府"]}
    types = {"東海市政府": "GovernmentAgency", "市政府": "GovernmentAgency"}
    relations = extractor._relations(["東海市政府今天開會。"], surfaces, types, ONTOLOGY)
    assert relations == []
    assert fake.calls["choice"] == 0


def test_person_and_organization_fallbacks_are_not_merge_candidates():
    class AlwaysSame(FakeSystemOne):
        def noul(self, state, instructions):
            self.calls["noul"] += 1
            return NoulAnswer(noul=0.99)

    fake = AlwaysSame()
    from app.graph.local_extractor import _Typed

    canonical = LocalExtractor(fake)._merge(
        [_Typed("王明", "Person", 0.9), _Typed("王明基金會", "Organization", 0.9)], [], []
    )
    assert canonical["王明"] == "王明" and canonical["王明基金會"] == "王明基金會"
    assert fake.calls["noul"] == 0


def test_aliases_accumulate_across_episodes(tmp_path):
    from app.graph.extractor import ExtractedEntity, Extraction
    from app.graph.local_store import LocalGraphStore
    from app.graph.store import TextEpisode

    class Scripted:
        def __init__(self):
            self.items = iter([
                Extraction([ExtractedEntity("凌雲飛行智能公司", "TechCompany", "a", {"aliases": ["凌雲科技"]})]),
                Extraction([ExtractedEntity("凌雲飛行智能公司", "TechCompany", "b", {"aliases": ["凌雲飛行"]})]),
            ])

        def extract(self, text, ontology, known_entities=None):
            return next(self.items)

    store = LocalGraphStore(str(tmp_path), embedder=HashEmbedder(), extractor=Scripted())
    store.create_graph("g", graph_id="g1")
    store.add_text_episodes("g1", [TextEpisode("a"), TextEpisode("b")], durable=True)
    node = store.list_nodes("g1")[0]
    assert node.attributes["aliases"] == ["凌雲科技", "凌雲飛行"]
    store.close()
