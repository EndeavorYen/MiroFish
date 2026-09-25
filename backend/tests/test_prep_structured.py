"""Decode-free preparation (#11) with a fake System One client."""

import hashlib
import json
from pathlib import Path

import pytest

from app.config import Config
from app.services.prep_structured import (
    ENTITY_TYPE_COUNT,
    load_templates,
    seed_hot_topics,
    structured_agent_config,
    structured_event_config,
    structured_profile,
    template_ontology,
)
from app.system_one.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResponse,
)

SEED = (Path(__file__).parent / "fixtures" / "golden_scenario" / "news_seed.txt").read_text(encoding="utf-8")


class FakeSystemOne:
    def __init__(self):
        self.questions = 0

    def ask(self, request):
        answers = {}
        for name, q in request.questions.items():
            self.questions += 1
            h = hashlib.sha256((request.state + name).encode()).digest()
            if isinstance(q, ChoiceQuestion):
                keys = list(q.criteria)
                pick = keys[h[0] % len(keys)]
                probs = {k: (0.7 if k == pick else 0.3 / max(len(keys) - 1, 1)) for k in keys}
                answers[name] = ChoiceAnswer(choice=pick, probabilities=probs, confidence=0.7)
            elif isinstance(q, ScoreQuestion):
                n = len(q.criteria)
                answers[name] = ScoreAnswer(
                    score=(h[1] / 255) * (n - 1),
                    probabilities={c: 1 / n for c in q.criteria},
                    confidence=1 / n,
                )
            elif isinstance(q, NoulQuestion):
                answers[name] = NoulAnswer(noul=h[2] / 255)
        return SystemOneResponse(model="fake", answers=answers)


def test_template_ontology_has_ten_types_ending_in_fallbacks():
    from app.services.ontology_generator import OntologyGenerator

    ontology = template_ontology(FakeSystemOne(), SEED, "模擬輿論")
    names = [e["name"] for e in ontology["entity_types"]]
    assert len(names) == ENTITY_TYPE_COUNT == len(set(names))
    assert names[-2:] == ["Person", "Organization"]
    known = set(load_templates()["entity_types"])
    assert set(names) <= known
    for edge in ontology["edge_types"]:
        assert edge["source_targets"]
        for pair in edge["source_targets"]:
            assert pair["source"] in names and pair["target"] in names
    processed = OntologyGenerator.__new__(OntologyGenerator)._validate_and_process(dict(ontology))
    assert [e["name"] for e in processed["entity_types"]][-2:] == ["Person", "Organization"]


def test_every_template_type_is_defined():
    data = load_templates()
    for template in data["templates"].values():
        for name in template["core"] + template["optional"]:
            assert name in data["entity_types"], name
        assert len(template["core"]) <= ENTITY_TYPE_COUNT - 2


def test_structured_profile_fields_are_oasis_compatible():
    from app.services.oasis_profile_generator import OasisAgentProfile

    for kind_hash_seed in ("陳維", "東海市交通運輸委員會", "王淑芬"):
        data = structured_profile(FakeSystemOne(), kind_hash_seed, "Person", "摘要", "上下文", seed="x")
        profile = OasisAgentProfile(user_id=1, user_name="u", name=kind_hash_seed, **{
            k: v for k, v in data.items() if not k.endswith("_score")
        })
        assert profile.to_reddit_format()["persona"]
        assert profile.to_twitter_format()
        assert data["gender"] in ("male", "female", "other")
        assert 18 <= data["age"] <= 80
        assert 0 <= data["stance_score"] <= 1
        assert data["interested_topics"]


def test_generate_profile_from_entity_uses_structured_mode_without_llm(monkeypatch):
    from app.services import oasis_profile_generator as opg
    from app.services.zep_entity_reader import EntityNode

    monkeypatch.setattr(Config, "PROFILE_MODE", "structured")
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
    monkeypatch.setattr("app.system_one.client.get_system_one_client", lambda: FakeSystemOne())

    def no_llm(*args, **kwargs):
        raise AssertionError("LLM must not be called in structured mode")

    monkeypatch.setattr(opg.OasisProfileGenerator, "_generate_profile_with_llm", no_llm)
    generator = opg.OasisProfileGenerator(api_key="k", base_url="http://127.0.0.1:9", zep_api_key=None)
    entity = EntityNode(uuid="e1", name="陳維", labels=["Entity", "CorporateExecutive"], summary="凌雲科技執行長", attributes={})
    profile = generator.generate_profile_from_entity(entity, user_id=3, use_llm=True)
    assert profile.name == "陳維" and profile.persona and profile.bio


def test_structured_agent_and_event_config():
    client = FakeSystemOne()
    cfg = structured_agent_config(client, "林建東", "GovernmentOfficial", "交通委發言人")
    assert 0.1 <= cfg["activity_level"] <= 0.9
    assert cfg["stance"] in ("supportive", "opposing", "neutral")
    assert cfg["active_hours"] and all(0 <= h <= 23 for h in cfg["active_hours"])
    assert -1 <= cfg["sentiment_bias"] <= 1

    event = structured_event_config(
        client, SEED, "模擬", ["GovernmentAgency", "Company", "Person"], ["凌雲飛行智能公司", "林建東"]
    )
    assert 1 <= len(event["initial_posts"]) <= 3
    for post in event["initial_posts"]:
        assert post["content"] in SEED and post["poster_type"] in ("GovernmentAgency", "Company", "Person")
    assert "凌雲飛行智能公司" in event["hot_topics"]


def test_hot_topics_are_deterministic():
    assert seed_hot_topics(SEED, ["林建東"]) == seed_hot_topics(SEED, ["林建東"])


def test_config_generator_structured_mode_makes_no_llm_calls(monkeypatch):
    from app.services import simulation_config_generator as scg
    from app.services.zep_entity_reader import EntityNode

    monkeypatch.setattr(Config, "SIM_CONFIG_MODE", "structured")
    monkeypatch.setattr("app.system_one.client.get_system_one_client", lambda: FakeSystemOne())

    def no_llm(*args, **kwargs):
        raise AssertionError("LLM must not be called in structured mode")

    monkeypatch.setattr(scg.SimulationConfigGenerator, "_call_llm_with_retry", no_llm)
    generator = scg.SimulationConfigGenerator(api_key="k", base_url="http://127.0.0.1:9", model_name="m")
    entities = [
        EntityNode(uuid=f"e{i}", name=n, labels=["Entity", t], summary="s", attributes={})
        for i, (n, t) in enumerate([("林建東", "GovernmentOfficial"), ("凌雲飛行智能公司", "Company"), ("王淑芬", "Person")])
    ]
    params = generator.generate_config("sim", "proj", "g", "模擬", SEED, entities)
    assert len(params.agent_configs) == 3
    assert params.time_config.total_simulation_hours == 72
    assert params.event_config.initial_posts


def test_ontology_generator_template_mode_makes_no_llm_calls(monkeypatch, tmp_path):
    from app.services.ontology_generator import OntologyGenerator
    from app.utils.llm_usage import usage_stage

    monkeypatch.setattr(Config, "ONTOLOGY_MODE", "template")
    monkeypatch.setattr("app.system_one.client.get_system_one_client", lambda: FakeSystemOne())

    class NoLlm:
        def chat_json(self, *args, **kwargs):
            raise AssertionError("LLM must not be called in template mode")

    ontology = OntologyGenerator(llm_client=NoLlm()).generate([SEED], "模擬", metrics_dir=str(tmp_path))
    assert len(ontology["entity_types"]) == ENTITY_TYPE_COUNT
    assert not (tmp_path / "llm_usage.jsonl").exists()


def test_edge_pairs_cover_every_source_type_first():
    from app.services.prep_structured import MAX_SOURCE_TARGETS, _round_robin_pairs

    sources = ["A", "B", "C", "D", "E", "F"]
    pairs = _round_robin_pairs(sources, ["X", "Y", "A"])
    assert len(pairs) == MAX_SOURCE_TARGETS
    assert {p["source"] for p in pairs[:6]} == set(sources)
    assert all(p["source"] != p["target"] for p in pairs)


def test_use_llm_false_stays_rule_based_in_structured_mode(monkeypatch):
    from app.services import oasis_profile_generator as opg
    from app.services.zep_entity_reader import EntityNode

    monkeypatch.setattr(Config, "PROFILE_MODE", "structured")

    def no_system_one():
        raise AssertionError("use_llm=False must not call System One")

    monkeypatch.setattr("app.system_one.client.get_system_one_client", no_system_one)
    generator = opg.OasisProfileGenerator(api_key="k", base_url="http://127.0.0.1:9", zep_api_key=None)
    entity = EntityNode(uuid="e1", name="陳維", labels=["Entity", "Person"], summary="s", attributes={})
    assert generator.generate_profile_from_entity(entity, user_id=1, use_llm=False).name == "陳維"
