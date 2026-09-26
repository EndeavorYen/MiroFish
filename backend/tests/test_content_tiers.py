"""Tiered content generation and System One interviews (#10)."""

import asyncio
import json
import sqlite3

import pytest

from app.simulation_policy.content import ContentIntent
from app.simulation_policy.emotion import AgentStateStore
from app.simulation_policy.interview import augment_interview_prompt, interview_context
from app.simulation_policy.tiers import (
    FULL_MAX_TOKENS,
    SHARED_MAX_TOKENS,
    TieredContentProvider,
    distinct_2,
    load_templates,
    stance_band,
)


def _intent(agent=1, round_num=0, kind="criticism", stance=0.1, target="101"):
    return ContentIntent(
        kind=kind,
        stance=stance,
        intensity=0.5,
        target_ref=target,
        persona_ref=agent,
        platform="twitter",
        round_num=round_num,
        agent_name=f"agent{agent}",
        persona="市民",
        target_text="凌雲飛行智能公司票價太貴",
        topic="空中計程車票價",
    )


class FakeLlm:
    def __init__(self, tokens=30):
        self.calls = []
        self.tokens = tokens

    def __call__(self, prompt, max_tokens):
        self.calls.append((prompt, max_tokens))
        return f"生成內容{len(self.calls)}：凌雲飛行智能公司票價問題", self.tokens


def _provider(tmp_path=None, **kwargs):
    kwargs.setdefault("templates", load_templates("zh"))
    kwargs.setdefault("entities", ["凌雲飛行智能公司", "東海市政府"])
    if tmp_path is not None:
        kwargs.setdefault("metrics_path", str(tmp_path / "content_metrics.jsonl"))
    return TieredContentProvider(**kwargs)


def test_template_tier_needs_no_model_and_keeps_seed_entities():
    provider = _provider()
    text = provider.generate(_intent())
    assert text
    assert "凌雲飛行智能公司" in text or "空中計程車票價" in text
    assert provider._round.tiers == {"template": 1, "shared": 0, "full": 0}


def test_templates_exist_for_every_kind_and_band_in_both_languages():
    import yaml

    from app.simulation_policy.taxonomy import TAXONOMY_PATH

    kinds = yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))["dialogue_kinds"]
    for lang in ("zh", "en"):
        templates = load_templates(lang)["templates"]
        for kind in kinds:
            for band in ("neg", "neu", "pos"):
                assert templates[kind][band], (lang, kind, band)


def test_shared_tier_calls_the_model_once_per_bucket_and_varies_text():
    llm = FakeLlm()
    provider = _provider(llm_fn=llm, budget_per_round=1000)
    texts = [provider.generate(_intent(agent=a)) for a in range(1, 6)]
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == SHARED_MAX_TOKENS
    assert provider._round.tiers["shared"] == 5
    assert provider._round.cache_hits == 4
    assert all("凌雲飛行智能公司" in t for t in texts)
    assert len(set(texts)) > 1
    # A different stance band is a different bucket.
    provider.generate(_intent(agent=9, stance=0.9))
    assert len(llm.calls) == 2


def test_full_tier_for_top_influencers():
    llm = FakeLlm()
    followers = {i: 10 for i in range(1, 11)}
    followers[7] = 5000
    provider = _provider(llm_fn=llm, budget_per_round=1000, followers=followers, top_k_percent=10)
    provider.generate(_intent(agent=7))
    assert provider._round.tiers["full"] == 1
    assert llm.calls[0][1] == FULL_MAX_TOKENS
    assert "agent7" in llm.calls[0][0]


def test_budget_exhaustion_downgrades_to_templates_and_resets_next_round(tmp_path):
    llm = FakeLlm(tokens=SHARED_MAX_TOKENS)
    provider = _provider(tmp_path, llm_fn=llm, budget_per_round=SHARED_MAX_TOKENS * 2)
    for kind in ("criticism", "worry", "question", "humor"):
        provider.generate(_intent(kind=kind))
    stats = provider._round
    assert stats.tiers["shared"] == 2 and stats.tiers["template"] == 2
    assert stats.decode_tokens["shared"] <= SHARED_MAX_TOKENS * 2
    provider.generate(_intent(kind="rumor", round_num=1))
    assert provider._round.tiers["shared"] == 1  # new round, new budget
    provider.flush()
    rows = [json.loads(line) for line in (tmp_path / "content_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round"] for r in rows] == [0, 1]
    assert rows[0]["tiers"] == {"template": 2, "shared": 2, "full": 0}
    assert rows[0]["decode_tokens_total"] == SHARED_MAX_TOKENS * 2
    assert 0 < rows[0]["distinct_2"] <= 1


def test_model_failure_falls_back_to_template():
    def broken(prompt, max_tokens):
        raise TimeoutError("down")

    provider = _provider(llm_fn=broken, budget_per_round=1000)
    assert provider.generate(_intent())
    assert provider._round.tiers == {"template": 1, "shared": 0, "full": 0}
    assert provider._remaining == 1000


def test_distinct_2_and_collapse_warning(monkeypatch):
    import app.simulation_policy.tiers as tiers

    assert distinct_2(["abc", "abc"]) == 0.5
    assert distinct_2([]) == 0.0
    warnings = []
    monkeypatch.setattr(tiers.logger, "warning", lambda msg, *args: warnings.append(msg % args))
    provider = _provider(distinct2_warn=0.9)
    for agent in range(6):
        provider._rollover(0)
        provider._round.texts.append("一模一樣的貼文")
    provider.flush()
    assert warnings and "content collapse" in warnings[0]
    assert stance_band(0.1) == "neg" and stance_band(0.5) == "neu" and stance_band(0.8) == "pos"


def _write_history(sim_dir):
    rows = [
        {"round": 0, "platform": "twitter", "agent_id": 3, "action": "CREATE_POST",
         "args": {"content": "票價太貴了"}, "intent": {"stance": 0.1}},
        {"round": 1, "platform": "twitter", "agent_id": 3, "action": "LIKE_POST",
         "args": {"post_id": 5}},
        {"round": 1, "platform": "twitter", "agent_id": 4, "action": "REPOST", "args": {}},
    ]
    (sim_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    store = AgentStateStore(str(sim_dir / "agent_state.db"))
    store.save(0, "twitter", 3, {"anger": 0.2, "joy": 0.5})
    store.save(1, "twitter", 3, {"anger": 0.6, "joy": 0.3})
    store.close()


def test_interview_context_summarises_decisions_and_emotions(tmp_path, monkeypatch):
    _write_history(tmp_path)
    context = interview_context(str(tmp_path), 3, "twitter")
    assert "发帖 1 次" in context and "点赞帖子 1 次" in context
    assert "票價太貴了" in context
    assert "0.10" in context
    assert "0.20→0.60" in context
    assert interview_context(str(tmp_path), 99, "twitter") == ""

    monkeypatch.setenv("SIM_DECISION_BACKEND", "llm")
    assert augment_interview_prompt("你怎麼看？", str(tmp_path), 3, "twitter") == "你怎麼看？"
    monkeypatch.setenv("SIM_DECISION_BACKEND", "system_one")
    augmented = augment_interview_prompt("你怎麼看？", str(tmp_path), 3, "twitter")
    assert augmented.endswith("你怎麼看？") and "你在模拟中的行为" in augmented


def test_interview_replies_through_the_existing_ipc_flow(tmp_path, monkeypatch):
    from oasis import ActionType

    from scripts.run_twitter_simulation import IPCHandler

    monkeypatch.setenv("SIM_DECISION_BACKEND", "system_one")
    _write_history(tmp_path)
    db = sqlite3.connect(tmp_path / "twitter_simulation.db")
    db.execute("CREATE TABLE trace (user_id INTEGER, created_at TEXT, action TEXT, info TEXT)")
    db.commit()
    seen_prompts = []

    class FakeEnv:
        async def step(self, actions):
            for agent, action in actions.items():
                assert action.action_type == ActionType.INTERVIEW
                prompt = action.action_args["prompt"]
                seen_prompts.append(prompt)
                db.execute(
                    "INSERT INTO trace VALUES (?, ?, ?, ?)",
                    (agent.agent_id, "2026-09-26T00:00:00", ActionType.INTERVIEW.value,
                     json.dumps({"prompt": prompt, "response": "我覺得票價應該降低。"}, ensure_ascii=False)),
                )
                db.commit()

    class FakeAgent:
        agent_id = 3

    class FakeGraph:
        def get_agent(self, agent_id):
            return FakeAgent()

    handler = IPCHandler(str(tmp_path), FakeEnv(), FakeGraph())
    assert asyncio.run(handler.handle_interview("cmd-1", 3, "你怎麼看票價？"))
    response = json.loads((tmp_path / "ipc_responses" / "cmd-1.json").read_text(encoding="utf-8"))
    assert response["status"] == "completed"
    assert response["result"]["response"] == "我覺得票價應該降低。"
    assert "你在模拟中的行为" in seen_prompts[0] and seen_prompts[0].endswith("你怎麼看票價？")
    db.close()


def test_concurrent_agents_share_one_call_per_bucket():
    import threading
    import time

    calls = []

    def slow(prompt, max_tokens):
        calls.append(prompt)
        time.sleep(0.2)
        return "<think>草稿</think>共享內容：凌雲飛行智能公司", 20

    provider = _provider(llm_fn=slow, budget_per_round=1000)
    threads = [threading.Thread(target=provider.generate, args=(_intent(agent=a),)) for a in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert provider._round.tiers["shared"] == 6
    assert all("<think>" not in t and "草稿" not in t for t in provider._round.texts)


def test_null_influence_weight_does_not_break_provider(tmp_path):
    from app.simulation_policy.tiers import build_tiered_provider

    config = {"agent_configs": [{"agent_id": 1, "entity_name": "A", "influence_weight": None}]}
    provider = build_tiered_provider("twitter", str(tmp_path), config, llm_fn=lambda p, m: ("x", 1))
    assert provider.followers == {1: 0}


def test_failed_bucket_falls_back_without_serial_retries():
    import threading
    import time

    calls = []

    def failing(prompt, max_tokens):
        calls.append(prompt)
        time.sleep(0.2)
        raise TimeoutError("down")

    provider = _provider(llm_fn=failing, budget_per_round=1000)
    threads = [threading.Thread(target=provider.generate, args=(_intent(agent=a),)) for a in range(5)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert time.perf_counter() - started < 5
    assert provider._round.tiers == {"template": 5, "shared": 0, "full": 0}
    assert provider._remaining == 1000
    assert not provider._inflight


def test_template_errors_still_release_the_bucket(monkeypatch):
    provider = _provider(llm_fn=lambda p, m: ("", 0), budget_per_round=1000)

    def broken(intent):
        raise KeyError("placeholder")

    monkeypatch.setattr(provider, "template_text", broken)
    with pytest.raises(KeyError):
        provider.generate(_intent())
    assert not provider._inflight and provider._remaining == 1000


def test_clean_generation_handles_missing_opening_tag():
    from app.simulation_policy.tiers import clean_generation

    assert clean_generation("思考中…</think>真正的貼文") == "真正的貼文"
    assert clean_generation("<think>還沒想完") == ""
    assert clean_generation("「一般貼文」") == "一般貼文"


def test_strong_bands_and_five_level_prompts():
    from app.simulation_policy.tiers import stance_words, template_band

    by_kind = {"neg": [], "neu": [], "pos": [], "neg_strong": [], "pos_strong": []}
    assert template_band(0.1, by_kind) == "neg_strong"
    assert template_band(0.2, by_kind) == "neg" and template_band(0.8, by_kind) == "pos"  # boundaries
    assert template_band(0.3, by_kind) == "neg"
    assert template_band(0.7, by_kind) == "pos"
    assert template_band(0.9, by_kind) == "pos_strong"
    assert template_band(0.9, {"neg": [], "neu": [], "pos": []}) == "pos"  # locale without strong bands
    assert [stance_words(x) for x in (0.1, 0.3, 0.5, 0.7, 0.9)] == ["强烈反对", "反对或担忧", "中立", "支持", "强烈支持"]


def test_locales_have_strong_bands_for_every_kind():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "locales"
    for lang in ("zh", "en"):
        templates = json.loads((root / f"{lang}.json").read_text(encoding="utf-8"))["simContent"]["templates"]
        for kind, bands in templates.items():
            assert {"neg", "neu", "pos", "neg_strong", "pos_strong"} <= set(bands), (lang, kind)
    zh = json.loads((root / "zh.json").read_text(encoding="utf-8"))["simContent"]["templates"]
    en = json.loads((root / "en.json").read_text(encoding="utf-8"))["simContent"]["templates"]
    # support + negative stance backs the opposition, never the project
    assert not any("勉强支持" in line or "还算可以" in line for line in zh["support"]["neg"])
    assert not any("I'll back" in line or "did OK" in line for line in en["support"]["neg"])


def test_shared_prompt_wording_matches_its_bucket():
    from app.simulation_policy.content import ContentIntent
    from app.simulation_policy.tiers import TieredContentProvider

    provider = TieredContentProvider.__new__(TieredContentProvider)
    provider.templates = {"templates": {}}
    provider._topic = lambda intent: "试点"
    provider._entity_for = lambda intent: "市府"

    def prompt(stance):
        return provider._shared_prompt(ContentIntent(kind="opinion", stance=stance, intensity=0.5, target_ref=None,
                                                     persona_ref=1))

    assert prompt(0.1) == prompt(0.35)  # same bucket, same wording
    assert "强烈" not in prompt(0.1)
