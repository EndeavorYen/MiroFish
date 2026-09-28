import json

import pytest

from app.simulation_policy.content import ContentIntent
from app.simulation_policy.tiers import TieredContentProvider, load_stance_bank, load_templates


def _intent(stance, agent=1, kind="criticism"):
    return ContentIntent(
        kind=kind, stance=stance, intensity=0.5, target_ref=None, persona_ref=agent,
        platform="twitter", round_num=3, agent_name=f"a{agent}", persona="市民", topic="票價",
    )


BANK = {
    "lang": "zh",
    "seeds": [11, 12, 13],
    "levels": {
        "neg_strong": ["{entity}這樣做根本是把風險丟給大家，絕對不能接受", "堅決反對{entity}，{topic}必須停下來"],
        "neg": ["對{entity}還是很擔心"],
        "neu": ["想先看{entity}的完整資料"],
        "pos": ["{entity}這一步方向是對的"],
        "pos_strong": ["全力支持{entity}！"],
    },
}


def _provider(bank, **kwargs):
    return TieredContentProvider(templates=load_templates("zh"), entities=["凌雲公司"], stance_bank=bank, **kwargs)


def test_template_tier_draws_from_the_bank_at_the_same_level():
    provider = _provider(BANK, bank_share=1.0)
    lines = {provider.template_base(_intent(0.05, agent=i)) for i in range(20)}
    expected = {line.format(entity="凌雲公司", topic="票價") for line in BANK["levels"]["neg_strong"]}
    assert lines <= expected
    assert len(lines) == 2


def test_no_bank_keeps_the_locale_templates():
    with_bank = _provider(None)
    assert with_bank.stance_bank is None
    line = with_bank.template_base(_intent(0.05))
    assert "{" not in line


def test_shared_and_full_prompts_carry_one_same_level_example():
    provider = _provider(BANK, bank_share=0.0)
    prompt = provider._shared_prompt(_intent(0.9))
    assert "全力支持凌雲公司！" in prompt
    full = provider._full_prompt(_intent(0.5))
    assert "想先看凌雲公司的完整資料" in full
    assert "全力支持" not in full


def test_bank_refuses_eval_seeds(tmp_path):
    path = tmp_path / "zh_stance_bank.json"
    path.write_text(json.dumps({**BANK, "seeds": [2, 11]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_stance_bank("zh", path)
    assert load_stance_bank("zh", tmp_path / "missing.json") is None


def test_deidentify_keeps_one_entity_and_drops_the_rest():
    from scripts.build_stance_bank import deidentify

    names = ["星河食品", "星河", "南嶺市衛生局"]
    assert deidentify("星河食品這次真的要給交代", names) == "{entity}這次真的要給交代"
    assert deidentify("星河食品和南嶺市衛生局都有責任", names) is None
    assert deidentify("#食安 星河食品要負責", names) == "{entity}要負責"
    assert deidentify("看 http://x 星河食品", names) is None
    assert deidentify("大家先冷靜看檢驗結果", names) == "大家先冷靜看檢驗結果"


def test_bank_builder_refuses_eval_seeds(tmp_path):
    from scripts.build_stance_bank import main

    with pytest.raises(SystemExit):
        main(["--dirs", str(tmp_path), "--lang", "zh", "--seeds", "1", "11", "--out", str(tmp_path / "b.json")])


def test_bank_lines_must_be_in_the_bank_language():
    from scripts.build_stance_bank import usable_length

    assert usable_length("The toll is overdue, build the bike lane", "en")
    assert not usable_length("学生终于可以拥有一个安静的课后学习空间了", "en")
    assert usable_length("学生终于可以拥有一个安静的课后学习空间了", "zh")
    assert not usable_length("The toll is overdue, build the bike lane now", "zh")


def test_template_audit_passes_on_two_of_three_fills():
    from scripts.audit_templates import audit

    templates = {"opinion": {"pos": ["{topic} ok"], "neu": ["{entity} hm"]}}
    fills = [("E1", "T1"), ("E2", "T2"), ("E3", "T3")]
    reads = {"T1 ok": "pos", "T2 ok": "pos_strong", "T3 ok": "pos", "E1 hm": "pos", "E2 hm": "neu", "E3 hm": "pos"}
    report = audit(templates, fills, reads.__getitem__)
    assert report["passed"] == 1 and report["total"] == 2
    assert report["confusion"]["pos"] == {"pos": 2, "pos_strong": 1}
