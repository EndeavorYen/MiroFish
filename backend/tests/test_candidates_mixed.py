"""Candidates in mixed Chinese/English news (#64)."""

from app.graph.candidates import find_candidates
from app.graph.local_extractor import LocalExtractor, _Typed
from app.system_one.models import NoulAnswer


def _options(text):
    return {option for c in find_candidates(text) for option in c.options()}


def test_latin_names_inside_chinese_text_are_candidates():
    text = ("OpenAI 宣布停止販售每月 200 美元的 ChatGPT Pro 方案。"
            "科技記者指出，Anthropic 與 Google 也有高階方案。這反映生成式 AI 的真實成本。")
    options = _options(text)
    assert {"OpenAI", "ChatGPT Pro", "Anthropic", "Google"} <= options
    assert "AI" not in options  # a generic abbreviation, not a name
    assert "200" not in options


def test_a_title_glued_to_a_name_is_not_part_of_it():
    options = _options("產業分析師林書瑤認為，這反映真實成本。")
    assert "林書瑤" in options
    assert not any(o.endswith("林書瑤") and o != "林書瑤" for o in options)


def test_a_petition_is_not_an_organisation():
    options = _options("「台灣開發者社群」的版主劉家豪發起連署，要求保留方案。")
    assert "劉家豪" in options
    assert not any("連署" in o for o in options)


def test_a_fragment_inside_a_longer_name_is_not_its_own_candidate():
    options = _options("新創公司「雲梯科技」執行長高子恆說，他們願意付費。")
    assert "雲梯科技" in options
    assert "雲梯" not in options


def test_a_short_company_name_merges_with_its_full_name():
    class Same:
        calls = 0

        def noul(self, state, instructions):
            Same.calls += 1
            return NoulAnswer(noul=0.95)

    canonical = LocalExtractor(Same())._merge(
        # jieba reads 雲梯 as a person name; it is 雲梯科技 without 科技.
        [_Typed("雲梯", "Company", 0.9, "person"), _Typed("雲梯科技", "Company", 0.9, "org")],
        [],
        ["雲梯科技今天發表新產品。", "雲梯表示願意付費。"],
    )
    assert canonical["雲梯"] == "雲梯科技" and Same.calls == 1
