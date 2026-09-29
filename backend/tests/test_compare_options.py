"""Option comparison (#56)."""

import json

import pytest

from scripts import compare_options as co


def _scan(tendency, oppose, posts=()):
    return {"tendency": {"value": tendency}, "post_shares": {"oppose": oppose},
            "most_opposed_posts": [{"text": t, "stance": s} for t, s in posts]}


def test_options_file_is_checked(tmp_path):
    path = tmp_path / "o.json"
    path.write_text(json.dumps([{"name": "a", "text": "x"}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        co.load_options(path)
    path.write_text(json.dumps([{"name": "a", "text": "x"}, {"name": "a b", "text": "y"}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        co.load_options(path)
    path.write_text(json.dumps([{"name": "a", "text": "x"}, {"name": "b", "text": " y "}]), encoding="utf-8")
    assert co.load_options(path)[1] == {"name": "b", "text": "y"}


def test_option_fixture_adds_the_option(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "news_seed.txt").write_text("市府宣布試點。", encoding="utf-8")
    (base / "simulation_requirement.txt").write_text("模擬試點後的反應。", encoding="utf-8")
    (base / "README.md").write_text("x", encoding="utf-8")
    out = co.option_fixture(base, {"name": "free", "text": "首月免費"}, tmp_path / "free")
    assert (out / "news_seed.txt").read_text(encoding="utf-8").endswith("公布的方案：首月免費\n")
    assert (out / "simulation_requirement.txt").read_text(encoding="utf-8") == "模擬試點後的反應。公布的方案：首月免費"
    assert (out / "README.md").exists()


def test_paired_differences_need_every_seed_to_agree():
    results = {
        "base": {1: _scan(0.6, 0.2, [("太貴", 0.1)]), 2: _scan(0.5, 0.3), 3: _scan(0.55, 0.25)},
        "better": {1: _scan(0.7, 0.1), 2: _scan(0.6, 0.2), 3: _scan(0.65, 0.1)},
        "mixed": {1: _scan(0.7, 0.3), 2: _scan(0.4, 0.2), 3: _scan(0.56, 0.25)},
    }
    rows = co.compare(results, ["base", "better", "mixed"])
    better = rows[1]["vs_baseline"]
    assert better["tendency"]["mean"] == pytest.approx(0.1)
    assert better["tendency"]["distinct"] and better["oppose_share"]["distinct"]
    assert not rows[2]["vs_baseline"]["tendency"]["distinct"]
    assert rows[0]["most_opposed_posts"] == [{"text": "太貴", "stance": 0.1}]
    text = co.render(rows, "問題")
    assert "無法區分" in text and "local-llm" in text and "（基準）" in text
