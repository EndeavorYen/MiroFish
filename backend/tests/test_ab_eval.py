import math

import pytest

import scripts.ab_eval as ab


def test_js_divergence_bounds():
    same = {"like": 3, "post": 1}
    assert ab.js_divergence(same, {"like": 6, "post": 2}) == pytest.approx(0.0)
    assert ab.js_divergence({"like": 1}, {"post": 1}) == pytest.approx(1.0)
    value = ab.js_divergence({"like": 3, "post": 1}, {"like": 1, "post": 3})
    assert 0 < value < 1
    assert value == pytest.approx(ab.js_divergence({"like": 1, "post": 3}, {"like": 3, "post": 1}))


def test_pearson_skips_missing_rounds():
    assert ab.pearson([1, 2, None, 3], [2, 4, 5, 6]) == pytest.approx(1.0)
    assert ab.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert ab.pearson([1, 2], [1, 2]) is None  # too few points
    assert ab.pearson([1, 1, 1], [1, 2, 3]) is None  # flat curve


def test_mean_curve_ignores_empty_rounds():
    assert ab.mean_curve([[0.5, None, 1.0], [0.25, None, None]]) == [0.375, None, 1.0]


def test_render_reports_recommendation():
    report = {
        "seeds": [1, 2],
        "rounds": 3,
        "summary": {
            "A": {"decode_per_round": 150.0, "round_latency_mean_s": 9.0, "distinct_2": 0.4, "entity_mention_rate": 0.4},
            "B": {"decode_per_round": 10.0, "round_latency_mean_s": 20.0, "distinct_2": 0.3, "entity_mention_rate": 0.5},
            "extraction_recall": 1.0,
        },
        "gates": {
            "decode_ratio": {"value": 10 / 150, "passed": True},
            "action_js": {"b_vs_a": 0.05, "a_seed_noise": 0.04, "passed": True},
            "stance_correlation": {"value": math.nan, "a_seed_pairs_mean": None, "passed": False},
            "vram": {"peak_mib": 9000, "budget_mib": 10240, "passed": True},
        },
        "recommendation": "conditional",
    }
    text = ab.render(report)
    assert "有條件切換" in text
    assert "9000 MiB" in text


def test_llm_errors_counts_server_errors(tmp_path):
    (tmp_path / "simulation.log").write_text(
        "ok\nopenai.InternalServerError: Error code: 500 - context\nError code: 400 - bad\nfine\n",
        encoding="utf-8",
    )
    assert ab.llm_errors(tmp_path) == 2
    assert ab.llm_errors(tmp_path / "missing") == 0
