import json
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
            "stance_by_persona": {"value": math.nan, "a_seed_pairs_mean": None, "passed": False},
            "stance_distribution": {"b_vs_a": 0.01, "a_seed_noise": 0.02, "passed": True},
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


def _run(decode, actions, curve, vram=8000, errors=0, posts=3, persona=None, levels=None):
    persona = persona or {n: i / 11 for i, n in enumerate("甲乙丙丁戊己庚辛壬癸子丑")}
    levels = levels or {"強烈反對": 1, "反對": 2, "中立": 3, "支持": 4, "強烈支持": 1}
    return {
        "stance_curve_windowed": curve[:2],
        "stance_by_persona": persona,
        "stance_levels": levels,
        "decode_per_round": decode,
        "round_latency_mean_s": 1.0,
        "llm_errors": errors,
        "vram_peak_mib": vram,
        "actions": actions,
        "content": {"posts": posts, "distinct_2": 0.5 if posts else 0.0, "entity_mention_rate": 0.5 if posts else None},
        "stance_curve": curve,
    }


def test_evaluate_groups_switch_and_exclusions():
    curve = [0.2, 0.4, 0.6, 0.8]
    per_run = {
        "A": {
            1: _run(150, {"like": 10, "post": 5}, curve),
            2: _run(160, {"like": 9, "post": 6}, [0.25, 0.4, 0.55, 0.8]),
            3: _run(10, {"post": 1}, [0.9, 0.1, 0.9, 0.1], errors=4),  # lost turns
        },
        "B": {1: _run(5, {"like": 10, "post": 5}, curve), 2: _run(0, {"like": 9, "post": 6}, curve, posts=0)},
    }
    gates, recommendation, summary, excluded = ab.evaluate_groups(per_run)
    assert excluded == ["A_seed3"]
    assert summary["A"]["clean_runs"] == 2
    assert summary["B"]["entity_mention_rate"] == 0.5  # the run without posts is skipped
    assert gates["decode_ratio"]["passed"] and gates["vram"]["passed"]
    assert recommendation == "switch"


def test_evaluate_groups_unmeasured_vram_and_zero_decode_do_not_pass():
    curve = [0.2, 0.4, 0.6, 0.8]
    per_run = {
        "A": {1: _run(0, {"like": 1}, curve), 2: _run(0, {"like": 1}, curve)},
        "B": {1: _run(0, {"like": 1}, curve, vram=None)},
    }
    gates, recommendation, _, _ = ab.evaluate_groups(per_run)
    assert not gates["decode_ratio"]["passed"]
    assert gates["vram"]["peak_mib"] is None and not gates["vram"]["passed"]
    assert recommendation == "keep_llm"


def test_reused_runs_must_match_and_be_clean(tmp_path):
    good = {"exit_code": 0, "decision_backend": "llm", "seed": 1, "max_rounds": 24}
    ab.check_reused(tmp_path, good, "llm", 1, 24)
    for bad in ({**good, "exit_code": 1}, {**good, "max_rounds": 12}, {**good, "seed": 2}):
        with pytest.raises(SystemExit):
            ab.check_reused(tmp_path, bad, "llm", 1, 24)


def test_planned_rounds_follow_the_sim_config(tmp_path):
    import json

    (tmp_path / "sim").mkdir()
    (tmp_path / "sim" / "simulation_config.json").write_text(
        json.dumps({"time_config": {"total_simulation_hours": 6, "minutes_per_round": 30}}), encoding="utf-8"
    )
    assert ab.planned_rounds(tmp_path, 24) == 12
    assert ab.planned_rounds(tmp_path, 8) == 8


def test_action_counts_skip_do_nothing_and_round_zero(tmp_path):
    import json

    (tmp_path / "sim" / "twitter").mkdir(parents=True)
    rows = [
        {"round": 0, "agent_id": 1, "action_type": "CREATE_POST"},  # scripted initial post
        {"round": 0, "event_type": "round_start"},
        {"round": 1, "agent_id": 1, "action_type": "LIKE_POST"},
        {"round": 1, "agent_id": 2, "action_type": "like_post"},
        {"round": 2, "agent_id": 2, "action_type": "DO_NOTHING"},
    ]
    (tmp_path / "sim" / "twitter" / "actions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    assert dict(ab.action_counts(tmp_path)) == {"twitter:LIKE_POST": 2}


def test_stance_measures_and_persona_gates():
    cache = {"a": {"unit": 0.0, "levels": [1, 0, 0, 0, 0]}, "b": {"unit": 1.0, "levels": [0, 0, 0, 0, 1]},
             "c": {"unit": 0.5, "levels": [0, 0, 1, 0, 0]}}
    rows = [{"round": 1, "agent": "甲", "text": "a"}, {"round": 2, "agent": "乙", "text": "b"},
            {"round": 5, "agent": "甲", "text": "c"}]
    m = ab.stance_measures(rows, cache, 8)
    assert m["stance_curve"][:5] == [0.0, 1.0, None, None, 0.5]
    assert m["stance_curve_windowed"] == [0.5, 0.5]
    assert m["stance_by_persona"] == {"甲": 0.25, "乙": 1.0}
    assert m["stance_levels"]["強烈反對"] == 1 and m["stance_levels"]["中立"] == 1

    curve = [0.2, 0.4, 0.6, 0.8]
    flipped = {n: 1 - i / 11 for i, n in enumerate("甲乙丙丁戊己庚辛壬癸子丑")}
    per_run = {"A": {1: _run(150, {"like": 1}, curve), 2: _run(150, {"like": 1}, curve)},
               "B": {1: _run(5, {"like": 1}, curve, persona=flipped)}}
    gates, _, summary, _ = ab.evaluate_groups(per_run)
    assert gates["stance_by_persona"]["value"] < 0
    assert gates["stance_by_persona"]["status"] == "fail"
    assert gates["stance_by_persona"]["passed"] is False
    assert gates["stance_distribution"]["status"] == "pass"
    assert summary["stance_curve_report"]["per_round"]["a_seed_pairs_mean"] == pytest.approx(1.0)


def test_persona_gate_needs_enough_shared_personas():
    names = "甲乙丙丁戊己庚辛壬癸子丑"
    a = {n: i / 11 for i, n in enumerate(names)}
    assert ab.enough_common_personas(a, a) == (12, True)
    few = {k: a[k] for k in "甲乙丙"}
    assert ab.enough_common_personas(a, few) == (3, False)
    curve = [0.2, 0.4, 0.6, 0.8]
    per_run = {"A": {1: _run(150, {"like": 1}, curve), 2: _run(150, {"like": 1}, curve)},
               "B": {1: _run(5, {"like": 1}, curve, persona=few)}}
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["stance_by_persona"]["common_personas"] == 3
    assert gates["stance_by_persona"]["status"] == "undecidable"
    assert gates["stance_by_persona"]["passed"] is None


def _persona(n=12, flip=False):
    names = list("甲乙丙丁戊己庚辛壬癸子丑")
    if n > len(names):
        names = [f"人{i:02d}" for i in range(n)]
    span = max(n - 1, 1)
    return {name: (1 - i / span if flip else i / span) for i, name in enumerate(names[:n])}


def test_bootstrap_ci_percentiles():
    values = [float(i) for i in range(1, 1001)]
    assert ab.bootstrap_ci(values) == (26.0, 975.0)


def test_bootstrap_is_reproducible():
    curve = [0.2, 0.4, 0.6, 0.8]
    persona = _persona()
    per_run = {
        "A": {i: _run(150, {"like": 10, "post": 5}, curve, persona=persona) for i in (1, 2, 3)},
        "B": {i: _run(5, {"like": 9, "post": 6}, curve, persona=persona) for i in (1, 2)},
    }
    first = ab.evaluate_groups(per_run)
    second = ab.evaluate_groups(per_run)
    assert first[0] == second[0]


def test_persona_relative_threshold_passes():
    curve = [0.2, 0.4, 0.6, 0.8]
    persona = _persona()
    per_run = {
        "A": {i: _run(150, {"like": 1}, curve, persona=persona) for i in (1, 2, 3)},
        "B": {i: _run(5, {"like": 1}, curve, persona=persona) for i in (1, 2, 3)},
    }
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["stance_by_persona"]["ci"][0] == 1.0
    assert gates["stance_by_persona"]["threshold"] == pytest.approx(0.85)
    assert gates["stance_by_persona"]["status"] == "pass"


def test_persona_relative_threshold_fails():
    curve = [0.2, 0.4, 0.6, 0.8]
    per_run = {
        "A": {i: _run(150, {"like": 1}, curve, persona=_persona()) for i in (1, 2, 3)},
        "B": {i: _run(5, {"like": 1}, curve, persona=_persona(flip=True)) for i in (1, 2, 3)},
    }
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["stance_by_persona"]["value"] == pytest.approx(-1.0)
    assert gates["stance_by_persona"]["status"] == "fail"
    assert gates["stance_by_persona"]["passed"] is False


def test_persona_undecidable_below_12():
    curve = [0.2, 0.4, 0.6, 0.8]
    full = _persona(12)
    shared = {name: full[name] for name in list(full)[:11]}
    actions = {"like": 10, "post": 5}
    per_run = {
        "A": {i: _run(150, actions, curve, persona=full) for i in (1, 2, 3)},
        "B": {i: _run(5, actions, curve, persona=shared) for i in (1, 2, 3)},
    }
    gates, recommendation, _, _ = ab.evaluate_groups(per_run)
    assert gates["stance_by_persona"]["common_personas"] == 11
    assert gates["stance_by_persona"]["status"] == "undecidable"
    assert gates["stance_by_persona"]["passed"] is None
    assert recommendation == "switch"


def test_persona_share_rule_still_fails():
    curve = [0.2, 0.4, 0.6, 0.8]
    full = _persona(20)
    shared = {name: full[name] for name in list(full)[:12]}
    per_run = {
        "A": {i: _run(150, {"like": 1}, curve, persona=full) for i in (1, 2, 3)},
        "B": {i: _run(5, {"like": 1}, curve, persona=shared) for i in (1, 2, 3)},
    }
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["stance_by_persona"]["status"] == "fail"


def test_action_js_one_sided_pass():
    curve = [0.2, 0.4, 0.6, 0.8]
    a_actions = ({"like": 10, "post": 5}, {"like": 9, "post": 6}, {"like": 10, "post": 6})
    per_run = {
        "A": {i + 1: _run(150, actions, curve) for i, actions in enumerate(a_actions)},
        "B": {i + 1: _run(5, actions, curve) for i, actions in enumerate(a_actions)},
    }
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["action_js"]["p_value"] >= 0.05
    assert gates["action_js"]["status"] == "pass"


def test_action_js_one_sided_fail():
    curve = [0.2, 0.4, 0.6, 0.8]
    a_actions = ({"like": 10, "post": 5}, {"like": 9, "post": 6}, {"like": 10, "post": 6})
    per_run = {
        "A": {i + 1: _run(150, actions, curve) for i, actions in enumerate(a_actions)},
        "B": {i + 1: _run(5, {"post": 15}, curve) for i in range(3)},
    }
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["action_js"]["p_value"] == 0.0
    assert gates["action_js"]["status"] == "fail"


def test_stance_distribution_identical_mix_passes():
    curve = [0.2, 0.4, 0.6, 0.8]
    levels = {"強烈反對": 1, "反對": 2, "中立": 3, "支持": 4, "強烈支持": 1}
    per_run = {
        "A": {i: _run(150, {"like": 1}, curve, levels=levels) for i in (1, 2, 3)},
        "B": {i: _run(5, {"like": 1}, curve, levels=levels) for i in (1, 2, 3)},
    }
    gates, _, _, _ = ab.evaluate_groups(per_run)
    assert gates["stance_distribution"]["p_value"] == 1.0
    assert gates["stance_distribution"]["status"] == "pass"


def test_within_noise_skips_same_seed_pairs():
    import random
    import statistics

    a_runs = [{"actions": {"like": 10, "post": 1}}, {"actions": {"post": 10, "like": 1}}]

    def mean_js(pairs):
        return statistics.mean(ab.js_divergence(x["actions"], y["actions"]) for x, y in pairs)

    values = ab.bootstrap_within(a_runs, mean_js, 200, random.Random(0))
    assert values
    assert all(value > 0 for value in values)
    assert len(values) <= 200


def test_stance_question_from_fixture(tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    question = "這則貼文對「X」的立場是什麼？"
    (fx / "stance_question.txt").write_text(question, encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ab.stance_question_for({"fixture": str(fx)}) == question
    assert ab.stance_question_for({"fixture": str(empty)}) == ab.STANCE_QUESTION


def test_stance_question_survives_a_moved_fixture(tmp_path):
    import scripts.ab_suite as suite

    qingpu = next(row for row in suite.load_manifest() if row["name"] == "qingpu_clinic")
    expected = (suite.fixture_dir(qingpu) / "stance_question.txt").read_text(encoding="utf-8").strip()
    moved = {"fixture": str(tmp_path / "gone" / "qingpu_clinic"), "fixture_digest": qingpu["digest"]}
    assert ab.stance_question_for(moved) == expected
    with pytest.raises(SystemExit):
        ab.stance_question_for({"fixture": str(tmp_path / "gone"), "fixture_digest": "0" * 64})
    assert ab.stance_question_for({}) == ab.STANCE_QUESTION


def test_independent_scorer_client():
    from app.system_one.backends import LocalReadoutBackend

    client = ab.scorer_client("http://127.0.0.1:8002/v1", "phi-4-mini", "plain")
    assert isinstance(client.backend, LocalReadoutBackend)
    assert (client.backend.base_url, client.backend.model, client.backend.prompt_format) == (
        "http://127.0.0.1:8002/v1", "phi-4-mini", "plain")
    assert ab.scorer_client(None, None, "chatml") is None


def test_scorer_by_pool_name(monkeypatch):
    pool = [
        {"name": "qwen", "base_url": "http://127.0.0.1:8000/v1", "model": "qwen3.5-4b"},
        {"name": "phi", "base_url": "http://127.0.0.1:8002/v1", "model": "phi-4-mini", "prompt_format": "plain"},
    ]
    monkeypatch.setenv("MODEL_POOL", json.dumps(pool))
    assert ab.scorer_endpoint("phi") == ("http://127.0.0.1:8002/v1", "phi-4-mini", "plain")
    with pytest.raises(SystemExit, match="gemma"):
        ab.scorer_endpoint("gemma")
    monkeypatch.delenv("MODEL_POOL")
    with pytest.raises(SystemExit, match="MODEL_POOL"):
        ab.scorer_endpoint("phi")
