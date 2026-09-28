import pytest

import scripts.scan_gates as sg

LEVELS = ["強烈反對", "反對", "中立", "支持", "強烈支持"]


def _run(personas, levels, curve):
    return {"stance_by_persona": personas, "stance_levels": dict(zip(LEVELS, levels)),
            "stance_curve_windowed": curve}


def _report(a_runs, b_runs, decode=0.05, vram=8000):
    return {
        "groups": {"A": {str(i): r for i, r in enumerate(a_runs, 1)}, "B": {str(i): r for i, r in enumerate(b_runs, 1)}},
        "gates": {
            "decode_ratio": {"value": decode, "passed": decode <= 0.10},
            "vram": {"peak_mib": vram, "passed": vram <= 10240},
            "action_js": {"b_vs_a": 0.04},
            "stance_distribution": {"b_vs_a": 0.07},
        },
    }


NAMES = [f"p{i}" for i in range(12)]


def test_spearman_ranks():
    assert sg.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert sg.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert sg.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_camp_is_the_largest_group():
    assert sg.camp({"a": 0.1, "b": 0.2, "c": 0.9}) == "oppose"
    assert sg.camp({"a": 0.5, "b": 0.45, "c": 0.9}) == "neutral"


def test_scan_gates_pass_when_the_direction_matches():
    a = [_run({n: i / 11 for i, n in enumerate(NAMES)}, [1, 1, 2, 3, 3], [0.4, 0.5, 0.6])]
    b = [_run({n: 0.05 + 0.9 * i / 11 for i, n in enumerate(NAMES)}, [0, 2, 2, 4, 2], [0.45, 0.5, 0.62])]
    gates = sg.evaluate(_report(a, b))
    assert gates["persona_rank"]["status"] == "pass"
    assert gates["camp"]["status"] == "pass"
    assert gates["lean"]["status"] == "pass"
    assert gates["trend"]["status"] == "pass"
    assert sg.scenario_status(gates) == "pass"
    assert gates["report_only"]["stance_distribution_js"] == 0.07


def test_scan_gates_fail_on_reversed_direction_and_cost():
    a = [_run({n: i / 11 for i, n in enumerate(NAMES)}, [0, 0, 1, 4, 5], [0.3, 0.5, 0.8])]
    b = [_run({n: 1 - i / 11 for i, n in enumerate(NAMES)}, [5, 4, 1, 0, 0], [0.8, 0.5, 0.3])]
    gates = sg.evaluate(_report(a, b, decode=0.2))
    assert gates["persona_rank"]["status"] == "fail"
    assert gates["lean"]["status"] == "fail"
    assert gates["trend"]["status"] == "fail"
    assert gates["cost"]["status"] == "fail"
    assert sg.scenario_status(gates) == "fail"


def test_few_shared_personas_are_undecidable():
    a = [_run({n: i / 4 for i, n in enumerate(NAMES[:5])}, [1, 1, 1, 1, 1], [0.5, 0.5])]
    gates = sg.evaluate(_report(a, a))
    assert gates["persona_rank"]["status"] == "undecidable"


def test_suite_verdict():
    assert sg.suite_verdict(["pass"] * 5 + ["fail"]) == "scan_ready"
    assert sg.suite_verdict(["pass"] * 4 + ["fail"] * 2) == "not_ready"
