import json

import pytest

import scripts.ab_suite as suite


def test_manifest_library():
    import scripts.golden_pipeline as gp

    rows = suite.load_manifest()
    assert len(rows) >= 6
    domains = set()
    structures = set()
    languages = set()
    big = 0
    for row in rows:
        directory = suite.fixture_dir(row)
        for name in ("news_seed.txt", "simulation_requirement.txt", "stance_question.txt", "README.md"):
            assert (directory / name).is_file(), f"{directory} missing {name}"
        assert gp.scenario_digest(directory) == row["digest"]
        domains.add(row["domain"])
        structures.add(row["structure"])
        languages.add(row["language"])
        assert row["expected_personas"] >= 12
        if row["expected_personas"] >= 25:
            big += 1
    assert domains >= {"交通", "醫療", "教育", "能源", "公共衛生", "消費爭議"}
    assert "en" in languages
    assert structures >= {"一面倒", "兩極對立", "多數中立"}
    assert big >= 2


def test_manifest_digest_mismatch_exits(tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "news_seed.txt").write_text("seed", encoding="utf-8")
    (fx / "simulation_requirement.txt").write_text("req", encoding="utf-8")
    entry = {"name": "tmp", "path": str(fx), "digest": "0" * 64}
    with pytest.raises(SystemExit):
        suite.verify_digest(entry)


def test_ensure_prepared_reuses_and_guards(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    fx = tmp_path / "fx"
    fx.mkdir()
    digest = "abc"
    (work / "prepared.json").write_text(json.dumps({"fixture_digest": digest}), encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(suite.subprocess, "run", boom)
    prepared = suite.ensure_prepared(work, fx, "llm", digest)
    assert prepared["fixture_digest"] == digest
    with pytest.raises(SystemExit):
        suite.ensure_prepared(work, fx, "llm", "x")


def test_suite_recommendation_thresholds():
    assert suite.suite_recommendation(["pass"] * 5 + ["fail"]) == "switch"
    assert suite.suite_recommendation(["pass"] * 3 + ["fail"] * 3) == "conditional"
    assert suite.suite_recommendation(["pass"] * 2 + ["fail"] * 4) == "keep_llm"


def test_scenario_status_counts_undecidable_as_pass():
    undecided = {
        "decode_ratio": {"status": "pass"},
        "action_js": {"status": "pass"},
        "stance_by_persona": {"status": "undecidable"},
        "stance_distribution": {"status": "pass"},
        "vram": {"status": "pass"},
    }
    failed = {**undecided, "action_js": {"status": "fail"}}
    assert suite.scenario_status(undecided) == "pass"
    assert suite.scenario_status(failed) == "fail"


def test_quick_mode_args():
    args = suite.parse_args(["--out", "x", "--quick"])
    assert args.seeds == [1, 2, 3]
    assert args.rounds == 12


def test_render_suite_has_ci_and_summary():
    rows = [
        {"name": "one", "status": "pass", "b_vs_a_ci": [0.01, 0.03]},
        {"name": "two", "status": "fail", "b_vs_a_ci": [0.01, 0.03]},
    ]
    text = suite.render_suite(rows, "conditional", 3600)
    assert text.count("[0.010, 0.030]") >= 2
    assert "1/2" in text
    assert "conditional" in text
