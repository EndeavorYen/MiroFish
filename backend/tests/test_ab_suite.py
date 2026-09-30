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


def test_digest_ignores_checkout_line_endings(tmp_path):
    import scripts.golden_pipeline as gp

    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for directory, newline in ((lf, b"\n"), (crlf, b"\r\n")):
        directory.mkdir()
        for name in gp.SCENARIO_FILES:
            (directory / name).write_bytes(newline.join([b"line one", b"line two", b""]))
    assert gp.scenario_digest(lf) == gp.scenario_digest(crlf)


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


def test_screen_rows_prints_three_decisions(tmp_path, capsys):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({
        "rows": [
            {"name": "target", "higher_is_better": True, "baseline": [0.1, 0.1, 0.1], "new": [0.4, 0.4, 0.4]},
            {"name": "worse", "higher_is_better": True, "baseline": [0.4, 0.4, 0.4], "new": [0.1, 0.1, 0.1]},
            {"name": "flat", "higher_is_better": True, "baseline": [0.2, 0.2, 0.2], "new": [0.2, 0.2, 0.2]},
        ]
    }), encoding="utf-8")
    assert suite.main(["--screen-rows", str(path)]) == 0
    first = capsys.readouterr().out
    assert suite.main(["--screen-rows", str(path)]) == 0
    second = capsys.readouterr().out
    assert first == second
    assert first.splitlines() == ["target improve", "worse worsen", "flat inconclusive"]


def test_calibration_refuses_eval_seeds_and_stays_in_unit_interval(tmp_path, monkeypatch):
    from app.services.prep_structured import structured_agent_config
    from app.services.stance_calibration import apply_calibration, fit_isotonic
    from app.system_one.models import ChoiceAnswer, ScoreAnswer, SystemOneResponse

    with pytest.raises(ValueError):
        fit_isotonic([(0.2, 0.1), (0.8, 0.9)], [1, 2, 11])
    knots = fit_isotonic([(0.0, 0.2), (0.5, 0.5), (1.0, 0.8)], [11, 12, 13])
    assert all(0.0 <= raw <= 1.0 and 0.0 <= target <= 1.0 for raw, target in knots)
    mapped = apply_calibration(0.0, knots)
    assert 0.0 <= mapped <= 1.0
    path = tmp_path / "stance_calibration.json"
    path.write_text(json.dumps({"seeds": [11, 12, 13], "knots": knots}), encoding="utf-8")
    monkeypatch.setenv("STANCE_CALIBRATION", str(path))

    class Recorder:
        def ask(self, request):
            answers = {}
            for name, question in request.questions.items():
                if hasattr(question, "criteria") and isinstance(question.criteria, dict):
                    key = next(iter(question.criteria))
                    answers[name] = ChoiceAnswer(choice=key, probabilities={key: 1.0}, confidence=1.0)
                else:
                    answers[name] = ScoreAnswer(score=0.0, probabilities={}, confidence=1.0)
            return SystemOneResponse(answers=answers)

    cfg = structured_agent_config(
        Recorder(), "工會", "LaborUnion", "司機工會", event="空中計程車試點", context="- 工會要求轉崗基金"
    )
    stance = (cfg["sentiment_bias"] + 1) / 2
    assert stance == pytest.approx(mapped)
    assert 0.0 <= stance <= 1.0


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


def test_calibration_library_is_disjoint_from_the_suite():
    import scripts.golden_pipeline as gp

    calibration = suite.load_manifest(suite.CALIBRATION_MANIFEST)
    evaluation = suite.load_manifest()
    assert len(calibration) >= 6
    assert {row["language"] for row in calibration} >= {"zh", "en"}
    eval_digests = {row["digest"] for row in evaluation}
    for row in calibration:
        directory = suite.fixture_dir(row)
        for name in ("news_seed.txt", "simulation_requirement.txt", "stance_question.txt", "README.md"):
            assert (directory / name).is_file(), f"{directory} missing {name}"
        assert gp.scenario_digest(directory) == row["digest"]
        assert row["digest"] not in eval_digests


def test_calibration_manifest_refuses_eval_seeds(tmp_path):
    with pytest.raises(SystemExit):
        suite.main(["--manifest", str(suite.CALIBRATION_MANIFEST), "--out", str(tmp_path), "--seeds", "11", "3"])
    args = suite.parse_args(["--manifest", str(suite.CALIBRATION_MANIFEST), "--out", "x"])
    assert args.seeds == [11, 12, 13]


def test_suite_tasks_span_every_scenario_group_and_seed(tmp_path):
    from scripts import ab_suite as suite

    tasks = suite.suite_tasks(tmp_path, ["s1", "s2"], [1, 2], 24)
    assert len(tasks) == 8
    assert (tmp_path / "s2" / "A", tmp_path / "s2" / "runs" / "A_seed2", "llm", 2, 24, None) in tasks
    assert (tmp_path / "s1" / "B", tmp_path / "s1" / "runs" / "B_seed1", "system_one", 1, 24, "tiered") in tasks
