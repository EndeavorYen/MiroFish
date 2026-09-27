import json

import pytest

from app.system_one.models import ChoiceAnswer, ScoreAnswer, SystemOneResponse


class Recorder:
    """First option of every choice, score 0 (strongly opposed / very low)."""

    def ask(self, request):
        answers = {}
        for name, question in request.questions.items():
            if hasattr(question, "criteria") and isinstance(question.criteria, dict):
                key = next(iter(question.criteria))
                answers[name] = ChoiceAnswer(choice=key, probabilities={key: 1.0}, confidence=1.0)
            else:
                answers[name] = ScoreAnswer(score=0.0, probabilities={}, confidence=1.0)
        return SystemOneResponse(answers=answers)


def _config(monkeypatch, calibration=None, tmp_path=None):
    from app.services.prep_structured import structured_agent_config

    if calibration is None:
        monkeypatch.setenv("STANCE_CALIBRATION", str(tmp_path / "missing.json"))
    else:
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(calibration), encoding="utf-8")
        monkeypatch.setenv("STANCE_CALIBRATION", str(path))
    return structured_agent_config(
        Recorder(), "工會", "LaborUnion", "司機工會", event="空中計程車試點", context="- 工會要求轉崗基金"
    )


def test_agent_config_records_role_and_raw_stance(monkeypatch, tmp_path):
    cfg = _config(monkeypatch, tmp_path=tmp_path)
    assert cfg["stakeholder_role"] == "beneficiary"
    assert cfg["stance_raw"] == 0.0
    # No calibration file: activity keeps the readout formula (floor at score 0).
    from app.services.prep_structured import ACTIVITY_FLOOR

    assert cfg["activity_level"] == pytest.approx(ACTIVITY_FLOOR)


def test_activity_follows_the_fitted_role_mean(monkeypatch, tmp_path):
    cfg = _config(
        monkeypatch,
        {"seeds": [11, 12, 13], "knots": [], "activity_by_role": {"beneficiary": 0.85, "regulator": 0.2}},
        tmp_path,
    )
    assert cfg["activity_level"] == pytest.approx(0.85)
    assert cfg["posts_per_hour"] == pytest.approx(0.1 + 0.9 * 0.85)


def test_role_activity_refuses_eval_seeds(tmp_path):
    from app.services.stance_calibration import load_activity_by_role

    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"seeds": [3, 11], "knots": [], "activity_by_role": {"harmed": 0.9}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_activity_by_role(path)


def test_fit_pairs_b_raw_with_a_llm_prep():
    from scripts.fit_prep_calibration import fit

    a = {"x": {"sentiment_bias": -0.8, "activity_level": 0.9},
         "y": {"sentiment_bias": 0.6, "activity_level": 0.2},
         "z": {"sentiment_bias": 0.0, "activity_level": 0.5}}
    b = {"x": {"stance_raw": 0.2, "stakeholder_role": "harmed"},
         "y": {"stance_raw": 0.9, "stakeholder_role": "regulator"},
         "z": {"stance_raw": 0.5, "stakeholder_role": "harmed"},
         "only_b": {"stance_raw": 0.1, "stakeholder_role": "media"}}
    result = fit([(a, b)], seeds=[11, 12, 13])
    assert result["pairs"] == 3
    assert result["activity_by_role"] == {"harmed": pytest.approx(0.7), "regulator": pytest.approx(0.2)}
    assert [t for _, t in result["knots"]] == pytest.approx([0.1, 0.5, 0.8])
    with pytest.raises(ValueError):
        fit([(a, b)], seeds=[1, 11])
