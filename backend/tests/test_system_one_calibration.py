import json
import math

from app.system_one.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    top1,
)
from app.system_one.client import load_temperatures


def test_apply_temperature_identity_and_flattening():
    probs = [0.7, 0.2, 0.1]
    assert all(math.isclose(a, b) for a, b in zip(apply_temperature(probs, 1.0), probs))
    flat = apply_temperature(probs, 4.0)
    assert math.isclose(sum(flat), 1.0)
    assert flat[0] < 0.7 and flat[2] > 0.1


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error([0.95] * 20, [True] * 19 + [False]) < 1e-9
    assert math.isclose(expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5), 0.4)


def test_fit_temperature_softens_overconfident_readout():
    # Always 0.99 confident but only 60% right -> T > 1.
    dists = [[0.99, 0.01]] * 6 + [[0.01, 0.99]] * 4
    labels = [0] * 10
    temperature = fit_temperature(dists, labels)
    assert temperature > 1.5
    conf_before, ok = top1(dists, labels)
    conf_after, _ = top1(dists, labels, temperature)
    assert expected_calibration_error(conf_after, ok) < expected_calibration_error(conf_before, ok)


def test_load_temperatures_reads_calibration_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperatures": {"choice": 1.7}}), encoding="utf-8")
    assert load_temperatures(path) == {"choice": 1.7}
    assert load_temperatures(tmp_path / "missing.json") == {}


def test_eval_fixture_has_at_least_200_labelled_items():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "system_one_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 200
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        if row["type"] == "choice":
            assert row["label"] in row["criteria"]
        elif row["type"] == "score":
            assert 0 <= row["label"] < len(row["criteria"])
        else:
            assert isinstance(row["label"], bool)
