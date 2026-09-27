"""Fit the structured-prep stance map and role activity (#46).

Inputs are calibration scenario dirs made by ``ab_suite.py --manifest
tests/fixtures/calibration/calibration.json``: each has ``A`` (LLM prep) and
``B`` (structured prep). Entities are paired by name across the two
``simulation_config.json`` files:

* ``knots``: isotonic map from B's raw System One stance (``stance_raw``)
  to A's stance ``(sentiment_bias + 1) / 2``;
* ``activity_by_role``: mean A ``activity_level`` per B stakeholder role.

Only calibration seeds may be recorded; evaluation seeds 1-5 are refused.

Usage:
    uv run python scripts/fit_prep_calibration.py --dirs <cal>/<scenario> ... \\
        --out app/services/stance_calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from app.services.stance_calibration import fit_isotonic, refuse_eval_seeds  # noqa: E402


def agent_configs(work: Path) -> dict[str, dict[str, Any]]:
    """Entity name -> agent config from a prepared work dir."""

    prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
    path = Path(prepared["prepared_dir"]) / "simulation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return {row["entity_name"]: row for row in config.get("agent_configs", [])}


def fit(groups: list[tuple[dict, dict]], seeds: list[int]) -> dict[str, Any]:
    """``groups``: (A configs, B configs) per scenario, keyed by entity name."""

    refuse_eval_seeds(seeds)
    pairs: list[tuple[float, float]] = []
    activity: dict[str, list[float]] = {}
    for a_configs, b_configs in groups:
        for name, b in b_configs.items():
            a = a_configs.get(name)
            if a is None or b.get("stance_raw") is None:
                continue
            pairs.append((float(b["stance_raw"]), (float(a["sentiment_bias"]) + 1) / 2))
            role = b.get("stakeholder_role")
            if role:
                activity.setdefault(role, []).append(float(a["activity_level"]))
    knots = fit_isotonic(pairs, seeds)
    return {
        "seeds": list(seeds),
        "pairs": len(pairs),
        "knots": [[round(x, 4), round(y, 4)] for x, y in knots],
        "activity_by_role": {role: round(statistics.mean(v), 4) for role, v in sorted(activity.items())},
        "activity_by_role_n": {role: len(v) for role, v in sorted(activity.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", required=True, type=Path, help="calibration scenario dirs with A/ and B/")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    groups = [(agent_configs(d / "A"), agent_configs(d / "B")) for d in args.dirs]
    result = fit(groups, args.seeds)
    result["fitted_on"] = [str(d) for d in args.dirs]
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "fitted_on"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
