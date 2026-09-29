"""Fit weights on the active-hours readout so its pattern mix matches the
LLM prep (#53).

The structured prep asks System One which of ``ACTIVE_PATTERNS`` an entity
keeps and takes the most likely one. Over the calibration scenarios it picked
office hours far more often than the LLM prep does, so few agents were
online in the evening (activity-weighted evening coverage 0.18-0.64 against
0.50-0.79) and the last hours of a run were written by a small, one-sided
group: the stance curve swung where the LLM path's stayed flat.

For each calibration scenario (a ``<dir>`` with ``A/`` from the LLM prep and
``B/`` from the structured prep, as ``ab_suite.py`` writes them) this reads
the pattern probabilities of every B entity again, maps each A agent's hour
list to the nearest pattern (Jaccard), and scales the per-pattern weights
until the argmax of ``probability x weight`` picks each pattern about as
often as the A configs do. Only calibration scenarios may be used; the
weights go to ``app/services/hours_priors.json``.

Usage:
    uv run python scripts/fit_hours_priors.py --dirs <cal-out>/<scenario> ... \\
        --out app/services/hours_priors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))


def nearest_pattern(hours: list[int], patterns: dict[str, list[int]]) -> str:
    wanted = set(hours)

    def jaccard(key: str) -> float:
        have = set(patterns[key])
        return len(wanted & have) / len(wanted | have) if wanted | have else 0.0

    return max(patterns, key=jaccard)


def pick(probs: dict[str, float], weights: dict[str, float]) -> str:
    return max(probs, key=lambda k: probs[k] * weights.get(k, 1.0))


def fit_weights(
    readouts: list[dict[str, float]], target: dict[str, float], rounds: int = 200
) -> dict[str, float]:
    """Multiplicative updates toward the target mix of argmax picks."""

    keys = sorted(target)
    weights = {k: 1.0 for k in keys}
    best, best_gap = dict(weights), float("inf")
    for _ in range(rounds):
        counts = Counter(pick(p, weights) for p in readouts)
        share = {k: counts[k] / len(readouts) for k in keys}
        gap = sum(abs(share[k] - target[k]) for k in keys)
        if gap < best_gap:
            best, best_gap = dict(weights), gap
        for k in keys:
            weights[k] *= ((target[k] + 0.02) / (share[k] + 0.02)) ** 0.3
        norm = max(weights.values())
        weights = {k: w / norm for k, w in weights.items()}
    return {k: round(v, 4) for k, v in best.items()}


def read_b_patterns(work: Path) -> list[dict[str, float]]:
    """The hours readout of every entity of a structured-prep work dir."""

    import golden_pipeline as gp

    gp.force_local_config(work)
    from app.services.prep_structured import ACTIVE_PATTERNS, _ask
    from app.services.zep_entity_reader import ZepEntityReader
    from app.system_one.client import get_system_one_client
    from app.system_one.models import ChoiceQuestion

    prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
    ontology = json.loads((work / "ontology.json").read_text(encoding="utf-8"))
    types = [t["name"] for t in ontology.get("entity_types", [])]
    entities = ZepEntityReader().filter_defined_entities(prepared["graph_id"], types, enrich_with_edges=False).entities
    client = get_system_one_client()
    rows = []
    for entity in entities:
        entity_type = entity.get_entity_type() or "Unknown"
        state = f"實體：{entity.name}（類型：{entity_type}）\n摘要：{(entity.summary or '')[:400]}"
        answer = _ask(client, state, {"hours": ChoiceQuestion(
            instructions=f"「{entity.name}」通常什麼時段上網發言？",
            criteria={k: v[0] for k, v in ACTIVE_PATTERNS.items()},
        )})["hours"]
        rows.append(dict(answer.probabilities))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import ab_suite

    from app.services.prep_structured import ACTIVE_PATTERNS

    patterns = {k: v[1] for k, v in ACTIVE_PATTERNS.items()}
    calibration = {row["digest"] for row in ab_suite.load_manifest(ab_suite.CALIBRATION_MANIFEST)}
    target_counts: Counter[str] = Counter()
    readouts: list[dict[str, float]] = []
    for scenario in args.dirs:
        prepared = json.loads((scenario / "B" / "prepared.json").read_text(encoding="utf-8"))
        if prepared.get("fixture_digest") not in calibration:
            raise SystemExit(f"{scenario} is not a calibration scenario; hours priors are fitted on those only")
        config = json.loads((scenario / "A" / "prepared" / "simulation_config.json").read_text(encoding="utf-8"))
        target_counts.update(nearest_pattern(a["active_hours"], patterns) for a in config["agent_configs"])
        readouts += read_b_patterns(scenario / "B")
    total = sum(target_counts.values())
    target = {k: target_counts[k] / total for k in patterns}
    before = Counter(max(p, key=p.get) for p in readouts)
    weights = fit_weights(readouts, target)
    after = Counter(pick(p, weights) for p in readouts)
    payload: dict[str, Any] = {
        "weights": weights,
        "target_share": {k: round(v, 4) for k, v in target.items()},
        "argmax_share_before": {k: round(before[k] / len(readouts), 4) for k in patterns},
        "argmax_share_after": {k: round(after[k] / len(readouts), 4) for k in patterns},
        "entities": len(readouts),
        "scenarios": [p.name for p in args.dirs],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
