"""Role-conditioned stance mapped onto 0..1 with a calibration fit (#46).

The fit may use calibration seeds only. Evaluation seeds 1–5 are refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

EVAL_SEEDS = frozenset({1, 2, 3, 4, 5})
DEFAULT_PATH = Path(__file__).with_name("stance_calibration.json")

ROLES = {
    "beneficiary": "受益者，事件對其有利",
    "harmed": "受損者，生計或權益受影響",
    "regulator": "監管或決策者",
    "observer": "評論或觀察者",
    "media": "媒體",
}


def role_question(name: str):
    from ..system_one.models import ChoiceQuestion

    return ChoiceQuestion(
        instructions=f"在這個事件裡，「{name}」最接近哪一種角色？",
        criteria=ROLES,
    )


def refuse_eval_seeds(seeds: list[int]) -> None:
    overlap = sorted(set(seeds) & EVAL_SEEDS)
    if overlap:
        raise ValueError(f"calibration must not use evaluation seeds {overlap}")


def fit_isotonic(
    pairs: list[tuple[float, float]],
    seeds: list[int],
) -> list[tuple[float, float]]:
    """Pool-adjacent-violators map from raw scores to targets. Knots stay in 0..1."""

    refuse_eval_seeds(seeds)
    if not pairs:
        raise ValueError("calibration needs at least one pair")
    ordered = sorted((float(raw), float(target)) for raw, target in pairs)
    blocks: list[list[tuple[float, float]]] = [[pair] for pair in ordered]

    def _mean(block: list[tuple[float, float]]) -> float:
        return sum(target for _, target in block) / len(block)

    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(blocks) - 1:
            if _mean(blocks[index]) > _mean(blocks[index + 1]):
                blocks[index] = blocks[index] + blocks[index + 1]
                del blocks[index + 1]
                changed = True
            else:
                index += 1
    knots: list[tuple[float, float]] = []
    for block in blocks:
        raw = sum(item[0] for item in block) / len(block)
        target = _mean(block)
        knots.append((_clamp(raw), _clamp(target)))
    return knots


def apply_calibration(raw: float, knots: list[tuple[float, float]] | None) -> float:
    """Piecewise-linear map. Missing knots leave the score unchanged, still in 0..1."""

    value = _clamp(raw)
    if not knots:
        return value
    points = sorted(knots)
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return _clamp(y1)
            weight = (value - x0) / (x1 - x0)
            return _clamp(y0 + weight * (y1 - y0))
    return value


def _load(path: Path | None = None) -> dict | None:
    setting = os.environ.get("STANCE_CALIBRATION", "").strip()
    chosen = path if path is not None else (Path(setting) if setting else DEFAULT_PATH)
    if not chosen.exists():
        return None
    data = json.loads(chosen.read_text(encoding="utf-8"))
    refuse_eval_seeds([int(seed) for seed in data.get("seeds", [])])
    return data


def load_knots(path: Path | None = None) -> list[tuple[float, float]] | None:
    data = _load(path)
    if data is None:
        return None
    return [(float(raw), float(target)) for raw, target in data.get("knots") or []]


def load_activity_by_role(path: Path | None = None) -> dict[str, float] | None:
    """Mean LLM-prep activity per stakeholder role, fitted on calibration
    scenarios. The activity readout alone put almost every entity near 0.35,
    while the LLM prep makes aggrieved groups the most active and officials
    the least (#46)."""

    data = _load(path)
    if data is None or not data.get("activity_by_role"):
        return None
    return {role: _clamp(value) for role, value in data["activity_by_role"].items()}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
