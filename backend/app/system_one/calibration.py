"""Temperature calibration and calibration metrics.

Readout probabilities are a softmax over label logprobs, so re-tempering a
stored distribution is ``p_T ∝ p ** (1 / T)``; no second model call is needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def apply_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logs = [math.log(max(p, 1e-300)) / temperature for p in probs]
    peak = max(logs)
    exps = [math.exp(v - peak) for v in logs]
    total = sum(exps)
    return [e / total for e in exps]


def nll(dists: Sequence[Sequence[float]], labels: Sequence[int], temperature: float = 1.0) -> float:
    total = 0.0
    for probs, label in zip(dists, labels):
        tempered = apply_temperature(probs, temperature)
        total -= math.log(max(tempered[label], 1e-300))
    return total / max(len(labels), 1)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """Standard top-1 ECE with equal-width confidence bins."""

    if not confidences:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, ok in zip(confidences, correct):
        index = min(int(conf * bins), bins - 1)
        buckets[index].append((conf, ok))
    total = len(confidences)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        error += len(bucket) / total * abs(avg_conf - accuracy)
    return error


def top1(dists: Sequence[Sequence[float]], labels: Sequence[int], temperature: float = 1.0):
    """Return (confidences, correct flags) after tempering."""

    confidences, correct = [], []
    for probs, label in zip(dists, labels):
        tempered = apply_temperature(probs, temperature)
        best = max(range(len(tempered)), key=lambda i: tempered[i])
        confidences.append(tempered[best])
        correct.append(best == label)
    return confidences, correct


def fit_temperature(
    dists: Sequence[Sequence[float]],
    labels: Sequence[int],
    grid: Sequence[float] | None = None,
) -> float:
    """Grid-search the NLL-minimising temperature (log-spaced 0.1..10)."""

    if not labels:
        return 1.0
    candidates = grid or [round(10 ** (k / 40), 4) for k in range(-40, 41)]
    return min(candidates, key=lambda t: (nll(dists, labels, t), abs(math.log(t))))
