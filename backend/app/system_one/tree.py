"""Hierarchical questioning.

A tree node is ``{"name": str, "question": <choice question dict>,
"children": {option_key: node}, "priors": {option_key: weight}}``.
``children`` is optional; an option with no child ends the path. ``priors``
is optional: the sampled distribution is the readout times the prior weight,
renormalised (a product of experts; options without a weight keep 1.0). It
corrects base rates the readout gets wrong, e.g. a small model that reads
"posting" as the most likely thing anyone does (#26).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from .models import ChoiceQuestion, SystemOneRequest


@dataclass(frozen=True)
class TreeStep:
    node: str
    choice: str
    probabilities: dict[str, float]  # what was sampled from (after priors)
    readout: dict[str, float] | None = None  # the raw readout when priors applied


def apply_priors(probabilities: dict[str, float], priors: dict[str, float] | None) -> dict[str, float]:
    """``p_i * w_i`` renormalised; unchanged without priors or if all weights vanish."""

    if not priors:
        return dict(probabilities)
    weighted = {k: p * float(priors.get(k, 1.0)) for k, p in probabilities.items()}
    total = sum(weighted.values())
    if total <= 0:
        return dict(probabilities)
    return {k: v / total for k, v in weighted.items()}


def sample_option(probabilities: dict[str, float], rng: random.Random) -> str:
    """Inverse-CDF sample in dict order, so the result depends only on rng."""

    draw = rng.random()
    cumulative = 0.0
    last = None
    for key, prob in probabilities.items():
        cumulative += prob
        last = key
        if draw < cumulative:
            return key
    return last  # float rounding left draw just above the total


def ask_tree(
    client,
    state: str,
    tree: dict[str, Any],
    rng: random.Random,
    *,
    mode: Literal["sample", "argmax"] = "sample",
    max_depth: int = 8,
) -> list[TreeStep]:
    path: list[TreeStep] = []
    node: dict[str, Any] | None = tree
    current_state = state
    while node is not None and len(path) < max_depth:
        question = ChoiceQuestion.model_validate(node["question"])
        answer = client.ask(
            SystemOneRequest(state=current_state, questions={node["name"]: question})
        ).answers[node["name"]]
        priors = node.get("priors")
        probabilities = apply_priors(answer.probabilities, priors)
        if mode == "argmax":
            chosen = max(probabilities, key=probabilities.get) if priors else answer.choice
        else:
            chosen = sample_option(probabilities, rng)
        path.append(
            TreeStep(node["name"], chosen, probabilities, dict(answer.probabilities) if priors else None)
        )
        current_state = f"{current_state}\n{node['name']}: {chosen}"
        node = (node.get("children") or {}).get(chosen)
    if node is not None:
        raise ValueError(f"decision tree is deeper than max_depth={max_depth}")
    return path
