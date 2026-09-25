"""Hierarchical questioning.

A tree node is ``{"name": str, "question": <choice question dict>,
"children": {option_key: node}}``. ``children`` is optional; an option with
no child ends the path.
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
    probabilities: dict[str, float]


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
        if mode == "argmax":
            chosen = answer.choice
        else:
            chosen = sample_option(answer.probabilities, rng)
        path.append(TreeStep(node["name"], chosen, dict(answer.probabilities)))
        current_state = f"{current_state}\n{node['name']}: {chosen}"
        node = (node.get("children") or {}).get(chosen)
    return path
