"""Logit readout: turn first-token logprobs into System One answers.

One prefill, no decode. Options are labelled with single tokens (``A..Z`` for
``choice``, ``0..N-1`` for ``score``, ``Yes``/``No`` for ``noul``). The softmax
runs only over the option labels. Readout probabilities are not calibrated;
``calibration.json`` holds per-type temperatures fitted on the eval set.
"""

from __future__ import annotations

import math
import string
from typing import Literal

from .models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

MAX_CHOICE_OPTIONS = 26

PromptFormat = Literal["chatml", "plain"]

SYSTEM_PROMPT = (
    "You are a fast judgment model. Read the state, then answer the question "
    "with exactly one option label and nothing else."
)


class TooManyOptionsError(ValueError):
    """A choice question has more options than single-token labels allow."""


def choice_labels(question: ChoiceQuestion) -> list[str]:
    count = len(question.criteria)
    if count > MAX_CHOICE_OPTIONS:
        raise TooManyOptionsError(
            f"choice has {count} options; at most {MAX_CHOICE_OPTIONS} are "
            "supported. Split it into a hierarchy (ask_tree)."
        )
    return list(string.ascii_uppercase[:count])


def score_labels(question: ScoreQuestion) -> list[str]:
    # 0-based like Jev's score; "10" would be two tokens for Qwen.
    return [str(i) for i in range(len(question.criteria))]


NOUL_LABELS = ["Yes", "No"]


def labels_for(question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return choice_labels(question)
    if isinstance(question, ScoreQuestion):
        return score_labels(question)
    if isinstance(question, NoulQuestion):
        return list(NOUL_LABELS)
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _question_block(question) -> str:
    labels = labels_for(question)
    lines = [f"Question: {question.instructions}"]
    if isinstance(question, ChoiceQuestion):
        lines.append("Options:")
        for label, (key, description) in zip(labels, question.criteria.items()):
            text = f"{key}: {description}" if description and description != key else key
            lines.append(f"{label}. {text}")
        lines.append("Reply with the option letter only.")
    elif isinstance(question, ScoreQuestion):
        lines.append("Levels (low to high):")
        for label, description in zip(labels, question.criteria):
            lines.append(f"{label}. {description}")
        lines.append("Reply with the level number only.")
    else:
        lines.append("Reply with Yes or No only.")
    return "\n".join(lines)


def build_prompt(state: str, question, prompt_format: PromptFormat = "chatml") -> str:
    """State first, question last, so questions on one state share a prefix."""

    body = f"State:\n{state}\n\n{_question_block(question)}"
    if prompt_format == "plain":
        return f"{SYSTEM_PROMPT}\n\n{body}\nAnswer:"
    # Qwen ChatML with an empty think block: the next token is the label.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _label_logprobs(labels: list[str], top_logprobs: dict[str, float]) -> dict[str, float]:
    best: dict[str, float] = {}
    for token, logprob in top_logprobs.items():
        key = str(token).strip()
        if key in labels and (key not in best or logprob > best[key]):
            best[key] = float(logprob)
    return best


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def label_distribution(
    labels: list[str],
    top_logprobs: dict[str, float],
    temperature: float = 1.0,
) -> tuple[list[float], float, list[int]]:
    """Return (probabilities per label, coverage, indexes of missing labels).

    A label absent from top-k gets an upper-bound floor: it cannot be more
    likely than the smallest top-k entry, nor than the probability mass the
    top-k list leaves over. ``coverage`` is the raw probability mass of the
    labels that were present. With no label present the distribution is
    uniform.
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    found = _label_logprobs(labels, top_logprobs)
    missing = [i for i, label in enumerate(labels) if label not in found]
    coverage = sum(math.exp(lp) for lp in found.values())
    if not found:
        return [1.0 / len(labels)] * len(labels), 0.0, missing
    leftover = 1.0 - sum(math.exp(float(v)) for v in top_logprobs.values())
    floor = min(
        min(float(v) for v in top_logprobs.values()),
        math.log(max(leftover, 1e-12)),
    )
    logits = [found.get(label, floor) / temperature for label in labels]
    return _softmax(logits), min(coverage, 1.0), missing


def readout_answer(question, top_logprobs: dict[str, float], temperature: float = 1.0):
    labels = labels_for(question)
    probs, coverage, missing_idx = label_distribution(labels, top_logprobs, temperature)

    if isinstance(question, ChoiceQuestion):
        keys = list(question.criteria)
        probabilities = dict(zip(keys, probs))
        best = max(range(len(keys)), key=lambda i: probs[i])
        return ChoiceAnswer(
            choice=keys[best],
            probabilities=probabilities,
            confidence=probs[best],
            coverage=coverage,
            missing=[keys[i] for i in missing_idx],
        )
    if isinstance(question, ScoreQuestion):
        levels = list(question.criteria)
        return ScoreAnswer(
            score=sum(i * p for i, p in enumerate(probs)),
            probabilities=dict(zip(levels, probs)),
            confidence=max(probs),
            coverage=coverage,
            missing=[levels[i] for i in missing_idx],
        )
    return NoulAnswer(
        noul=probs[0],
        coverage=coverage,
        missing=[labels[i] for i in missing_idx],
    )
