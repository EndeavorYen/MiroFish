"""Jev-compatible request/response models for ``/v1/systemone``.

Request shape (TypeSafe Jev):

    {"state": str, "model": str,
     "questions": {name: {"type": "choice"|"score"|"noul",
                          "instructions": str,
                          "criteria": {...} | [...]}}}

``choice`` criteria map option keys to descriptions, ``score`` criteria are
2-10 ordered level descriptions (level index is 0-based), ``noul`` has no
criteria. Answers carry ``probabilities`` and ``confidence`` (``choice`` /
``score``) or a single ``noul`` probability. ``coverage`` and ``missing`` are
local readout extensions and are omitted by Jev.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str


class ChoiceQuestion(_Question):
    type: Literal["choice"] = "choice"
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _at_least_two(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) < 2:
            raise ValueError("choice needs at least two options")
        return value


class ScoreQuestion(_Question):
    type: Literal["score"] = "score"
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, value: list[str]) -> list[str]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, got {len(value)}"
            )
        if len(set(value)) != len(value):
            # probabilities are keyed by level text; duplicates would merge.
            raise ValueError("score levels must be distinct")
        return value


class NoulQuestion(_Question):
    type: Literal["noul"] = "noul"


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")
]


class SystemOneRequest(BaseModel):
    state: str
    model: str | None = None
    questions: dict[str, Question]


class _Answer(BaseModel):
    coverage: float | None = None
    missing: list[str] | None = None


class ChoiceAnswer(_Answer):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(_Answer):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(_Answer):
    """Jev noul: the probability of "yes". Jev has no separate confidence."""

    type: Literal["noul"] = "noul"
    noul: float


Answer = Annotated[Union[ChoiceAnswer, ScoreAnswer, NoulAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str | None = None
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
