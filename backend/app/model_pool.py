"""Several local models behind one simulation (#48).

``MODEL_POOL`` is a JSON list of OpenAI-compatible endpoints::

    [{"name": "qwen", "base_url": "http://127.0.0.1:8000/v1", "model": "qwen3.5-4b",
      "roles": ["readout", "generate", "decide"]},
     {"name": "gemma", "base_url": "http://127.0.0.1:8002/v1", "model": "gemma-3-4b",
      "roles": ["readout", "generate"], "prompt_format": "plain"}]

Unset, the pool is the one configured model and nothing changes.

* generate: each agent is tied to one model by a hash of its persona ref, so
  it keeps one voice. A failed call falls back to the first model.
* readout: ``SYSTEM_ONE_ENSEMBLE`` names the question keys (or ``all``) that
  every readout model answers; their probabilities are averaged and the
  answer is recomputed. Other questions go to the first readout model. Still
  zero decode. A failed model is left out of that answer.
* Per-model temperatures live under ``models.<name>.temperatures`` in
  ``app/system_one/calibration.json`` (``scripts/calibrate_model.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .system_one.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

logger = logging.getLogger("mirofish.model_pool")

ROLES = frozenset({"readout", "generate", "decide"})
LlmFn = Callable[[str, int], tuple[str, int]]


@dataclass(frozen=True)
class PoolModel:
    name: str
    base_url: str
    model: str
    roles: frozenset[str] = ROLES
    prompt_format: str = "chatml"
    api_key: str | None = None


def load_pool() -> list[PoolModel]:
    raw = os.environ.get("MODEL_POOL", "").strip()
    if not raw:
        from .config import Config

        return [
            PoolModel(
                name=Config.LLM_MODEL_NAME or "default",
                base_url=Config.LLM_BASE_URL or "",
                model=Config.LLM_MODEL_NAME or "",
                prompt_format=Config.SYSTEM_ONE_PROMPT_FORMAT or "chatml",
                api_key=Config.LLM_API_KEY,
            )
        ]
    rows = json.loads(raw)
    pool = [
        PoolModel(
            name=str(row["name"]),
            base_url=str(row["base_url"]),
            model=str(row["model"]),
            roles=frozenset(row.get("roles") or ROLES) & ROLES,
            prompt_format=str(row.get("prompt_format") or "chatml"),
            api_key=row.get("api_key"),
        )
        for row in rows
    ]
    if not pool:
        raise ValueError("MODEL_POOL is empty")
    return pool


def models_for(pool: Iterable[PoolModel], role: str) -> list[PoolModel]:
    return [model for model in pool if role in model.roles]


def _bucket(persona_ref: int, count: int) -> int:
    digest = hashlib.sha256(f"model-pool:{persona_ref}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % count


def assign(persona_ref: int, models: list[PoolModel]) -> PoolModel:
    """The same persona always gets the same model."""

    return models[_bucket(persona_ref, len(models))]


class PooledGenerator:
    """Per-persona generation over named ``LlmFn``s; the first is the fallback."""

    def __init__(self, named: list[tuple[str, LlmFn]]) -> None:
        if not named:
            raise ValueError("PooledGenerator needs at least one model")
        self.named = list(named)

    @property
    def default(self) -> LlmFn:
        return self.named[0][1]

    def name_for(self, persona_ref: int) -> str:
        return self.named[_bucket(persona_ref, len(self.named))][0]

    def for_persona(self, persona_ref: int) -> LlmFn:
        name, fn = self.named[_bucket(persona_ref, len(self.named))]
        if fn is self.default:
            return fn

        def call(prompt: str, max_tokens: int) -> tuple[str, int]:
            try:
                return fn(prompt, max_tokens)
            except Exception as error:  # noqa: BLE001 - degrade to the first model
                logger.warning("model %s failed, using %s: %s", name, self.named[0][0], error)
                return self.default(prompt, max_tokens)

        return call


def _mean(dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = list(dicts[0])
    return {k: sum(d.get(k, 0.0) for d in dicts) / len(dicts) for k in keys}


def _coverage(answers: list) -> float | None:
    values = [a.coverage for a in answers if getattr(a, "coverage", None) is not None]
    return min(values) if values else None


def combine(question, answers: list) -> object:
    """Average the models' distributions and rebuild the answer; coverage is
    the lowest of the models'."""

    if len(answers) == 1:
        return answers[0]
    coverage = _coverage(answers)
    if isinstance(question, ChoiceQuestion):
        probs = _mean([a.probabilities for a in answers])
        best = max(probs, key=probs.get)
        return ChoiceAnswer(choice=best, probabilities=probs, confidence=probs[best], coverage=coverage)
    if isinstance(question, ScoreQuestion):
        probs = _mean([a.probabilities for a in answers])
        levels = list(question.criteria)
        values = [probs.get(level, 0.0) for level in levels]
        return ScoreAnswer(
            score=sum(i * p for i, p in enumerate(values)),
            probabilities=probs,
            confidence=max(values),
            coverage=coverage,
        )
    return NoulAnswer(noul=sum(a.noul for a in answers) / len(answers), coverage=coverage)


class EnsembleBackend:
    """Readout ensemble over several System One backends (#48).

    ``scope`` None means every question; otherwise only those keys are asked
    of every backend, the rest of the first backend alone.
    """

    def __init__(self, backends: list, scope: set[str] | None) -> None:
        if not backends:
            raise ValueError("EnsembleBackend needs at least one backend")
        self.backends = list(backends)
        self.scope = scope

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        shared = {
            name: q for name, q in request.questions.items() if self.scope is None or name in self.scope
        }
        responses = []
        errors = []
        for index, backend in enumerate(self.backends):
            questions = request.questions if index == 0 else shared
            if not questions:
                continue
            try:
                responses.append((index, backend.ask(request.model_copy(update={"questions": questions}))))
            except Exception as error:  # noqa: BLE001 - one endpoint down leaves the others
                logger.warning("readout backend %s failed: %s", index, error)
                errors.append(error)
        if not responses:
            raise errors[0] if errors else RuntimeError("no readout backend answered")
        answers = {}
        for name, question in request.questions.items():
            got = [resp.answers[name] for _, resp in responses if name in resp.answers]
            if not got:
                raise RuntimeError(f"no readout backend answered {name!r}")
            answers[name] = combine(question, got) if name in shared else got[0]
        return SystemOneResponse(
            model="+".join(str(resp.model or i) for i, resp in responses),
            answers=answers,
            usage=Usage(
                input_tokens=sum(resp.usage.input_tokens for _, resp in responses),
                output_tokens=sum(resp.usage.output_tokens for _, resp in responses),
            ),
        )


def ensemble_scope() -> set[str] | None:
    """``SYSTEM_ONE_ENSEMBLE``: ``all`` -> None (every question), else key set."""

    raw = os.environ.get("SYSTEM_ONE_ENSEMBLE", "").strip()
    if raw.lower() == "all":
        return None
    return {key.strip() for key in raw.split(",") if key.strip()}
