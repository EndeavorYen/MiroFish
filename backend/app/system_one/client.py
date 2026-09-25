"""System One client and factory."""

from __future__ import annotations

import json
from pathlib import Path

from .backends import HttpBackend, LocalReadoutBackend, SystemOneBackend
from .models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


def load_temperatures(path: Path = CALIBRATION_PATH) -> dict[str, float]:
    """Per-question-type temperatures fitted by ``scripts/system_one_eval.py``."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {k: float(v) for k, v in (data.get("temperatures") or {}).items()}


class SystemOneClient:
    def __init__(self, backend: SystemOneBackend) -> None:
        self.backend = backend

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        return self.backend.ask(request)

    def choice(self, state: str, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        question = ChoiceQuestion(instructions=instructions, criteria=criteria)
        return self.ask(SystemOneRequest(state=state, questions={"q": question})).answers["q"]

    def score(self, state: str, instructions: str, criteria: list[str]) -> ScoreAnswer:
        question = ScoreQuestion(instructions=instructions, criteria=criteria)
        return self.ask(SystemOneRequest(state=state, questions={"q": question})).answers["q"]

    def noul(self, state: str, instructions: str) -> NoulAnswer:
        question = NoulQuestion(instructions=instructions)
        return self.ask(SystemOneRequest(state=state, questions={"q": question})).answers["q"]


def get_system_one_client() -> SystemOneClient:
    """Build the client from ``SYSTEM_ONE_*`` settings.

    ``local`` reads logprobs from the local model server at
    ``SYSTEM_ONE_BASE_URL`` (an OpenAI-compatible ``/v1`` base). ``http``
    posts to ``{SYSTEM_ONE_BASE_URL}/v1/systemone``.
    """

    from ..config import Config

    backend = (Config.SYSTEM_ONE_BACKEND or "local").strip().lower()
    if backend == "local":
        if Config.SYSTEM_ONE_PROMPT_FORMAT not in ("chatml", "plain"):
            raise ValueError(
                "SYSTEM_ONE_PROMPT_FORMAT must be chatml or plain, "
                f"got {Config.SYSTEM_ONE_PROMPT_FORMAT!r}"
            )
        return SystemOneClient(
            LocalReadoutBackend(
                base_url=Config.SYSTEM_ONE_BASE_URL,
                model=Config.SYSTEM_ONE_MODEL,
                api_key=Config.SYSTEM_ONE_API_KEY,
                top_k=Config.SYSTEM_ONE_TOP_K,
                prompt_format=Config.SYSTEM_ONE_PROMPT_FORMAT,
                temperatures=load_temperatures(),
            )
        )
    if backend == "http":
        return SystemOneClient(
            HttpBackend(
                base_url=Config.SYSTEM_ONE_BASE_URL,
                api_key=Config.SYSTEM_ONE_API_KEY,
                model=Config.SYSTEM_ONE_MODEL,
            )
        )
    raise ValueError(f"SYSTEM_ONE_BACKEND must be local or http, got {backend!r}")
