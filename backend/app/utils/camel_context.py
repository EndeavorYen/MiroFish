"""Chat-memory budget for OASIS agents on small local models (#28).

camel sizes an agent's context from ``model_backend.token_limit``, which is
999_999_999 for a model name it does not know (``qwen3.5-4b`` on
llama.cpp). OASIS agents therefore resend every past turn: prompts grow from
~3K to 30-60K tokens over 24 rounds, a server with 8K per slot rejects them,
and those agent turns are silently lost.

Lowering ``token_limit`` is not a fix: once memory is near the limit camel
slices each *new* message into small chunks and then drops the oldest
chunks, so the agent sees a fragment of its current feed. Instead
``TurnBudgetMemory`` keeps whole turns. A turn runs from one user message
(the feed observation or an interview question) to the next, so a tool call
never loses its result. The system message and the current turn are always
kept; earlier turns are added newest first while they fit in the budget.

The budget counts chat messages only (camel's token counter, which
over-counts Chinese for Qwen tokenizers); leave room in the server's
per-slot context for the tool schemas and the reply.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from camel.memories import ChatHistoryMemory, ContextRecord
from camel.types import OpenAIBackendRole

ENV_VAR = "SIM_AGENT_CONTEXT_TOKENS"
# Unset + a local model server: fits llama-server -c 65536 -np 8 (8K per slot)
# with room for the tool schemas and the reply.
DEFAULT_LOCAL_BUDGET = 3072
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _local_llm() -> bool:
    host = urlparse(os.environ.get("LLM_BASE_URL") or "").hostname or ""
    return host in _LOCAL_HOSTS


def context_budget_from_env() -> int | None:
    """``SIM_AGENT_CONTEXT_TOKENS`` as a positive int; unset means
    DEFAULT_LOCAL_BUDGET for a local LLM_BASE_URL and None (no budget,
    camel's default) for a hosted model. ``0`` or ``off`` disables it."""

    raw = (os.environ.get(ENV_VAR) or "").strip()
    if not raw:
        return DEFAULT_LOCAL_BUDGET if _local_llm() else None
    if raw.lower() in ("0", "off"):
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_VAR} must be an integer, got {raw!r}") from error
    if value < 0:
        raise ValueError(f"{ENV_VAR} must be positive, got {value}")
    return value


class TurnBudgetMemory(ChatHistoryMemory):
    """ChatHistoryMemory that returns the system message, the current turn
    and as many whole earlier turns as fit in ``_turn_budget`` tokens."""

    _turn_budget: int = 0

    def retrieve(self) -> list[ContextRecord]:
        records = super().retrieve()
        # camel sends only the first system message (ScoreBasedContextCreator
        # drops later ones, e.g. OASIS's per-ManualAction records), so only
        # that one is kept and charged.
        system = records[:1] if records and records[0].memory_record.role_at_backend == OpenAIBackendRole.SYSTEM else []
        rest = [r for r in records if r.memory_record.role_at_backend != OpenAIBackendRole.SYSTEM]
        turns: list[list[ContextRecord]] = []
        for record in rest:
            if record.memory_record.role_at_backend == OpenAIBackendRole.USER or not turns:
                turns.append([])
            turns[-1].append(record)
        if not turns:
            return records

        counter = self.get_context_creator().token_counter

        def cost(group: list[ContextRecord]) -> int:
            return counter.count_tokens_from_messages(
                [r.memory_record.to_openai_message() for r in group]
            )

        used = cost(system) + cost(turns[-1])
        kept = [turns[-1]]
        for turn in reversed(turns[:-1]):
            size = cost(turn)
            if used + size > self._turn_budget:
                break
            kept.append(turn)
            used += size
        return system + [r for turn in reversed(kept) for r in turn]


def apply_memory_budget(agent: Any, tokens: int | None) -> None:
    """Give a camel ChatAgent a TurnBudgetMemory (same storage, in place)."""

    if tokens is None:
        return
    memory = agent.memory
    if not isinstance(memory, ChatHistoryMemory):
        raise TypeError(f"unsupported agent memory: {type(memory).__name__}")
    memory.__class__ = TurnBudgetMemory
    memory._turn_budget = tokens


def apply_agent_graph_budget(agent_graph: Any, tokens: int | None) -> int:
    """Apply the budget to every agent in an OASIS AgentGraph; returns the count."""

    if tokens is None:
        return 0
    count = 0
    for _, agent in agent_graph.get_agents():
        apply_memory_budget(agent, tokens)
        count += 1
    return count
