"""Glue between the simulation scripts (OASIS) and SystemOnePolicy.

Imported by run_twitter_simulation.py, run_reddit_simulation.py and
run_parallel_simulation.py when ``SIM_DECISION_BACKEND=system_one``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .emotion import DEFAULT_ALPHA, AgentStateStore
from .policy import DecisionLog, Observation, SystemOnePolicy
from .taxonomy import load_taxonomy

DECISION_BACKENDS = ("llm", "system_one")
logger = logging.getLogger("mirofish.simulation_policy")


def decision_backend(cli_value: str | None = None) -> str:
    value = (cli_value or os.environ.get("SIM_DECISION_BACKEND") or "llm").strip().lower()
    if value not in DECISION_BACKENDS:
        raise ValueError(f"SIM_DECISION_BACKEND must be llm or system_one, got {value!r}")
    return value


def build_policy(
    platform: str,
    simulation_dir: str,
    *,
    seed: int | None,
    client=None,
    content_provider=None,
    alpha: float | None = None,
) -> SystemOnePolicy:
    """Policy writing agent_state.db and decisions.jsonl in the simulation dir.

    Twitter and Reddit share both files; rows carry the platform.
    """

    if client is None:
        from ..system_one.client import get_system_one_client

        client = get_system_one_client()
    if alpha is None:
        alpha = float(os.environ.get("SIM_EMOTION_ALPHA", DEFAULT_ALPHA))
    return SystemOnePolicy(
        client,
        load_taxonomy(platform),
        seed=seed or 0,
        alpha=alpha,
        content_provider=content_provider,
        state_store=AgentStateStore(os.path.join(simulation_dir, "agent_state.db")),
        decision_log=DecisionLog(os.path.join(simulation_dir, "decisions.jsonl")),
    )


def _persona(agent: Any) -> tuple[str, str]:
    info = getattr(agent, "user_info", None)
    if info is None:
        return f"Agent {getattr(agent, 'social_agent_id', '?')}", ""
    profile = info.profile or {}
    other = profile.get("other_info") if isinstance(profile, dict) else None
    details = []
    if info.description:
        details.append(str(info.description))
    if isinstance(other, dict):
        for key in ("user_profile", "persona", "mbti", "profession", "interested_topics"):
            if other.get(key):
                details.append(str(other[key]))
    return info.name or info.user_name or "", " ".join(details)


async def system_one_actions(
    env: Any,
    active_agents: list[tuple[int, Any]],
    policy: SystemOnePolicy,
    platform: str,
    round_num: int,
    *,
    topics: list[str] | None = None,
) -> dict[Any, Any]:
    """Decide every active agent's action for one round without decode."""

    from oasis import ActionType, ManualAction

    user_names = {}
    for agent_id, agent in env.agent_graph.get_agents():
        name, _ = _persona(agent)
        user_names[agent_id] = name or f"User {agent_id}"

    observations = []
    for agent_id, agent in active_agents:
        refreshed = await agent.env.action.refresh()
        feed = refreshed.get("posts", []) if isinstance(refreshed, dict) and refreshed.get("success") else []
        name, persona = _persona(agent)
        observations.append(
            (
                agent,
                Observation(
                    platform=platform,
                    round_num=round_num,
                    agent_id=agent_id,
                    agent_name=name or user_names.get(agent_id, f"User {agent_id}"),
                    persona=persona,
                    feed=feed,
                    user_names=user_names,
                    topics=list(topics or []),
                ),
            )
        )
    # Bound concurrent agents so the model server's slots are not flooded
    # (each decision already issues several parallel readouts).
    limit = asyncio.Semaphore(max(1, int(os.environ.get("SIM_DECISION_CONCURRENCY", "4"))))

    async def decide(obs: Observation):
        async with limit:
            try:
                return await asyncio.to_thread(policy.decide, obs)
            except Exception as error:  # noqa: BLE001 - one agent must not stop the round
                logger.warning(
                    "System One decision failed for agent %s round %s: %s",
                    obs.agent_id, obs.round_num, error,
                )
                policy.log.write(
                    {
                        "round": obs.round_num,
                        "platform": obs.platform,
                        "agent_id": obs.agent_id,
                        "action": "DO_NOTHING",
                        "args": {},
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                return None

    decisions = await asyncio.gather(*(decide(obs) for _, obs in observations))
    actions = {}
    for (agent, _), decision in zip(observations, decisions):
        if decision is None:
            actions[agent] = ManualAction(action_type=ActionType.DO_NOTHING, action_args={})
        else:
            actions[agent] = ManualAction(
                action_type=ActionType[decision.action], action_args=decision.args
            )
    return actions
