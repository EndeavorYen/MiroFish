"""Persistent agent emotion state (#9).

Emotion is a state with inertia, not a per-round pick:

    e_t = alpha * e_{t-1} + (1 - alpha) * readout_t

``readout_t`` is a System One ``score`` per dimension on the round's
observation, mapped to 0..1. ``agent_state.db`` keeps every round.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence

DIMENSIONS: tuple[str, ...] = ("anger", "fear", "trust", "joy", "sadness")
DIMENSION_LABELS = {
    "anger": "憤怒",
    "fear": "恐懼",
    "trust": "信任",
    "joy": "喜悅",
    "sadness": "悲傷",
}
LEVELS = ["完全沒有", "輕微", "中等", "強烈", "非常強烈"]
DEFAULT_ALPHA = 0.7
NEUTRAL = 0.2


def neutral_state(dimensions: Sequence[str] = DIMENSIONS) -> dict[str, float]:
    return {d: NEUTRAL for d in dimensions}


def update_emotion(
    previous: Mapping[str, float], readout: Mapping[str, float], alpha: float
) -> dict[str, float]:
    """``alpha * previous + (1 - alpha) * readout`` per dimension, clipped to 0..1.

    alpha = 1 keeps the previous state; alpha = 0 takes the readout.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within 0..1")
    result = {}
    for dim, prev in previous.items():
        value = alpha * prev + (1 - alpha) * readout.get(dim, prev)
        result[dim] = min(1.0, max(0.0, value))
    return result


def score_to_unit(score: float, levels: int = len(LEVELS)) -> float:
    """System One score (0-based expected level) -> 0..1."""

    return min(1.0, max(0.0, score / (levels - 1)))


def describe(state: Mapping[str, float]) -> str:
    return "、".join(f"{DIMENSION_LABELS.get(d, d)} {v:.2f}" for d, v in state.items())


class AgentStateStore:
    """``agent_state.db``: emotion per (round, agent)."""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emotion ("
            " round INTEGER NOT NULL, platform TEXT NOT NULL, agent_id INTEGER NOT NULL,"
            " state TEXT NOT NULL, PRIMARY KEY (round, platform, agent_id))"
        )
        self._conn.commit()

    def latest(self, platform: str, agent_id: int, before_round: int) -> dict[str, float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM emotion WHERE platform = ? AND agent_id = ? AND round < ? "
                "ORDER BY round DESC LIMIT 1",
                (platform, agent_id, before_round),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, round_num: int, platform: str, agent_id: int, state: Mapping[str, float]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO emotion VALUES (?, ?, ?, ?)",
                (round_num, platform, agent_id, json.dumps(dict(state), sort_keys=True)),
            )
            self._conn.commit()

    def history(self, platform: str, agent_id: int) -> list[tuple[int, dict[str, float]]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT round, state FROM emotion WHERE platform = ? AND agent_id = ? ORDER BY round",
                (platform, agent_id),
            ).fetchall()
        return [(r, json.loads(s)) for r, s in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
