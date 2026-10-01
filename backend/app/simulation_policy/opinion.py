"""Opinion dynamics (#59): a role's stance moves with what it reads.

Without it, a role's stance is the prep label for the whole run: half of
every post's intent is anchored on it, so a role does not change its mind
after reading others, and the trend comes from activity, not opinion.

With ``SIM_OPINION_DYNAMICS=bounded`` each role's stance is a state. Once a
round, when the role is activated, it reads the posts in its feed (the same
posts and comments its decision prompt shows, without its own), each scored
by the zero-decode stance readout on the run's question (one readout per
distinct text, shared). Bounded confidence (Hegselmann-Krause): only posts
within ``radius`` of its stance pull it, towards their mean, by

    stance += mu * (1 - stubbornness) * (mean(close posts) - stance)

so a role that reads nothing, or nothing close enough, keeps its stance.
The update reads posts, not other roles' states, so it does not depend on
the order in which concurrent decisions run: a seeded run replays.

Parameters: ``SIM_OPINION_MU`` (0.3), ``SIM_OPINION_RADIUS`` (0.3),
``SIM_OPINION_STUBBORNNESS`` (0.5); stubbornness per entity type from
``opinion_params.json`` next to this module (fitted on the calibration
scenarios only). Off by default until the evaluation says otherwise.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

ENV_VAR = "SIM_OPINION_DYNAMICS"
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "opinion_params.json")
MAX_FEED = 20  # policy.MAX_FEED: what the decision prompt shows
COMMENTS_PER_POST = 2


@dataclass(frozen=True)
class OpinionParams:
    mu: float = 0.3
    radius: float = 0.3
    stubbornness: float = 0.5
    by_type: Mapping[str, float] = field(default_factory=dict)  # entity type -> stubbornness

    def __post_init__(self) -> None:
        for name in ("mu", "radius", "stubbornness"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"opinion {name} must be within 0..1, got {value}")
        for kind, value in self.by_type.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"opinion stubbornness of {kind} must be within 0..1, got {value}")

    def stubbornness_of(self, entity_type: str | None) -> float:
        return self.by_type.get(entity_type or "", self.stubbornness)


def load_params(env: Mapping[str, str] | None = None, path: str = PARAMS_FILE) -> OpinionParams | None:
    """The parameters when opinion dynamics is on, else ``None``."""

    env = os.environ if env is None else env
    mode = (env.get(ENV_VAR) or "off").strip().lower()
    if mode in ("", "0", "off", "false", "no"):
        return None
    if mode not in ("1", "on", "bounded"):
        raise ValueError(f"{ENV_VAR} must be off or bounded, got {mode!r}")
    fitted: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            fitted = json.load(f)

    def number(name: str, key: str, default: float) -> float:
        raw = env.get(name)
        try:
            return float(raw) if raw not in (None, "") else float(fitted.get(key, default))
        except ValueError as error:
            raise ValueError(f"{name} must be a number within 0..1") from error

    return OpinionParams(
        mu=number("SIM_OPINION_MU", "mu", 0.3),
        radius=number("SIM_OPINION_RADIUS", "radius", 0.3),
        stubbornness=number("SIM_OPINION_STUBBORNNESS", "stubbornness", 0.5),
        by_type={str(k): float(v) for k, v in (fitted.get("stubbornness_by_type") or {}).items()},
    )


def bounded_update(stance: float, readings: Iterable[float], mu: float, radius: float,
                   stubbornness: float) -> tuple[float, int]:
    """The new stance and how many readings were close enough to pull it."""

    close = [r for r in readings if abs(r - stance) <= radius]
    if not close:
        return stance, 0
    target = sum(close) / len(close)
    moved = stance + mu * (1.0 - stubbornness) * (target - stance)
    return min(1.0, max(0.0, moved)), len(close)


def feed_texts(feed: list[dict[str, Any]], agent_id: int) -> list[str]:
    """The texts a role reads this round: the posts and comments of its
    decision prompt, without its own."""

    texts = []
    for post in feed[:MAX_FEED]:
        if post.get("user_id") != agent_id and post.get("content"):
            texts.append(str(post["content"]))
        for comment in (post.get("comments") or [])[:COMMENTS_PER_POST]:
            if comment.get("user_id") != agent_id and comment.get("content"):
                texts.append(str(comment["content"]))
    return texts


class OpinionState:
    """Every role's current stance on one platform; thread safe."""

    def __init__(self, initial: Mapping[int, float], params: OpinionParams,
                 score: Callable[[str], float], entity_types: Mapping[int, str] | None = None) -> None:
        self.params = params
        self._score = score
        self._stance = dict(initial)
        self._types = dict(entity_types or {})
        self._updated: dict[int, int] = {}  # agent -> last round updated
        self._cache: dict[str, float] = {}
        self._lock = threading.Lock()

    def stance(self, agent_id: int) -> float | None:
        with self._lock:
            return self._stance.get(agent_id)

    def _reading(self, text: str) -> float:
        with self._lock:
            if text in self._cache:
                return self._cache[text]
        value = float(self._score(text))  # outside the lock: a model call
        with self._lock:
            return self._cache.setdefault(text, value)

    def update(self, agent_id: int, round_num: int, feed: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Read this round's feed once; the record of the change, or ``None``
        when the role has no stance to move or already read this round."""

        with self._lock:
            before = self._stance.get(agent_id)
            if before is None or self._updated.get(agent_id) == round_num:
                return None
            self._updated[agent_id] = round_num
        texts = feed_texts(feed, agent_id)
        readings = [self._reading(text) for text in texts]
        stubbornness = self.params.stubbornness_of(self._types.get(agent_id))
        after, close = bounded_update(before, readings, self.params.mu, self.params.radius, stubbornness)
        with self._lock:
            self._stance[agent_id] = after
        return {"before": round(before, 4), "after": round(after, 4), "read": len(readings), "close": close}
