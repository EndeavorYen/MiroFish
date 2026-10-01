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
One state per simulation, shared by Twitter and Reddit: a role has one
stance, and what it reads on either platform moves it (once per platform and
round). A reading that fails is skipped; the role keeps its stance when none
succeeds. The update reads posts, not other roles' states, so the order of
concurrent decisions does not matter, and each text is scored once; a
seeded run replays when the readouts do (a -np 1 model service with
SYSTEM_ONE_CACHE_PROMPT=0, #82).

Only the System One decisions use it (``SIM_DECISION_BACKEND=system_one``).

Parameters: ``SIM_OPINION_MU`` (0.3), ``SIM_OPINION_RADIUS`` (0.3),
``SIM_OPINION_STUBBORNNESS`` (0.5); stubbornness per entity type from
``opinion_params.json`` next to this module (fitted on the calibration
scenarios only). Off by default until the evaluation says otherwise.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

ENV_VAR = "SIM_OPINION_DYNAMICS"
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "opinion_params.json")
MAX_FEED = 20  # policy.MAX_FEED: what the decision prompt shows
COMMENTS_PER_POST = 2
# OASIS wraps posts in its feed (social_platform/platform_utils.py).
_REPORTED = re.compile(r"^\[Warning: This post has been reported \d+ times\]\n")
_REPOST = re.compile(r"^User \d+ reposted a post from User (\d+)\. Repost content: (.*)\. $", re.S)
_QUOTE = re.compile(r"^User \d+ quoted a post from User \d+\. Quote content: (.*?)\. Original Content: ", re.S)


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
    if mode not in ("1", "on", "true", "yes", "bounded"):
        raise ValueError(f"{ENV_VAR} must be off or bounded, got {mode!r}")
    fitted: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            fitted = json.load(f)
        if not isinstance(fitted, dict) or not isinstance(fitted.get("stubbornness_by_type", {}), dict):
            raise ValueError(f"{path} must be an object with an optional stubbornness_by_type object")

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


def _words(post: dict[str, Any]) -> tuple[str, Any]:
    """The words a feed entry carries and whose they are: a repost carries
    the original author's, a quote the quoter's; no report warning."""

    text = _REPORTED.sub("", str(post.get("content") or ""))
    author = post.get("user_id")
    repost = _REPOST.match(text)
    if repost:
        return repost.group(2), int(repost.group(1))
    quote = _QUOTE.match(text)
    if quote:
        return quote.group(1), author
    return text, author


def feed_texts(feed: list[dict[str, Any]], agent_id: int) -> list[str]:
    """The texts a role reads this round: the posts and comments of its
    decision prompt, without its own words (also not reposted back to it)."""

    texts = []
    for post in feed[:MAX_FEED]:
        words, author = _words(post)
        if author != agent_id and words.strip():
            texts.append(words)
        for comment in (post.get("comments") or [])[:COMMENTS_PER_POST]:
            if comment.get("user_id") != agent_id and comment.get("content"):
                texts.append(str(comment["content"]))
    return texts


class OpinionState:
    """Every role's current stance in one simulation; thread safe."""

    def __init__(self, initial: Mapping[int, float], params: OpinionParams,
                 score: Callable[[str], float], entity_types: Mapping[int, str] | None = None) -> None:
        self.params = params
        self._score = score
        self._stance = dict(initial)
        self._types = dict(entity_types or {})
        self._updated: dict[tuple[int, str], int] = {}  # (agent, platform) -> last round read
        self._cache: dict[str, float | None] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def stance(self, agent_id: int) -> float | None:
        with self._lock:
            return self._stance.get(agent_id)

    def _reading(self, text: str) -> float | None:
        """The text's stance, scored once even when several roles read it at
        the same time; ``None`` when the readout failed (not retried)."""

        with self._lock:
            if text in self._cache:
                return self._cache[text]
            waiter = self._inflight.get(text)
            if waiter is None:
                self._inflight[text] = threading.Event()
        if waiter is not None:
            waiter.wait()
            with self._lock:
                return self._cache.get(text)
        try:
            value: float | None = float(self._score(text))  # outside the lock: a model call
        except Exception:
            value = None
        with self._lock:
            self._cache[text] = value
            self._inflight.pop(text).set()
        return value

    def update(self, agent_id: int, round_num: int, feed: list[dict[str, Any]],
               platform: str = "") -> dict[str, Any] | None:
        """Read this round's feed on ``platform`` once; the record of the
        change, or ``None`` when the role has no stance to move or already
        read this round there. Readings that fail are skipped."""

        with self._lock:
            before = self._stance.get(agent_id)
            if before is None or self._updated.get((agent_id, platform)) == round_num:
                return None
            self._updated[(agent_id, platform)] = round_num
        texts = feed_texts(feed, agent_id)
        readings = [self._reading(text) for text in texts]
        scored = [r for r in readings if r is not None]
        stubbornness = self.params.stubbornness_of(self._types.get(agent_id))
        with self._lock:
            # The stance now: the other platform may have moved it meanwhile.
            current = self._stance[agent_id]
            after, close = bounded_update(current, scored, self.params.mu, self.params.radius, stubbornness)
            self._stance[agent_id] = after
        record = {"before": round(current, 4), "after": round(after, 4), "read": len(scored), "close": close}
        if len(scored) < len(readings):
            record["failed"] = len(readings) - len(scored)
        return record
