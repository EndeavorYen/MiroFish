"""Hierarchical System One decision policy (#9).

Per active agent and round, with no decode:

1. observe: persona + current emotion + the refreshed feed (at most 20 posts);
2. update emotion: one System One ``score`` per dimension, with inertia;
3. pick an action path with ``ask_tree`` over the platform taxonomy;
4. pick a target (post, comment, user, query) with ``choice`` when needed;
5. when text is needed, build a ``ContentIntent`` for the ContentProvider;
6. return the OASIS action and append the decision to ``decisions.jsonl``.

Every random draw comes from an RNG seeded by (seed, platform, round, agent),
sampled from the distribution (never argmax): the same observations and
readouts give the same decisions. A live parallel run is not bit-for-bit
replayable, because OASIS draws feeds from the global ``random`` module
shared by both platforms and the model server's batching is not
deterministic.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from dataclasses import dataclass, field
from typing import Any

from ..system_one.models import ChoiceQuestion, ScoreQuestion, SystemOneRequest
from ..system_one.tree import ask_tree, sample_option
from .content import ContentIntent, ContentProvider, TemplateContentProvider
from .emotion import (
    DEFAULT_ALPHA,
    DIMENSION_LABELS,
    DIMENSIONS,
    LEVELS,
    AgentStateStore,
    describe,
    neutral_state,
    score_to_unit,
    update_emotion,
)
from .taxonomy import Taxonomy

MAX_FEED = 20
MAX_ACTIONS_PER_ROUND = 3
MAX_TEXT = 120
STANCE_LEVELS = ["強烈反對", "反對", "中立", "支持", "強烈支持"]
INTENSITY_LEVELS = ["平和", "有些情緒", "非常激動"]


def decision_rng(seed: int, platform: str, round_num: int, agent_id: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{platform}:{round_num}:{agent_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


@dataclass
class Observation:
    platform: str
    round_num: int
    agent_id: int
    agent_name: str
    persona: str
    feed: list[dict[str, Any]]
    user_names: dict[int, str] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)


@dataclass
class Decision:
    action: str  # OASIS ActionType name, e.g. "LIKE_POST"
    args: dict[str, Any]
    record: dict[str, Any]
    # (emotion, readout) of this round, reused by later actions in it
    emotion: tuple[dict[str, float], dict[str, float]] | None = None


class DecisionLog:
    """Append-only ``decisions.jsonl`` (sorted keys, one line per decision)."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class SystemOnePolicy:
    def __init__(
        self,
        client,
        taxonomy: Taxonomy,
        *,
        seed: int = 0,
        alpha: float = DEFAULT_ALPHA,
        content_provider: ContentProvider | None = None,
        state_store: AgentStateStore | None = None,
        decision_log: DecisionLog | None = None,
        extra_action_rate: float = 0.0,
    ) -> None:
        if not 0.0 <= extra_action_rate < 1.0:
            raise ValueError(f"extra_action_rate must be within [0, 1), got {extra_action_rate}")
        self.client = client
        self.taxonomy = taxonomy
        self.seed = seed
        self.alpha = alpha
        self.content = content_provider or TemplateContentProvider()
        self.state_store = state_store
        self.log = decision_log or DecisionLog(None)
        # Probability of one more action after each action in a round (#26):
        # LLM agents take ~1.6 actions per activation.
        self.extra_action_rate = extra_action_rate
        self._memory: dict[tuple[str, int], dict[str, float]] = {}
        self._memory_lock = threading.Lock()

    # ----------------------------------------------------------- observation

    def feed_text(self, obs: Observation) -> str:
        lines = []
        for post in obs.feed[:MAX_FEED]:
            author = obs.user_names.get(post.get("user_id"), f"User {post.get('user_id')}")
            lines.append(f"[貼文 {post.get('post_id')}] {author}：{_clip(post.get('content', ''))}")
            for comment in (post.get("comments") or [])[:2]:
                c_author = obs.user_names.get(comment.get("user_id"), f"User {comment.get('user_id')}")
                lines.append(
                    f"  [留言 {comment.get('comment_id')}] {c_author}：{_clip(comment.get('content', ''), 60)}"
                )
        return "\n".join(lines) if lines else "（動態目前沒有貼文）"

    def base_state(self, obs: Observation) -> str:
        return f"平台：{obs.platform}\n人物：{obs.agent_name}。{_clip(obs.persona, 300)}\n動態：\n{self.feed_text(obs)}"

    # --------------------------------------------------------------- emotion

    def _previous_emotion(self, obs: Observation) -> dict[str, float]:
        key = (obs.platform, obs.agent_id)
        with self._memory_lock:
            if key in self._memory:
                return dict(self._memory[key])
        if self.state_store is not None:
            stored = self.state_store.latest(obs.platform, obs.agent_id, obs.round_num)
            if stored:
                return stored
        return neutral_state()

    def update_emotion(self, obs: Observation, state: str) -> tuple[dict[str, float], dict[str, float]]:
        questions = {
            dim: ScoreQuestion(
                instructions=f"看完這些動態後，這個人此刻的{DIMENSION_LABELS[dim]}程度？",
                criteria=LEVELS,
            )
            for dim in DIMENSIONS
        }
        response = self.client.ask(SystemOneRequest(state=state, questions=questions))
        readout = {dim: score_to_unit(response.answers[dim].score) for dim in DIMENSIONS}
        previous = self._previous_emotion(obs)
        current = update_emotion(previous, readout, self.alpha)
        with self._memory_lock:
            self._memory[(obs.platform, obs.agent_id)] = dict(current)
        if self.state_store is not None:
            self.state_store.save(obs.round_num, obs.platform, obs.agent_id, current)
        return current, readout

    # --------------------------------------------------------------- targets

    def _choose(
        self, state: str, instructions: str, criteria: dict[str, str], rng: random.Random
    ) -> tuple[str, dict[str, float]]:
        if len(criteria) == 1:
            only = next(iter(criteria))
            return only, {only: 1.0}
        answer = self.client.ask(
            SystemOneRequest(
                state=state,
                questions={"q": ChoiceQuestion(instructions=instructions, criteria=criteria)},
            )
        ).answers["q"]
        return sample_option(answer.probabilities, rng), dict(answer.probabilities)

    def _target_options(self, need: str, obs: Observation) -> dict[str, tuple[str, Any]]:
        """option key -> (description, value)."""

        options: dict[str, tuple[str, Any]] = {}
        if need == "post":
            for post in obs.feed[:MAX_FEED]:
                options[f"p{post['post_id']}"] = (_clip(post.get("content", ""), 80), post)
        elif need == "comment":
            for post in obs.feed[:MAX_FEED]:
                for comment in post.get("comments") or []:
                    if len(options) >= MAX_FEED:
                        break
                    options[f"c{comment['comment_id']}"] = (
                        _clip(comment.get("content", ""), 80),
                        comment,
                    )
        elif need == "user":
            for post in obs.feed[:MAX_FEED]:
                user_id = post.get("user_id")
                if user_id is None or user_id == obs.agent_id or f"u{user_id}" in options:
                    continue
                options[f"u{user_id}"] = (obs.user_names.get(user_id, f"User {user_id}"), user_id)
        elif need == "query":
            for i, topic in enumerate(obs.topics[:MAX_FEED]):
                options[f"t{i}"] = (topic, topic)
            if not options:
                for post in obs.feed[:MAX_FEED]:
                    snippet = _clip(post.get("content", ""), 12).rstrip("…")
                    if snippet:
                        options[f"p{post['post_id']}"] = (snippet, snippet)
        return options

    # ---------------------------------------------------------------- intent

    def build_intent(
        self,
        obs: Observation,
        state: str,
        emotion: dict[str, float],
        rng: random.Random,
        target: dict[str, Any] | None,
    ) -> tuple[ContentIntent, dict[str, Any]]:
        kinds = self.taxonomy.dialogue_kinds or {"opinion": "表達看法", "other": "其他"}
        target_text = (target or {}).get("content", "")
        intent_state = state + (f"\n要回應的貼文：{_clip(target_text)}" if target_text else "")
        kind, kind_probs = self._choose(intent_state, "這個人這次想說哪一種話？", kinds, rng)
        response = self.client.ask(
            SystemOneRequest(
                state=intent_state,
                questions={
                    "stance": ScoreQuestion(
                        instructions="這個人對目前討論的主要事件持什麼立場？", criteria=STANCE_LEVELS
                    ),
                    "intensity": ScoreQuestion(
                        instructions="這個人這次發言的情緒強度？", criteria=INTENSITY_LEVELS
                    ),
                },
            )
        )
        stance = score_to_unit(response.answers["stance"].score, len(STANCE_LEVELS))
        intensity = score_to_unit(response.answers["intensity"].score, len(INTENSITY_LEVELS))
        intent = ContentIntent(
            kind=kind,
            stance=stance,
            intensity=intensity,
            target_ref=str(target["post_id"]) if target and "post_id" in target else None,
            persona_ref=obs.agent_id,
            platform=obs.platform,
            round_num=obs.round_num,
            agent_name=obs.agent_name,
            persona=obs.persona,
            target_text=target_text,
            topic=rng.choice(obs.topics) if obs.topics else "",
            emotion=dict(emotion),
        )
        record = {
            "kind": kind,
            "kind_probs": kind_probs,
            "stance": round(stance, 4),
            "intensity": round(intensity, 4),
        }
        return intent, record

    # ------------------------------------------------------------------ main

    def decide_round(self, obs: Observation) -> list[Decision]:
        """The agent's actions this round: one decision, then another with
        probability ``extra_action_rate`` after each action, up to
        MAX_ACTIONS_PER_ROUND; a DO_NOTHING ends the round."""

        first = self.decide(obs)
        decisions = [first]
        if first.action == "DO_NOTHING" or self.extra_action_rate <= 0:
            return decisions
        emotion = first.emotion
        for index in range(1, MAX_ACTIONS_PER_ROUND):
            gate = decision_rng(self.seed, f"{obs.platform}+more{index}", obs.round_num, obs.agent_id)
            if gate.random() >= self.extra_action_rate:
                break
            done = [d.action for d in decisions]
            decision = self.decide(obs, index=index, emotion=emotion, done=done)
            if decision.action == "DO_NOTHING":
                break
            decisions.append(decision)
        return decisions

    def decide(
        self,
        obs: Observation,
        *,
        index: int = 0,
        emotion: tuple[dict[str, float], dict[str, float]] | None = None,
        done: list[str] | None = None,
    ) -> Decision:
        platform_key = obs.platform if index == 0 else f"{obs.platform}+{index}"
        rng = decision_rng(self.seed, platform_key, obs.round_num, obs.agent_id)
        base = self.base_state(obs)
        if emotion is None:
            emotion, readout = self.update_emotion(obs, base)
        else:
            emotion, readout = emotion  # later actions in the round reuse it
        state = f"{base}\n目前情緒：{describe(emotion)}"
        if done:
            state += f"\n這一輪已經做了：{'、'.join(done)}（接下來還會做什麼？）"
        steps = ask_tree(self.client, state, self.taxonomy.tree, rng)
        path = [step.choice for step in steps]
        leaf = self.taxonomy.leaf(path)
        record: dict[str, Any] = {
            "round": obs.round_num,
            "platform": obs.platform,
            "agent_id": obs.agent_id,
            "path": path,
            "options": [list(step.probabilities) for step in steps],
            "probs": [
                {k: round(v, 6) for k, v in step.probabilities.items()} for step in steps
            ],
            # Raw readouts before action priors (#26), for refitting them.
            **(
                {"raw_probs": [{k: round(v, 6) for k, v in (step.readout or step.probabilities).items()} for step in steps]}
                if any(step.readout is not None for step in steps)
                else {}
            ),
            "chosen": leaf.action,
            **({"action_index": index} if index else {}),
            "state_hash": hashlib.sha256(state.encode("utf-8")).hexdigest(),
            "emotion": {k: round(v, 6) for k, v in emotion.items()},
            "emotion_readout": {k: round(v, 6) for k, v in readout.items()},
        }

        action, args = leaf.action, {}
        target_post: dict[str, Any] | None = None
        for need in leaf.needs:
            if need == "content":
                continue
            options = self._target_options(need, obs)
            if not options:
                action, args = "DO_NOTHING", {}
                record["fallback"] = f"no {need} to act on"
                break
            key, probs = self._choose(
                state,
                {
                    "post": "這個人會選哪一則貼文？",
                    "comment": "這個人會選哪一則留言？",
                    "user": "這個人會選哪一位使用者？",
                    "query": "這個人會搜尋什麼？",
                }[need],
                {k: v[0] for k, v in options.items()},
                rng,
            )
            value = options[key][1]
            record[f"{need}_options"] = list(options)
            record[f"{need}_probs"] = {k: round(v, 6) for k, v in probs.items()}
            record[f"{need}_chosen"] = key
            if need == "post":
                target_post = value
                args["post_id"] = value["post_id"]
            elif need == "comment":
                args["comment_id"] = value["comment_id"]
            elif need == "user":
                args["mutee_id" if action == "MUTE" else "followee_id"] = value
            elif need == "query":
                args["query"] = value

        if action != "DO_NOTHING" and "content" in leaf.needs:
            intent, intent_record = self.build_intent(obs, state, emotion, rng, target_post)
            text = self.content.generate(intent)
            record["intent"] = intent_record
            args["quote_content" if action == "QUOTE_POST" else "content"] = text

        record["action"] = action
        record["args"] = args
        self.log.write(record)
        return Decision(action, args, record, (emotion, readout))
