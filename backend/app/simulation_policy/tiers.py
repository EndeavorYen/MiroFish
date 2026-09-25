"""Tiered content generation for System One agents (#10).

System One picks the intent; this module decides how the words are made:

1. template  — kind x stance band templates from locales/<lang>.json
               (``simContent``), filled with a seed entity / topic / target
               snippet; no decode.
2. shared    — agents in one round with the same (kind, stance band, target)
               share one small-model generation; each agent gets a variant
               (prefix, suffix, synonym swaps). Results are cached per bucket.
3. full      — the most influential k% of agents (by followers) get a
               persona-conditioned small-model generation.

``CONTENT_DECODE_BUDGET_PER_ROUND`` caps decode tokens per round; once it is
spent, every agent falls back to templates. Per-round tier counts, decode
tokens and the distinct-2 ratio of generated posts go to
``content_metrics.jsonl``; a distinct-2 below the threshold logs a warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .content import ContentIntent

logger = logging.getLogger("mirofish.content_tiers")

LOCALES_DIR = Path(__file__).resolve().parents[3] / "locales"
LlmFn = Callable[[str, int], tuple[str, int]]  # (prompt, max_tokens) -> (text, decode tokens)

SHARED_MAX_TOKENS = 80
FULL_MAX_TOKENS = 120
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
_LEADING_THINK_END_RE = re.compile(r"^.*?</think>", re.S)


def clean_generation(text: str) -> str:
    """Drop reasoning blocks (closed or cut off) and collapse whitespace."""

    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # reasoning whose opening tag the server dropped
        text = _LEADING_THINK_END_RE.sub("", text, count=1)
    return " ".join(text.split()).strip("「」\"")


def stance_band(stance: float) -> str:
    if stance < 0.4:
        return "neg"
    if stance > 0.6:
        return "pos"
    return "neu"


def distinct_2(texts: list[str]) -> float:
    """Unique character bigrams / all bigrams over the texts (0 if none)."""

    grams: list[str] = []
    for text in texts:
        compact = "".join(text.split())
        grams.extend(compact[i : i + 2] for i in range(len(compact) - 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def load_templates(lang: str = "zh") -> dict[str, Any]:
    path = LOCALES_DIR / f"{lang}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["simContent"]


def _rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


@dataclass
class RoundStats:
    round_num: int
    tiers: dict[str, int] = field(default_factory=lambda: {"template": 0, "shared": 0, "full": 0})
    decode_tokens: dict[str, int] = field(default_factory=lambda: {"shared": 0, "full": 0})
    cache_hits: int = 0
    texts: list[str] = field(default_factory=list)

    def row(self, platform: str) -> dict[str, Any]:
        return {
            "platform": platform,
            "round": self.round_num,
            "tiers": dict(self.tiers),
            "decode_tokens": dict(self.decode_tokens),
            "decode_tokens_total": sum(self.decode_tokens.values()),
            "shared_cache_hits": self.cache_hits,
            "posts": len(self.texts),
            "distinct_2": round(distinct_2(self.texts), 4),
        }


class TieredContentProvider:
    def __init__(
        self,
        *,
        templates: dict[str, Any],
        llm_fn: LlmFn | None = None,
        budget_per_round: int = 600,
        top_k_percent: float = 10.0,
        followers: dict[int, int] | None = None,
        entities: list[str] | None = None,
        metrics_path: str | None = None,
        distinct2_warn: float = 0.4,
        platform: str = "",
    ) -> None:
        self.templates = templates
        self.llm_fn = llm_fn
        self.budget_per_round = max(0, int(budget_per_round))
        self.top_k_percent = top_k_percent
        self.followers = dict(followers or {})
        self.entities = list(entities or [])
        self.metrics_path = metrics_path
        self.distinct2_warn = distinct2_warn
        self.platform = platform
        self._lock = threading.Lock()
        self._round: RoundStats | None = None
        self._remaining = self.budget_per_round
        self._shared_cache: dict[tuple, str] = {}
        # One model call per bucket: later agents wait for the first one.
        self._inflight: dict[tuple, threading.Event] = {}
        self._failed_buckets: set[tuple] = set()
        self.history: list[dict[str, Any]] = []
        ranked = sorted(self.followers, key=lambda a: (-self.followers[a], a))
        count = max(1, int(round(len(ranked) * top_k_percent / 100))) if ranked else 0
        self._influential = set(ranked[:count]) if top_k_percent > 0 else set()

    # ------------------------------------------------------------ helpers

    def influential(self, agent_id: int) -> bool:
        return agent_id in self._influential

    def _entity_for(self, intent: ContentIntent) -> str:
        for name in self.entities:
            if name and name in (intent.target_text or ""):
                return name
        for name in self.entities:
            if name and name in (intent.topic or ""):
                return name
        return self.entities[0] if self.entities else (intent.topic or "")

    def _topic(self, intent: ContentIntent) -> str:
        if intent.topic:
            return intent.topic
        if intent.target_text:
            return intent.target_text[:16]
        return self.entities[0] if self.entities else "这件事"

    def _vary(self, text: str, rng: random.Random, intent: ContentIntent) -> str:
        for word, options in (self.templates.get("synonyms") or {}).items():
            if word in text and rng.random() < 0.5:
                text = text.replace(word, rng.choice(options), 1)
        prefix = rng.choice(self.templates.get("prefixes") or [""])
        suffix = rng.choice(self.templates.get("suffixes") or [""])
        suffix = suffix.replace("{topicTag}", self._topic(intent).replace(" ", "")[:12])
        return f"{prefix}{text}{(' ' + suffix) if suffix else ''}".strip()

    # -------------------------------------------------------------- tiers

    def template_text(self, intent: ContentIntent) -> str:
        rng = _rng("template", intent.platform, intent.round_num, intent.persona_ref, intent.kind)
        by_kind = self.templates["templates"].get(intent.kind) or self.templates["templates"]["other"]
        options = by_kind.get(stance_band(intent.stance)) or by_kind["neu"]
        text = rng.choice(options).format(
            topic=self._topic(intent), entity=self._entity_for(intent), target=intent.target_text[:30]
        )
        return self._vary(text, rng, intent)

    def _shared_prompt(self, intent: ContentIntent) -> str:
        band = {"neg": "反对或担忧", "neu": "中立", "pos": "支持"}[stance_band(intent.stance)]
        target = f"\n回应的贴文：{intent.target_text[:120]}" if intent.target_text else ""
        return (
            f"用一句社群贴文（40字以内）表达「{intent.kind}」类的发言，立场{band}，"
            f"主题：{self._topic(intent)}，相关对象：{self._entity_for(intent)}。{target}\n"
            "只输出贴文本身。"
        )

    def _full_prompt(self, intent: ContentIntent) -> str:
        band = {"neg": "反对或担忧", "neu": "中立", "pos": "支持"}[stance_band(intent.stance)]
        target = f"\n你要回应的贴文：{intent.target_text[:160]}" if intent.target_text else ""
        return (
            f"你是{intent.agent_name}。人设：{intent.persona[:300]}\n"
            f"用你的口吻写一则社群贴文（60字以内），发言类型「{intent.kind}」，立场{band}，"
            f"情绪强度{intent.intensity:.1f}（0-1），主题：{self._topic(intent)}。{target}\n"
            "保留事件里的具体人名、机构或数字。只输出贴文本身。"
        )

    # ------------------------------------------------------------ metrics

    def _rollover(self, round_num: int) -> None:
        if self._round is not None and self._round.round_num == round_num:
            return
        self.flush()
        self._round = RoundStats(round_num)
        self._remaining = self.budget_per_round

    def flush(self) -> None:
        if self._round is None:
            return
        row = self._round.row(self.platform)
        self.history.append(row)
        if row["posts"] >= 5 and row["distinct_2"] < self.distinct2_warn:
            logger.warning(
                "content collapse: round %s distinct-2 %.3f < %.2f",
                row["round"], row["distinct_2"], self.distinct2_warn,
            )
        if self.metrics_path:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._round = None

    # --------------------------------------------------------------- main

    def _choose_tier(self, intent: ContentIntent, bucket: tuple) -> str:
        if self.llm_fn is None:
            return "template"
        if self.influential(intent.persona_ref) and self._remaining >= FULL_MAX_TOKENS:
            return "full"
        if bucket in self._failed_buckets:
            return "template"  # the shared call for this bucket already failed
        if bucket in self._shared_cache or self._remaining >= SHARED_MAX_TOKENS:
            return "shared"
        return "template"

    def generate(self, intent: ContentIntent) -> str:
        band = stance_band(intent.stance)
        bucket = (intent.round_num, intent.kind, band, intent.target_ref)
        waited: threading.Event | None = None
        while True:
            with self._lock:
                self._rollover(intent.round_num)
                tier = self._choose_tier(intent, bucket)
                waiter = self._inflight.get(bucket) if tier == "shared" else None
                if waiter is not None and waiter is not waited:
                    pass  # another agent is generating this bucket; wait below
                else:
                    stats = self._round
                    if tier == "shared" and bucket in self._shared_cache:
                        text = self._vary(
                            self._shared_cache[bucket],
                            _rng("shared", intent.persona_ref, *bucket),
                            intent,
                        )
                        stats.cache_hits += 1
                        stats.tiers["shared"] += 1
                        stats.texts.append(text)
                        return text
                    if waiter is not None:
                        tier = "template"  # the owner stalled past the wait
                    # Reserve the budget and claim the bucket under the lock.
                    reserve = {"full": FULL_MAX_TOKENS, "shared": SHARED_MAX_TOKENS}.get(tier, 0)
                    self._remaining -= reserve
                    owner = None
                    if tier == "shared":
                        owner = threading.Event()
                        self._inflight[bucket] = owner
                    break
            waiter.wait(timeout=60)
            waited = waiter

        spent = 0
        raw = ""
        text = ""
        try:
            if tier != "template":
                prompt = self._full_prompt(intent) if tier == "full" else self._shared_prompt(intent)
                try:
                    raw, spent = self.llm_fn(prompt, reserve)
                    raw = clean_generation(raw)
                except Exception as error:  # noqa: BLE001 - degrade, never fail a round
                    logger.warning("content generation failed, using template: %s", error)
                    raw, spent = "", 0
                if raw:
                    text = (
                        self._vary(raw, _rng("shared", intent.persona_ref, *bucket), intent)
                        if tier == "shared"
                        else raw
                    )
            if not text:
                generated_tier = tier
                tier = "template"
                text = self.template_text(intent)
            else:
                generated_tier = tier
        finally:
            with self._lock:
                self._remaining += reserve - spent
                if owner is not None:
                    if raw and bucket not in self._shared_cache:
                        self._shared_cache[bucket] = raw
                    elif not raw:
                        self._failed_buckets.add(bucket)
                    if self._inflight.get(bucket) is owner:
                        del self._inflight[bucket]
                    owner.set()  # always wake waiters, even if replaced

        with self._lock:
            self._rollover(intent.round_num)
            stats = self._round
            if generated_tier in ("shared", "full"):
                stats.decode_tokens[generated_tier] += spent
            stats.tiers[tier] += 1
            stats.texts.append(text)
        return text


def openai_llm_fn(stage_note: str = "content") -> LlmFn:
    """Small-model call through the configured OpenAI-compatible LLM.

    Usage is recorded by create_chat_completion under the active stage.
    """

    from openai import OpenAI

    from ..config import Config
    from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text

    # A stalled server must not stall a round: short timeout, no retries
    # (a failed call falls back to a template).
    client = OpenAI(
        api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL, timeout=30, max_retries=0
    )

    def call(prompt: str, max_tokens: int) -> tuple[str, int]:
        response = create_chat_completion(
            client,
            model=Config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        return extract_chat_completion_text(response), int(getattr(usage, "completion_tokens", 0) or 0)

    return call


def build_tiered_provider(
    platform: str,
    simulation_dir: str,
    config: dict[str, Any],
    *,
    llm_fn: LlmFn | None = None,
) -> TieredContentProvider:
    """Provider from env settings and the simulation config.

    CONTENT_MODE=tiered (default) or template; CONTENT_DECODE_BUDGET_PER_ROUND;
    CONTENT_TOP_K_PERCENT; CONTENT_LANG (zh|en).
    """

    mode = os.environ.get("CONTENT_MODE", "tiered").strip().lower()
    followers = {}
    for agent in config.get("agent_configs", []):
        agent_id = agent.get("agent_id")
        if agent_id is not None:
            weight = agent.get("follower_count")
            if weight is None:
                weight = float(agent.get("influence_weight") or 0) * 1000
            followers[int(agent_id)] = int(weight or 0)
    entities = [a.get("entity_name") for a in config.get("agent_configs", []) if a.get("entity_name")]
    if mode == "tiered" and llm_fn is None:
        llm_fn = openai_llm_fn()
    return TieredContentProvider(
        templates=load_templates(os.environ.get("CONTENT_LANG", "zh")),
        llm_fn=llm_fn if mode == "tiered" else None,
        budget_per_round=int(os.environ.get("CONTENT_DECODE_BUDGET_PER_ROUND", "600")),
        top_k_percent=float(os.environ.get("CONTENT_TOP_K_PERCENT", "10")),
        followers=followers,
        entities=entities,
        metrics_path=os.path.join(simulation_dir, f"content_metrics_{platform}.jsonl"),
        platform=platform,
    )
