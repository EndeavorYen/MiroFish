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
DEFAULT_BUDGET_PER_ROUND = FULL_MAX_TOKENS  # keeps the full tier reachable
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
_LEADING_THINK_END_RE = re.compile(r"^.*?</think>", re.S)


def clean_generation(text: str) -> str:
    """Drop reasoning blocks (closed or cut off) and collapse whitespace."""

    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # reasoning whose opening tag the server dropped
        text = _LEADING_THINK_END_RE.sub("", text, count=1)
    return " ".join(text.split()).strip("「」\"")


STRONG = 0.2  # stance below STRONG or above 1 - STRONG: the strong template band


def stance_words(stance: float, lang: str = "zh") -> str:
    """Five-level stance wording for the shared and full prompts (#41, #52)."""

    if lang == "en":
        if stance < STRONG:
            return "strongly opposed"
        if stance < 0.4:
            return "opposed or worried"
        if stance <= 0.6:
            return "neutral"
        if stance <= 1 - STRONG:
            return "supportive"
        return "strongly supportive"
    if stance < STRONG:
        return "强烈反对"
    if stance < 0.4:
        return "反对或担忧"
    if stance <= 0.6:
        return "中立"
    if stance <= 1 - STRONG:
        return "支持"
    return "强烈支持"


def template_band(stance: float, by_kind: dict) -> str:
    """neg/neu/pos, or neg_strong/pos_strong at the extremes when the locale has
    them: three bands put moderate support in enthusiastic lines and never
    produced strong opposition (#41)."""

    band = stance_band(stance)
    if band == "neg" and stance < STRONG and "neg_strong" in by_kind:
        return "neg_strong"
    if band == "pos" and stance > 1 - STRONG and "pos_strong" in by_kind:
        return "pos_strong"
    return band


def stance_band(stance: float) -> str:
    if stance < 0.4:
        return "neg"
    if stance > 0.6:
        return "pos"
    return "neu"


# Shared cache and the post-generation check use five levels (#45). 0.1 and
# 0.35 no longer share one generation.
LEVELS = ("neg_strong", "neg", "neu", "pos", "pos_strong")
_EN_FUNCTION = {
    "a", "an", "the", "for", "said", "and", "or", "of", "to", "in", "on",
    "with", "before", "after", "from", "that", "this", "is", "are", "was",
}


def stance_level(stance: float) -> str:
    if stance < STRONG:
        return "neg_strong"
    if stance < 0.4:
        return "neg"
    if stance <= 0.6:
        return "neu"
    if stance <= 1 - STRONG:
        return "pos"
    return "pos_strong"


def level_index(stance: float) -> int:
    return LEVELS.index(stance_level(stance))


def detect_content_lang(text: str) -> str:
    """English when the requirement is mostly Latin letters (#52)."""

    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if latin >= 40 and latin > cjk * 2:
        return "en"
    return "zh"


def usable_topic(text: str) -> bool:
    word = (text or "").strip()
    if len(word) < 2:
        return False
    if word.lower() in _EN_FUNCTION:
        return False
    return True


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


EVAL_SEEDS = frozenset({1, 2, 3, 4, 5})


def load_stance_bank(lang: str = "zh", path: Path | None = None) -> dict[str, Any] | None:
    """Tone examples per stance level, taken from calibration LLM runs (#45).

    ``locales/<lang>_stance_bank.json``: ``{"seeds": [...], "levels": {level:
    [line, ...]}}`` with names replaced by ``{entity}``. None when the file is
    missing or CONTENT_STANCE_BANK=0. A bank built from evaluation seeds is
    refused.
    """

    if path is None:
        if os.environ.get("CONTENT_STANCE_BANK", "1") == "0":
            return None
        path = LOCALES_DIR / f"{lang}_stance_bank.json"
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    overlap = sorted({int(seed) for seed in data.get("seeds", [])} & EVAL_SEEDS)
    if overlap:
        raise ValueError(f"stance bank built from evaluation seeds {overlap}")
    return data


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
    stance_check: dict[str, int] = field(default_factory=dict)

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
            "stance_check": dict(self.stance_check),
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
        lang: str = "zh",
        score_fn: Callable[[str], float] | None = None,
        stance_bank: dict[str, Any] | None = None,
        bank_share: float = 0.5,
    ) -> None:
        self.templates = templates
        self.stance_bank = stance_bank
        self.bank_share = bank_share
        self.llm_fn = llm_fn
        self.budget_per_round = max(0, int(budget_per_round))
        self.top_k_percent = top_k_percent
        self.followers = dict(followers or {})
        self.entities = list(entities or [])
        self.metrics_path = metrics_path
        self.distinct2_warn = distinct2_warn
        self.platform = platform
        self.lang = lang if lang in ("zh", "en") else "zh"
        self.score_fn = score_fn
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
        if usable_topic(intent.topic):
            return intent.topic.strip()
        if intent.target_text:
            snippet = intent.target_text[:16].strip()
            if usable_topic(snippet):
                return snippet
        if self.entities:
            return self.entities[0]
        return "这件事" if getattr(self, "lang", "zh") == "zh" else "this"

    def _vary(self, text: str, rng: random.Random, intent: ContentIntent) -> str:
        for word, options in (self.templates.get("synonyms") or {}).items():
            if word in text and rng.random() < 0.5:
                text = text.replace(word, rng.choice(options), 1)
        prefix = rng.choice(self.templates.get("prefixes") or [""])
        suffix = rng.choice(self.templates.get("suffixes") or [""])
        suffix = suffix.replace("{topicTag}", self._topic(intent).replace(" ", "")[:12])
        return f"{prefix}{text}{(' ' + suffix) if suffix else ''}".strip()

    # -------------------------------------------------------------- tiers

    def template_base(self, intent: ContentIntent, *, skip: str = "") -> str:
        """One formatted line, with no prefix, suffix, or hashtag."""

        rng = _rng("template-base", intent.platform, intent.round_num, intent.persona_ref, intent.kind, skip)
        by_kind = self.templates["templates"].get(intent.kind) or self.templates["templates"]["other"]
        band = template_band(intent.stance, by_kind)
        # High intensity prefers the strong band when the locale has one (#45).
        if intent.intensity >= 0.75 and band == "neg" and "neg_strong" in by_kind:
            band = "neg_strong"
        if intent.intensity >= 0.75 and band == "pos" and "pos_strong" in by_kind:
            band = "pos_strong"
        options = list(by_kind.get(band) or by_kind["neu"])
        bank = self._bank_lines(band)
        if bank and rng.random() < getattr(self, "bank_share", 0.0):
            options = bank  # a tone example at the same level (#45)
        if skip:
            options = [line for line in options if skip not in line] or options
        return rng.choice(options).format(
            topic=self._topic(intent), entity=self._entity_for(intent), target=intent.target_text[:30]
        )

    def _bank_lines(self, level: str) -> list[str]:
        bank = getattr(self, "stance_bank", None)
        if not bank:
            return []
        return list((bank.get("levels") or {}).get(level) or [])

    def _example_line(self, intent: ContentIntent) -> str:
        """One same-level tone example for the shared and full prompts, or ''."""

        lines = self._bank_lines(stance_level(intent.stance))
        if not lines:
            return ""
        line = _rng("example", intent.platform, intent.round_num, intent.persona_ref).choice(lines)
        example = line.format(topic=self._topic(intent), entity=self._entity_for(intent), target="")
        if getattr(self, "lang", "zh") == "en":
            return f"Tone example at the same stance (do not copy it): {example}\n"
        return f"同样立场的语气参考（不要照抄）：{example}\n"

    def template_text(self, intent: ContentIntent, *, skip: str = "") -> str:
        rng = _rng("template", intent.platform, intent.round_num, intent.persona_ref, intent.kind, skip)
        return self._vary(self.template_base(intent, skip=skip), rng, intent)

    def _shared_prompt(self, intent: ContentIntent) -> str:
        lang = getattr(self, "lang", "zh")
        band = stance_words(intent.stance, lang)
        target = f"\n回应的贴文：{intent.target_text[:120]}" if intent.target_text else ""
        if lang == "en":
            reply = f"\nPost you are replying to: {intent.target_text[:120]}" if intent.target_text else ""
            return (
                f"Write one social post (under 40 words) in English, kind \"{intent.kind}\", "
                f"stance {band}, topic: {self._topic(intent)}, about {self._entity_for(intent)}.{reply}\n"
                f"{self._example_line(intent)}"
                "Output only the post."
            )
        return (
            f"用一句社群贴文（40字以内）表达「{intent.kind}」类的发言，立场{band}，"
            f"主题：{self._topic(intent)}，相关对象：{self._entity_for(intent)}。{target}\n"
            f"{self._example_line(intent)}"
            "只输出贴文本身。"
        )

    def _full_prompt(self, intent: ContentIntent) -> str:
        lang = getattr(self, "lang", "zh")
        band = stance_words(intent.stance, lang)
        if lang == "en":
            reply = f"\nPost you are replying to: {intent.target_text[:160]}" if intent.target_text else ""
            return (
                f"You are {intent.agent_name}. Persona: {intent.persona[:300]}\n"
                f"Write one social post in English (under 60 words), kind \"{intent.kind}\", "
                f"stance {band}, intensity {intent.intensity:.1f} (0-1), "
                f"topic: {self._topic(intent)}.{reply}\n"
                f"{self._example_line(intent)}"
                "Keep the real names from the event. Output only the post."
            )
        target = f"\n你要回应的贴文：{intent.target_text[:160]}" if intent.target_text else ""
        return (
            f"你是{intent.agent_name}。人设：{intent.persona[:300]}\n"
            f"用你的口吻写一则社群贴文（60字以内），发言类型「{intent.kind}」，立场{band}，"
            f"情绪强度{intent.intensity:.1f}（0-1），主题：{self._topic(intent)}。{target}\n"
            f"{self._example_line(intent)}"
            "保留事件里的具体人名、机构或数字。只输出贴文本身。"
        )

    def _note_stance(self, intent: ContentIntent, text: str) -> str:
        """Zero-decode check. One repair when the text is more than one level off (#45)."""

        if self.score_fn is None:
            return text
        try:
            actual = float(self.score_fn(text))
        except Exception as error:  # noqa: BLE001 - a failed check keeps the text
            logger.warning("stance check failed: %s", error)
            return text
        intended = level_index(intent.stance)
        got = level_index(actual)
        key = f"{LEVELS[intended]}->{LEVELS[got]}"
        with self._lock:
            if self._round is not None:
                self._round.stance_check[key] = self._round.stance_check.get(key, 0) + 1
        if abs(intended - got) <= 1:
            return text
        return self.template_base(intent, skip=text)

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
        band = stance_level(intent.stance)
        bucket = (intent.round_num, intent.kind, band, intent.target_ref)
        waited: threading.Event | None = None
        cached_text: str | None = None
        cached_stats: RoundStats | None = None
        cached_slot: int | None = None
        owner: threading.Event | None = None
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
                        cached_text = self._shared_cache[bucket]
                        stats.cache_hits += 1
                        stats.tiers["shared"] += 1
                        cached_stats = stats
                        cached_slot = len(stats.texts)
                        stats.texts.append("")
                        break
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
            if cached_text is not None:
                break
            waiter.wait(timeout=60)
            waited = waiter

        if cached_text is not None:
            repaired = self._note_stance(intent, cached_text)
            text = self._vary(repaired, _rng("shared", intent.persona_ref, *bucket), intent)
            with self._lock:
                if cached_stats is not None and cached_slot is not None and cached_slot < len(cached_stats.texts):
                    cached_stats.texts[cached_slot] = text
            return text

        spent = 0
        raw = ""
        text = ""
        canonical = ""
        generated_tier = tier
        published = False
        try:
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
                        text = raw
                canonical = ""
                if not text:
                    generated_tier = "template"
                    tier = "template"
                    text = self.template_base(intent)
                else:
                    generated_tier = tier
                canonical = self._note_stance(intent, text)
                text = (
                    self._vary(canonical, _rng("shared", intent.persona_ref, *bucket), intent)
                    if tier == "shared"
                    else canonical
                )
            finally:
                with self._lock:
                    self._remaining += reserve - spent

            with self._lock:
                if owner is not None:
                    # The bucket keeps the repaired base, never a per-agent variant.
                    if raw and canonical:
                        self._shared_cache[bucket] = canonical
                    else:
                        self._failed_buckets.add(bucket)
                    if self._inflight.get(bucket) is owner:
                        del self._inflight[bucket]
                    owner.set()
                self._rollover(intent.round_num)
                stats = self._round
                if generated_tier in ("shared", "full"):
                    stats.decode_tokens[generated_tier] += spent
                stats.tiers[tier] += 1
                stats.texts.append(text)
            published = True
            return text
        finally:
            if owner is not None and not published:
                with self._lock:
                    self._failed_buckets.add(bucket)
                    if self._inflight.get(bucket) is owner:
                        del self._inflight[bucket]
                    owner.set()


def system_one_stance_score(text: str) -> float:
    """Zero-decode stance in 0..1, the same five labels the prep path uses (#45)."""

    from ..services.prep_structured import STANCE5, _level
    from ..system_one.client import get_system_one_client
    from ..system_one.models import ScoreQuestion, SystemOneRequest

    answer = get_system_one_client().ask(
        SystemOneRequest(
            state=f"貼文：{text[:300]}",
            questions={"s": ScoreQuestion(instructions="這則貼文的立場是什麼？", criteria=STANCE5)},
        )
    ).answers["s"]
    return _level(answer.score, len(STANCE5))


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
    score_fn: Callable[[str], float] | None = None,
) -> TieredContentProvider:
    """Provider from env settings and the simulation config.

    CONTENT_MODE=tiered (default) or template; CONTENT_DECODE_BUDGET_PER_ROUND;
    CONTENT_TOP_K_PERCENT; CONTENT_LANG (zh|en).
    """

    mode = os.environ.get("CONTENT_MODE", "tiered").strip().lower()
    lang = os.environ.get("CONTENT_LANG", "").strip().lower()
    if lang not in ("zh", "en"):
        lang = detect_content_lang(str(config.get("simulation_requirement") or ""))
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
    if score_fn is None and os.environ.get("CONTENT_STANCE_CHECK", "1") != "0":
        score_fn = system_one_stance_score
    return TieredContentProvider(
        templates=load_templates(lang),
        llm_fn=llm_fn if mode == "tiered" else None,
        # Per platform and round, for every CONTENT_MODE=tiered run. With
        # agents active as often as on the LLM path (#34), 600 let content
        # decode reach ~11% of the LLM path's. DEFAULT_BUDGET_PER_ROUND fits
        # one full-tier (influential) post or one shared generation a round.
        budget_per_round=int(os.environ.get("CONTENT_DECODE_BUDGET_PER_ROUND", str(DEFAULT_BUDGET_PER_ROUND))),
        top_k_percent=float(os.environ.get("CONTENT_TOP_K_PERCENT", "10")),
        followers=followers,
        entities=entities,
        metrics_path=os.path.join(simulation_dir, f"content_metrics_{platform}.jsonl"),
        platform=platform,
        lang=lang,
        score_fn=score_fn,
        stance_bank=load_stance_bank(lang),
        bank_share=float(os.environ.get("CONTENT_BANK_SHARE", "0.5")),
    )
