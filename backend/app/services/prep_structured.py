"""Decode-free preparation: ontology, profiles and simulation config (#11).

Each step replaces an LLM JSON generation with System One questions over
fixed options plus templates:

* ontology   (ONTOLOGY_MODE=template)   — pick a preset with ``choice``,
  keep optional entity types with ``noul``;
* profile    (PROFILE_MODE=structured)  — person/organisation, age band,
  gender, profession, stance, risk appetite, activity, interests by
  ``choice``/``score``/``noul``; bio and persona from templates;
* sim config (SIM_CONFIG_MODE=structured) — activity level, active hours,
  stance and influence per agent by ``choice``/``score``; hot topics and
  initial posts from the seed text; time settings are the deterministic
  defaults.

The LLM paths stay available (``llm`` modes).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..system_one.models import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

TEMPLATES_PATH = Path(__file__).with_name("ontology_templates.json")
ENTITY_TYPE_COUNT = 10
MAX_SOURCE_TARGETS = 10
STATE_CHARS = 1500

AGE_BANDS = {"18_24": 21, "25_34": 30, "35_44": 40, "45_54": 50, "55_64": 60, "65_plus": 70}
PROFESSIONS = {
    "student": "學生",
    "office_worker": "上班族",
    "driver": "司機或運輸業",
    "engineer": "工程師或科技業",
    "teacher": "教師",
    "civil_servant": "公務員",
    "business_owner": "自營業者或商人",
    "retiree": "退休人士",
    "journalist": "媒體工作者",
    "academic": "學者或研究人員",
    "healthcare": "醫療人員",
    "other": "其他",
}
TOPICS = {
    "transport": "交通",
    "technology": "科技",
    "economy": "經濟",
    "environment": "環境",
    "safety": "公共安全",
    "policy": "政策法規",
    "jobs": "就業與生計",
    "privacy": "隱私",
    "education": "教育",
    "health": "健康",
}
LEVEL5 = ["非常低", "偏低", "中等", "偏高", "非常高"]
STANCE5 = ["強烈反對", "反對", "中立", "支持", "強烈支持"]
ACTIVE_PATTERNS = {
    "office_hours": ("主要在上班時間活動（機構帳號）", list(range(9, 18))),
    "daytime_evening": ("白天與晚上都會上網", [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23]),
    "evening": ("主要在晚上活動", [12, 13, 19, 20, 21, 22, 23]),
    "all_day": ("全天候活動（如媒體）", list(range(7, 24))),
    "night_owl": ("夜貓子，深夜最活躍", [20, 21, 22, 23, 0, 1, 2]),
}
MBTI = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP",
]


@lru_cache(maxsize=1)
def load_templates() -> dict[str, Any]:
    return json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))


def _rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _ask(client, state: str, questions: dict[str, Any]):
    return client.ask(SystemOneRequest(state=state, questions=questions)).answers


def _level(score: float, levels: int) -> float:
    return max(0.0, min(1.0, score / (levels - 1)))


# ------------------------------------------------------------------ ontology


def _round_robin_pairs(sources: list[str], targets: list[str]) -> list[dict[str, str]]:
    """Up to MAX_SOURCE_TARGETS pairs, one target per source per pass, so
    every source type (media, residents, fallbacks) gets a pair before any
    source gets a second one."""

    queues = {s: [t for t in targets if t != s] for s in sources}
    pairs: list[dict[str, str]] = []
    while len(pairs) < MAX_SOURCE_TARGETS and any(queues.values()):
        for source in sources:
            if queues[source] and len(pairs) < MAX_SOURCE_TARGETS:
                pairs.append({"source": source, "target": queues[source].pop(0)})
    return pairs


def template_ontology(client, document_text: str, requirement: str) -> dict[str, Any]:
    data = load_templates()
    templates = data["templates"]
    types = data["entity_types"]
    state = f"模擬需求：{requirement}\n素材：{document_text[:STATE_CHARS]}"
    preset = _ask(
        client,
        state,
        {
            "preset": ChoiceQuestion(
                instructions="這份素材最接近哪一類社群輿論情境？",
                criteria={k: v["description"] for k, v in templates.items()},
            )
        },
    )["preset"]
    template = templates[preset.choice]
    core = list(template["core"])
    optional = [t for t in template["optional"] if t not in core]
    answers = _ask(
        client,
        state,
        {
            t: NoulQuestion(
                instructions=f"素材中是否有可以歸類為「{types[t]['description']}」的具體行為者？"
            )
            for t in optional
        },
    ) if optional else {}
    ranked = sorted(optional, key=lambda t: (-answers[t].noul, optional.index(t)))
    slots = ENTITY_TYPE_COUNT - 2 - len(core)
    chosen = core + ranked[: max(0, slots)]
    names = chosen + ["Person", "Organization"]

    entity_types = [
        {"name": name, "description": types[name]["description"], "attributes": [], "examples": []}
        for name in names
    ]
    kinds = {name: types[name]["kind"] for name in names}

    def allowed(side: str) -> list[str]:
        return [n for n in names if side == "any" or kinds[n] == side]

    edge_types = []
    for edge in data["edge_types"]:
        pairs = _round_robin_pairs(allowed(edge["source"]), allowed(edge["target"]))
        if pairs:
            edge_types.append(
                {
                    "name": edge["name"],
                    "description": edge["description"],
                    "source_targets": pairs,
                    "attributes": [],
                }
            )
    return {
        "entity_types": entity_types,
        "edge_types": edge_types,
        "analysis_summary": (
            f"模板本體：{preset.choice}（System One 信心 {preset.confidence:.2f}）；"
            f"可選類型保留 {', '.join(ranked[: max(0, slots)])}。"
        ),
        "template": preset.choice,
        "template_probabilities": preset.probabilities,
        "optional_scores": {t: round(answers[t].noul, 4) for t in optional},
    }


# ------------------------------------------------------------------- profile


def _entity_state(name: str, entity_type: str, summary: str, context: str) -> str:
    return f"實體：{name}（類型：{entity_type}）\n摘要：{summary[:400]}\n相關資訊：{context[:800]}"


def structured_profile(
    client, name: str, entity_type: str, summary: str, context: str, seed: str = ""
) -> dict[str, Any]:
    state = _entity_state(name, entity_type, summary, context)
    kind = _ask(
        client,
        state,
        {
            "kind": ChoiceQuestion(
                instructions=f"「{name}」是一個人，還是一個機構／團體的帳號？",
                criteria={"person": "一個人", "organization": "機構、團體或媒體帳號"},
            )
        },
    )["kind"].choice
    questions: dict[str, Any] = {
        "stance": ScoreQuestion(instructions=f"「{name}」對素材中主要事件的立場？", criteria=STANCE5),
        "risk": ScoreQuestion(instructions=f"「{name}」面對新事物的風險偏好？", criteria=LEVEL5),
        "activity": ScoreQuestion(instructions=f"「{name}」在社群上發言的活躍程度？", criteria=LEVEL5),
    }
    for key, label in TOPICS.items():
        questions[f"topic_{key}"] = NoulQuestion(instructions=f"「{name}」會關心「{label}」議題嗎？")
    if kind == "person":
        questions["age"] = ChoiceQuestion(
            instructions=f"「{name}」最可能的年齡區間？",
            criteria={k: k.replace("_", "-").replace("plus", "+") for k in AGE_BANDS},
        )
        questions["gender"] = ChoiceQuestion(
            instructions=f"「{name}」的性別？", criteria={"male": "男性", "female": "女性"}
        )
        questions["profession"] = ChoiceQuestion(
            instructions=f"「{name}」最可能的職業？", criteria=PROFESSIONS
        )
        questions["mbti_ei"] = ChoiceQuestion(
            instructions=f"「{name}」比較外向還是內向？", criteria={"E": "外向", "I": "內向"}
        )
    answers = _ask(client, state, questions)
    stance = _level(answers["stance"].score, len(STANCE5))
    risk = _level(answers["risk"].score, len(LEVEL5))
    activity = _level(answers["activity"].score, len(LEVEL5))
    topics = sorted(TOPICS, key=lambda k: -answers[f"topic_{k}"].noul)
    interested = [TOPICS[k] for k in topics[:3] if answers[f"topic_{k}"].noul >= 0.3] or [TOPICS[topics[0]]]
    stance_word = STANCE5[int(round(stance * 4))]
    rng = _rng(seed, name)
    if kind == "person":
        age = AGE_BANDS[answers["age"].choice]
        gender = answers["gender"].choice
        profession = PROFESSIONS[answers["profession"].choice]
        ei = answers["mbti_ei"].choice
        mbti = rng.choice([m for m in MBTI if m.startswith(ei)])
        bio = f"{name}｜{profession}｜關注{'、'.join(interested)}"
        persona = (
            f"{name}，約 {age} 歲，{profession}。對目前事件{stance_word}，"
            f"風險偏好{LEVEL5[int(round(risk * 4))]}，發言活躍度{LEVEL5[int(round(activity * 4))]}。"
            f"關注{'、'.join(interested)}。{summary[:200]}"
        )
    else:
        age, gender, mbti, profession = 30, "other", "ISTJ", entity_type
        bio = f"{name} 官方帳號｜{entity_type}"
        persona = (
            f"{name} 是{entity_type}的官方帳號，對目前事件{stance_word}，"
            f"發言活躍度{LEVEL5[int(round(activity * 4))]}，主要關注{'、'.join(interested)}。"
            f"{summary[:200]}"
        )
    return {
        "bio": bio,
        "persona": persona,
        "age": age,
        "gender": gender,
        "mbti": mbti,
        "country": "中国",  # same spelling as the LLM and rule paths
        "profession": profession,
        "interested_topics": interested,
        "stance_score": round(stance, 4),
        "risk_score": round(risk, 4),
        "activity_score": round(activity, 4),
    }


# ---------------------------------------------------------------- sim config


ACTIVITY_FLOOR = 0.3
# Share of agents the runner may activate per hour. The LLM time config picks
# about half the agents (golden: 8-9 of 16); the generic fallback (n/15 to
# n/5) activated 1-5 of 16, a third of the LLM path's activity (#35).
AGENTS_PER_HOUR_SHARE = (0.4, 0.6)


def structured_time_config(num_entities: int) -> dict[str, Any]:
    """Time config without an LLM: the usual daily rhythm, activation share
    matched to what the LLM time config chooses."""

    low = max(1, round(num_entities * AGENTS_PER_HOUR_SHARE[0]))
    high = max(low + 1, round(num_entities * AGENTS_PER_HOUR_SHARE[1]))
    return {
        "total_simulation_hours": 72,
        "minutes_per_round": 60,
        "agents_per_hour_min": min(low, max(1, num_entities - 1)),
        "agents_per_hour_max": min(high, max(2, num_entities)),
        "peak_hours": [19, 20, 21, 22],
        "off_peak_hours": [0, 1, 2, 3, 4, 5],
        "morning_hours": [6, 7, 8],
        "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "reasoning": "structured: daily rhythm, 40-60% of agents per hour",
    }


def structured_agent_config(client, name: str, entity_type: str, summary: str) -> dict[str, Any]:
    state = f"實體：{name}（類型：{entity_type}）\n摘要：{summary[:400]}"
    answers = _ask(
        client,
        state,
        {
            "activity": ScoreQuestion(instructions=f"「{name}」在社群上的整體活躍度？", criteria=LEVEL5),
            "influence": ScoreQuestion(instructions=f"「{name}」發言的影響力？", criteria=LEVEL5),
            "stance": ScoreQuestion(instructions=f"「{name}」對主要事件的立場？", criteria=STANCE5),
            "hours": ChoiceQuestion(
                instructions=f"「{name}」通常什麼時段上網發言？",
                criteria={k: v[0] for k, v in ACTIVE_PATTERNS.items()},
            ),
        },
    )
    activity = _level(answers["activity"].score, len(LEVEL5))
    influence = _level(answers["influence"].score, len(LEVEL5))
    stance = _level(answers["stance"].score, len(STANCE5))
    stance_label = "opposing" if stance < 0.35 else "supportive" if stance > 0.65 else "neutral"
    return {
        # activity_level gates whether an agent is a candidate each round.
        # The readout places most entities at the low end, and 0.1 + 0.8a gave
        # a golden mean of 0.18 against 0.44 from the LLM config, so agents
        # were rarely active; the floor keeps quiet entities participating.
        "activity_level": round(ACTIVITY_FLOOR + (0.9 - ACTIVITY_FLOOR) * activity, 3),
        "posts_per_hour": round(0.1 + 0.9 * activity, 3),
        "comments_per_hour": round(0.2 + 1.3 * activity, 3),
        "active_hours": ACTIVE_PATTERNS[answers["hours"].choice][1],
        "response_delay_min": 5,
        "response_delay_max": 60,
        "sentiment_bias": round(stance * 2 - 1, 3),
        "stance": stance_label,
        "influence_weight": round(0.5 + 2.5 * influence, 3),
    }


_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]")
_STOP = set("的了是在和與及並而也就都將被把對從向為以於等這那有個們")


def seed_hot_topics(document_text: str, entity_names: list[str], limit: int = 8) -> list[str]:
    """Frequent content words of the seed plus the most mentioned entities."""

    import jieba

    jieba.setLogLevel(60)
    words = [
        w for w in jieba.cut(document_text)
        if len(w) >= 2 and re.fullmatch(r"[一-鿿A-Za-z0-9]+", w) and not set(w) & _STOP
    ]
    counts = Counter(words)
    mentioned = sorted(
        (n for n in entity_names if n in document_text),
        key=lambda n: -document_text.count(n),
    )
    topics = [n for n in mentioned[:3]]
    for word, _ in counts.most_common(limit * 3):
        if len(topics) >= limit:
            break
        if not any(word in t or t in word for t in topics):
            topics.append(word)
    return topics


def structured_event_config(
    client,
    document_text: str,
    requirement: str,
    entity_types: list[str],
    entity_names: list[str],
    max_posts: int = 3,
) -> dict[str, Any]:
    sentences = [s.strip() for s in _SENTENCE_RE.findall(document_text) if len(s.strip()) >= 15]
    posts = []
    types = [t for t in dict.fromkeys(entity_types) if t][:25]
    for sentence in sentences[:max_posts]:
        poster_type = types[0] if types else "Person"
        if len(types) >= 2:
            poster_type = _ask(
                client,
                f"貼文內容：{sentence}",
                {
                    "poster": ChoiceQuestion(
                        instructions="哪一類帳號最可能第一個發出這則消息？",
                        criteria={t: t for t in types},
                    )
                },
            )["poster"].choice
        posts.append({"content": sentence[:200], "poster_type": poster_type})
    topics = seed_hot_topics(document_text, entity_names)
    return {
        "hot_topics": topics,
        "narrative_direction": f"圍繞{'、'.join(topics[:3])}的討論，各方立場逐步分化。" if topics else "",
        "initial_posts": posts,
        "reasoning": "structured: seed sentences + System One poster type",
    }
