"""ContentProvider interface (#9) and the minimal template provider.

System One decides *that* an agent speaks and with what intent; a
ContentProvider turns the intent into text. #10 adds the tiered providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ContentIntent:
    """What to say, decided without decode.

    ``stance`` and ``intensity`` are 0..1 (System One ``score``). ``kind`` is a
    dialogue kind from taxonomy.yaml. ``target_ref`` is the post the text
    responds to, if any.
    """

    kind: str
    stance: float
    intensity: float
    target_ref: str | None
    persona_ref: int
    platform: str = ""
    round_num: int = 0
    agent_name: str = ""
    persona: str = ""
    target_text: str = ""
    topic: str = ""
    emotion: dict[str, float] = field(default_factory=dict)


class ContentProvider(Protocol):
    def generate(self, intent: ContentIntent) -> str: ...


_STANCE_WORDS = ["非常反對", "不太認同", "還在觀望", "蠻認同", "非常支持"]
_KIND_TEMPLATES = {
    "opinion": "我對{topic}的看法：{stance}。",
    "support": "{stance}！{topic}值得肯定。",
    "criticism": "{topic}問題不少，我{stance}。",
    "question": "{topic}到底怎麼回事？有沒有人能說明一下？",
    "worry": "有點擔心{topic}，{stance}。",
    "rumor": "聽說{topic}還有內情，大家怎麼看？",
    "clarification": "澄清一下：關於{topic}，目前資訊不完整，先別急著下結論。",
    "humor": "{topic}這件事，真的讓人哭笑不得。",
}


def stance_word(stance: float) -> str:
    index = min(len(_STANCE_WORDS) - 1, max(0, int(round(stance * (len(_STANCE_WORDS) - 1)))))
    return _STANCE_WORDS[index]


class TemplateContentProvider:
    """Placeholder until #10: one template per dialogue kind."""

    def generate(self, intent: ContentIntent) -> str:
        topic = intent.topic or (intent.target_text[:20] if intent.target_text else "這件事")
        template = _KIND_TEMPLATES.get(intent.kind, "關於{topic}，{stance}。")
        text = template.format(topic=topic, stance=stance_word(intent.stance))
        if intent.intensity >= 0.75:
            text += "！"
        return text
