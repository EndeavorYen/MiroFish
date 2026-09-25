"""Extractor interface: text episode -> entities and relations.

``LocalGraphStore.add_text_episodes`` hands each episode to an injected
extractor. ``StubExtractor`` is a deterministic lexicon matcher for tests;
the real zero-decode extractor is ``LocalExtractor`` (#7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    entity_type: str
    summary: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedRelation:
    name: str
    source: str
    target: str
    fact: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Extraction:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


class Extractor(Protocol):
    def extract(self, text: str, ontology: dict[str, Any] | None) -> Extraction: ...


_SENTENCE_RE = re.compile(r"[^。！？!?\.\n]+[。！？!?\.]?")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text or "") if s.strip()]


class StubExtractor:
    """Match a fixed ``{name: entity_type}`` lexicon.

    Entities of a type absent from the ontology are dropped. Two entities in
    one sentence get a relation named after the first ontology edge type
    whose ``source_targets`` allow the pair, else ``RELATED_TO``; the
    sentence is the fact.
    """

    def __init__(self, lexicon: dict[str, str] | None = None) -> None:
        self.lexicon = dict(lexicon or {})

    def extract(self, text: str, ontology: dict[str, Any] | None) -> Extraction:
        allowed = {e["name"] for e in (ontology or {}).get("entity_types", [])}
        lexicon = {
            name: etype
            for name, etype in self.lexicon.items()
            if not allowed or etype in allowed
        }
        entities: dict[str, ExtractedEntity] = {}
        relations: list[ExtractedRelation] = []
        for sentence in split_sentences(text):
            present = [name for name in lexicon if name in sentence]
            for name in present:
                entities.setdefault(
                    name, ExtractedEntity(name=name, entity_type=lexicon[name], summary=sentence)
                )
            for i, first in enumerate(present):
                for second in present[i + 1 :]:
                    name, source, target = self._edge(ontology, lexicon, first, second)
                    relations.append(
                        ExtractedRelation(name=name, source=source, target=target, fact=sentence)
                    )
        return Extraction(list(entities.values()), relations)

    @staticmethod
    def _edge(
        ontology: dict[str, Any] | None, lexicon: dict[str, str], first: str, second: str
    ) -> tuple[str, str, str]:
        """Pick the ontology edge type and direction for an entity pair."""

        for source, target in ((first, second), (second, first)):
            for edge in (ontology or {}).get("edge_types", []):
                for pair in edge.get("source_targets", []) or []:
                    if pair.get("source") == lexicon[source] and pair.get("target") == lexicon[target]:
                        return edge["name"], source, target
        return "RELATED_TO", first, second
