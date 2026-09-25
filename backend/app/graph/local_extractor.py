"""Zero-decode entity and relation extraction for the local graph (#7).

Pipeline per episode (the caller has already chunked the document with
``TextProcessor``):

1. Candidates: ``find_candidates`` (jieba person names, organisation-suffix
   spans, quoted names) or, with ``ner="gliner"``, GLiNER multi on CPU.
2. Typing: System One ``choice`` over the ontology entity types plus
   ``none``; ``none`` drops fragments, slogans and generic words.
3. Aliases: pairs whose names, with organisation suffixes removed, share a
   distinctive substring (or one contains the other) are confirmed with
   System One ``noul`` and folded into one canonical name, preferring a name
   already in the graph. Embedding similarity is off by default: with
   multilingual-e5-small, true aliases scored 0.948-0.976 and distinct
   organisations 0.911-0.953 on the golden seed, so it cannot separate them.
4. Relations: for two entities in one sentence, System One ``choice`` over
   the ontology edge types whose ``source_targets`` allow the pair (either
   direction) plus ``none``.
5. Summary: template, the first sentences that mention the entity. An
   optional small-model summary (``EXTRACT_SUMMARY_LLM=1``) is off by
   default.

Only System One readouts are used by default, so no decode step runs.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .candidates import ORG_SUFFIXES, find_candidates
from .embedding import Embedder
from .extractor import ExtractedEntity, ExtractedRelation, Extraction, split_sentences

NONE_KEY = "none"
MAX_SUMMARY_SENTENCES = 3
MAX_CHOICE_TYPES = 25  # 26 labels minus "none"
FALLBACK_TYPES = {"Person", "Organization"}

SummaryFn = Callable[[str, str, list[str]], str]


@dataclass(frozen=True)
class _Typed:
    name: str
    entity_type: str
    confidence: float
    source: str = ""  # candidate source: "person" | "org" | "quoted" | ""


def _common_substrings(a: str, b: str, min_len: int = 2) -> set[str]:
    found = set()
    for i in range(len(a)):
        for j in range(i + min_len, len(a) + 1):
            piece = a[i:j]
            if piece in b:
                found.add(piece)
            else:
                break
    return found


def _strip_suffix(name: str) -> str:
    """Drop a trailing organisation suffix (部門, 委員會 ...) before comparing."""

    for suffix in sorted(ORG_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LocalExtractor:
    def __init__(
        self,
        system_one,
        *,
        embedder: Embedder | None = None,
        ner: Literal["candidates", "gliner"] = "candidates",
        gliner_model: str = "urchade/gliner_multi-v2.1",
        gliner_threshold: float = 0.5,
        merge_cosine: float | None = None,
        merge_min_noul: float = 0.7,
        summary_fn: SummaryFn | None = None,
    ) -> None:
        self.system_one = system_one
        self.embedder = embedder
        self.ner = ner
        self.gliner_model = gliner_model
        self.gliner_threshold = gliner_threshold
        self.merge_cosine = merge_cosine
        self.merge_min_noul = merge_min_noul
        self.summary_fn = summary_fn
        self._gliner = None
        self._gliner_lock = threading.Lock()

    # ----------------------------------------------------------- candidates

    def _entity_types(self, ontology: dict[str, Any] | None) -> dict[str, str]:
        types = {
            e["name"]: (e.get("description") or e["name"])
            for e in (ontology or {}).get("entity_types", [])
        }
        if len(types) > MAX_CHOICE_TYPES:
            raise ValueError(f"ontology has {len(types)} entity types; at most {MAX_CHOICE_TYPES}")
        return types

    def _load_gliner(self):
        with self._gliner_lock:
            return self._load_gliner_locked()

    def _load_gliner_locked(self):
        if self._gliner is None:
            try:
                from gliner import GLiNER
                from gliner.data_processing import WordsSplitter
            except ImportError as error:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "LOCAL_NER=gliner needs the optional packages: gliner, torch, jieba3"
                ) from error
            model = GLiNER.from_pretrained(self.gliner_model)
            # The default splitter is whitespace-based; Chinese needs jieba.
            model.data_processor.words_splitter = WordsSplitter("jieba")
            self._gliner = model
        return self._gliner

    @staticmethod
    def _gliner_label(type_name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", " ", type_name).lower()

    def _gliner_entities(self, text: str, types: dict[str, str]) -> list[_Typed]:
        model = self._load_gliner()
        label_to_type = {self._gliner_label(t): t for t in types}
        found: dict[str, _Typed] = {}
        for sentence in split_sentences(text):
            for item in model.predict_entities(
                sentence, list(label_to_type), threshold=self.gliner_threshold
            ):
                name = item["text"].strip("，。、；：「」 ")
                if len(name) < 2:
                    continue
                typed = _Typed(name, label_to_type[item["label"]], float(item["score"]))
                if name not in found or typed.confidence > found[name].confidence:
                    found[name] = typed
        return list(found.values())

    def _typed_candidates(self, text: str, types: dict[str, str]) -> list[_Typed]:
        sentences = split_sentences(text)
        criteria = dict(types)
        criteria[NONE_KEY] = (
            "不是具名的人或組織：泛稱（如「政府」「監管部門」「市民」）、職稱、口號、"
            "項目名稱、不完整的片段"
        )
        typed: dict[str, _Typed] = {}
        for candidate in find_candidates(text):
            options = candidate.options()
            sentence = next((s for s in sentences if options[0] in s), text[:200])
            name = options[0]
            if len(options) > 1:
                variants = {f"v{i}": option for i, option in enumerate(options)}
                variants[NONE_KEY] = "都不是完整的專有名稱"
                boundary = self.system_one.choice(
                    f"句子：{sentence}",
                    "下列哪一個是句子中邊界正確、完整的專有名稱（人名或機構全名）？",
                    variants,
                )
                if boundary.choice == NONE_KEY:
                    continue
                name = variants[boundary.choice]
            if name in typed:
                continue
            answer = self.system_one.choice(
                f"句子：{sentence}\n候選名稱：{name}",
                f"「{name}」這個名稱本身指的是哪一類實體？"
                "只判斷名稱本身：人名要選人物類型，不要選他所屬組織的類型。"
                "若它不是一個具名的人或組織，選 none。",
                criteria,
            )
            if answer.choice != NONE_KEY:
                source = candidate.source if name in candidate.options() else ""
                typed[name] = _Typed(name, answer.choice, answer.confidence, source)
        return list(typed.values())

    # --------------------------------------------------------------- aliases

    def _merge(
        self,
        typed: list[_Typed],
        known: list[tuple[str, str]],
        sentences: list[str],
    ) -> dict[str, str]:
        """Map every surface name to its canonical name."""

        canonical = {t.name: t.name for t in typed}
        type_of = {name: etype for name, etype in known}
        type_of.update({t.name: t.entity_type for t in typed})
        # Only known names that share a character bigram with a new name can
        # be aliases; this keeps the pool small as the graph grows.
        new_grams = {
            core[i : i + 2]
            for t in typed
            for core in [_strip_suffix(t.name)]
            for i in range(len(core) - 1)
        }
        related_known = [
            name
            for name, _ in known
            # The cosine path must still see every known name (北大/北京大學
            # share no bigram).
            if self.merge_cosine is not None or any(g in _strip_suffix(name) for g in new_grams)
        ]
        pool = list(dict.fromkeys(related_known + [t.name for t in typed]))
        known_names = {name for name, _ in known}

        # A substring shared by 3+ names (東海, 東海市 ...) is not evidence.
        counts: dict[str, int] = {}
        cores = [_strip_suffix(name) for name in pool]
        for i, a in enumerate(cores):
            pieces: set[str] = set()
            for b in cores[i + 1 :]:
                pieces |= _common_substrings(a, b)
            for piece in pieces:
                counts[piece] = counts.get(piece, 0) + 1

        vectors: dict[str, list[float]] = {}
        if self.merge_cosine is not None and self.embedder is not None and pool:
            vectors = dict(zip(pool, self.embedder.embed_documents(pool)))

        source_of = {t.name: t.source for t in typed}

        def kind(name: str) -> str | None:
            """person / org from how the name was found, or its suffix."""

            if source_of.get(name) == "person":
                return "person"
            if source_of.get(name) == "org" or _strip_suffix(name) != name:
                return "org"
            return None

        def compatible(a: str, b: str) -> bool:
            ka, kb = kind(a), kind(b)
            if ka and kb and ka != kb:
                return False  # 王明 vs 王明大學, whatever the ontology types
            ta, tb = type_of.get(a), type_of.get(b)
            if ta == tb:
                return True
            if {ta, tb} == FALLBACK_TYPES:
                return False
            return (ta in FALLBACK_TYPES) != (tb in FALLBACK_TYPES)

        def mentions(name: str) -> str:
            return " ".join(s for s in sentences if name in s)[:300]

        for t in typed:
            name = t.name
            for other in pool:
                if other == name or canonical.get(other, other) == canonical[name]:
                    continue
                if not compatible(name, other):
                    continue
                core_a, core_b = _strip_suffix(name), _strip_suffix(other)
                shared = {
                    p for p in _common_substrings(core_a, core_b) if counts.get(p, 0) < 3
                }
                contained = bool(core_a and core_b) and (core_a in core_b or core_b in core_a)
                close = (
                    self.merge_cosine is not None
                    and name in vectors
                    and other in vectors
                    and _cosine(vectors[name], vectors[other]) >= self.merge_cosine
                )
                if not (shared or contained or close):
                    continue
                answer = self.system_one.noul(
                    f"名稱一：{name}（{type_of.get(name)}）\n"
                    f"名稱二：{other}（{type_of.get(other)}）\n"
                    f"上下文：{mentions(name)} {mentions(other)}",
                    "這兩個名稱在文中指的是同一個實體嗎？",
                )
                if answer.noul >= self.merge_min_noul:
                    target = canonical.get(other, other)
                    keep = self._pick_canonical(canonical[name], target, known_names)
                    drop = target if keep != target else canonical[name]
                    for key, value in list(canonical.items()):
                        if value == drop:
                            canonical[key] = keep
                    canonical.setdefault(other, keep)
                    canonical[name] = keep
                    break
        return canonical

    @staticmethod
    def _pick_canonical(a: str, b: str, known_names: set[str]) -> str:
        if a in known_names and b not in known_names:
            return a
        if b in known_names and a not in known_names:
            return b
        return a if len(a) >= len(b) else b

    # ------------------------------------------------------------- relations

    @staticmethod
    def _allowed_edges(
        ontology: dict[str, Any] | None, a_type: str, b_type: str
    ) -> list[tuple[str, bool, str]]:
        """``(edge name, a_is_source, description)`` for allowed directions."""

        options = []
        for edge in (ontology or {}).get("edge_types", []):
            for pair in edge.get("source_targets", []) or []:
                if pair.get("source") == a_type and pair.get("target") == b_type:
                    options.append((edge["name"], True, edge.get("description", "")))
                if pair.get("source") == b_type and pair.get("target") == a_type:
                    options.append((edge["name"], False, edge.get("description", "")))
        return list(dict.fromkeys(options))[:25]

    def _relations(
        self,
        sentences: list[str],
        surfaces: dict[str, list[str]],
        entity_type: dict[str, str],
        ontology: dict[str, Any] | None,
    ) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()
        forms_longest_first = sorted(
            ((form, name) for name, forms in surfaces.items() for form in forms),
            key=lambda item: -len(item[0]),
        )
        for sentence in sentences:
            # Claim spans longest-first so a nested name (市政府 inside
            # 東海市政府) does not count as a second entity in the sentence.
            claimed = [False] * len(sentence)
            present: list[str] = []
            for form, name in forms_longest_first:
                for match in re.finditer(re.escape(form), sentence):
                    span = range(match.start(), match.end())
                    if any(claimed[i] for i in span):
                        continue
                    for i in span:
                        claimed[i] = True
                    if name not in present:
                        present.append(name)
            for i, a in enumerate(present):
                for b in present[i + 1 :]:
                    options = self._allowed_edges(ontology, entity_type[a], entity_type[b])
                    if not options:
                        continue
                    criteria: dict[str, str] = {}
                    lookup: dict[str, tuple[str, str, str]] = {}
                    for index, (edge, a_is_source, description) in enumerate(options):
                        source, target = (a, b) if a_is_source else (b, a)
                        key = f"r{index}"
                        criteria[key] = f"{source} {edge}（{description}） {target}"
                        lookup[key] = (edge, source, target)
                    criteria[NONE_KEY] = "句子沒有描述這兩者之間的上述關係"
                    answer = self.system_one.choice(
                        f"句子：{sentence}\n實體A：{a}（{entity_type[a]}）\n"
                        f"實體B：{b}（{entity_type[b]}）",
                        "這個句子描述了實體A與實體B之間的哪一種關係？",
                        criteria,
                    )
                    if answer.choice == NONE_KEY:
                        continue
                    edge, source, target = lookup[answer.choice]
                    if (source, edge, target) in seen:
                        continue
                    seen.add((source, edge, target))
                    relations.append(
                        ExtractedRelation(
                            name=edge,
                            source=source,
                            target=target,
                            fact=sentence,
                            attributes={"confidence": round(answer.confidence, 4)},
                        )
                    )
        return relations

    # ------------------------------------------------------------------ main

    def extract(
        self,
        text: str,
        ontology: dict[str, Any] | None,
        known_entities: list[tuple[str, str]] | None = None,
    ) -> Extraction:
        types = self._entity_types(ontology)
        if not types:
            return Extraction()
        sentences = split_sentences(text)
        if self.ner == "gliner":
            typed = self._gliner_entities(text, types)
        else:
            typed = self._typed_candidates(text, types)
        typed = [t for t in typed if t.entity_type in types]
        known = [(n, t) for n, t in (known_entities or []) if t in types]
        canonical = self._merge(typed, known, sentences)
        known_type = dict(known)

        surfaces: dict[str, list[str]] = {}
        entity_type: dict[str, str] = {}
        for t in typed:
            name = canonical[t.name]
            surfaces.setdefault(name, [])
            if t.name not in surfaces[name]:
                surfaces[name].append(t.name)
            if name != t.name and name not in surfaces[name] and name in text:
                surfaces[name].append(name)
            # The canonical name's own type wins (known graph node first).
            entity_type.setdefault(
                name, known_type.get(name) or next(
                    (x.entity_type for x in typed if x.name == name), t.entity_type
                )
            )

        entities = []
        for name, forms in surfaces.items():
            mentions = [s for s in sentences if any(f in s for f in forms)]
            if self.summary_fn is not None:
                summary = self.summary_fn(name, entity_type[name], mentions)
            else:
                summary = "".join(mentions[:MAX_SUMMARY_SENTENCES])
            aliases = [f for f in forms if f != name]
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=entity_type[name],
                    summary=summary,
                    attributes={"aliases": aliases} if aliases else {},
                )
            )
        relations = self._relations(sentences, surfaces, entity_type, ontology)
        return Extraction(entities, relations)


def llm_summary_fn(max_tokens: int = 120) -> SummaryFn:
    """Optional small-model node summary (EXTRACT_SUMMARY_LLM=1)."""

    from ..utils.llm_client import LLMClient

    client = LLMClient()

    def summarize(name: str, entity_type: str, mentions: list[str]) -> str:
        if not mentions:
            return ""
        return client.chat(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"用一到兩句話摘要「{name}」（{entity_type}）在下列句子中的角色，"
                        "只根據句子內容：\n" + "\n".join(mentions[:6])
                    ),
                }
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        ).strip()

    return summarize
