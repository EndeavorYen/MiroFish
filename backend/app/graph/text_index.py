"""Tokenisation for FTS5 over mixed Chinese / Latin text.

SQLite's ``unicode61`` tokenizer treats a run of CJK characters as one token,
so BM25 would only match whole sentences. Text is pre-tokenised in Python:
Latin/digit words are lower-cased, CJK runs become overlapping bigrams (plus
the single character for a one-character run). The same function builds the
index text and the query, so the two always agree.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]+"
    r"|[0-9A-Za-zÀ-ɏ]+"
)
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        run = match.group(0)
        if _CJK_RE.match(run):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run.lower())
    return tokens


def index_text(text: str) -> str:
    return " ".join(tokenize(text))


def match_query(text: str) -> str | None:
    """FTS5 MATCH expression: OR of quoted unique tokens, or None if empty."""

    seen: dict[str, None] = {}
    for token in tokenize(text):
        seen.setdefault(token, None)
    if not seen:
        return None
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in seen)


def char_ngrams(text: str, sizes: tuple[int, ...] = (1, 2, 3)) -> list[str]:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip()
    grams: list[str] = []
    for size in sizes:
        grams.extend(compact[i : i + size] for i in range(max(len(compact) - size + 1, 0)))
    return grams
