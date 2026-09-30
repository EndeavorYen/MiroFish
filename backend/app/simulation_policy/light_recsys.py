"""A recommender for OASIS that needs no torch (#65).

``oasis.social_platform.platform`` imports its recommender at module level,
and that module imports torch, sentence-transformers, scikit-learn and (via
``process_recsys_posts``) transformers, so ``import oasis`` alone costs about
280 MB and 7 s. The Twitter platform then loads twhin-bert (+269 MB) whose
pooler is randomly initialised: the vectors it ranks by are close to a random
projection. Profiling put the ranking at 0.1% of simulation time.

``install()`` puts this module in place of ``oasis.social_platform.recsys``
before OASIS is imported. It provides the four names the platform uses:

- ``rec_sys_personalized_twh`` (Twitter): overlap of word / character-bigram
  tokens between a user's bio plus latest post and each candidate post,
  weighted by OASIS's own recency score; users are not recommended their own
  posts while others are available.
- ``rec_sys_personalized_with_trace``: the same ranking.
- ``rec_sys_reddit`` and ``rec_sys_random``: OASIS's pure-Python versions.

``SIM_RECSYS=light`` (set by the local profiles) selects it; unset or
``oasis`` keeps OASIS as shipped.
"""

from __future__ import annotations

import heapq
import math
import random
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Dict, List

MODULE = "oasis.social_platform.recsys"
CANDIDATES = 500  # newest posts ranked per refresh
_WORD = re.compile(r"\w+")

_post_tokens: dict[int, frozenset[str]] = {}


def install(env: Mapping[str, str]) -> bool:
    """Stand in for OASIS's recommender when ``SIM_RECSYS=light``."""

    if (env.get("SIM_RECSYS") or "oasis").strip().lower() != "light":
        return False
    if "oasis.social_platform.platform" in sys.modules:
        raise RuntimeError("light_recsys.install() must run before oasis is imported")
    sys.modules[MODULE] = sys.modules[__name__]
    return True


def reset_globals() -> None:
    _post_tokens.clear()


def tokens(text: str | None) -> frozenset[str]:
    """Lower-cased words for ASCII text, character bigrams for the rest."""

    out: set[str] = set()
    for run in _WORD.findall((text or "").lower()):
        if run.isascii():
            out.add(run)
        elif len(run) == 1:
            out.add(run)
        else:
            out.update(run[i:i + 2] for i in range(len(run) - 1))
    return frozenset(out)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def recency(created_at: Any, current_time: int) -> float:
    """OASIS's twhin recency weight, kept positive past its 171-step horizon."""

    try:
        age = current_time - int(created_at)
    except (TypeError, ValueError):
        return 1.0
    return math.log(max((271.8 - age) / 100, 1.0001))


def _post_token_set(post: Dict[str, Any]) -> frozenset[str]:
    cached = _post_tokens.get(post["post_id"])
    if cached is None:
        cached = _post_tokens[post["post_id"]] = tokens(post.get("content"))
    return cached


def rec_sys_personalized_twh(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    latest_post_count: int,
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    current_time: int,
    recall_only: bool = False,
    enable_like_score: bool = False,
    use_openai_embedding: bool = False,
) -> List[List]:
    if len(post_table) <= max_rec_post_len:
        post_ids = [post["post_id"] for post in post_table]
        return [post_ids] * len(rec_matrix)

    candidates = post_table[-CANDIDATES:]
    latest: dict[Any, str] = {}
    for post in post_table:
        latest[post["user_id"]] = post.get("content") or ""
    weights = [(post, _post_token_set(post), recency(post.get("created_at"), current_time)) for post in candidates]

    new_rec_matrix = []
    for index in range(len(rec_matrix)):
        user = user_table[index] if index < len(user_table) else {}
        user_id = user.get("user_id", index)
        profile = tokens(f"{user.get('bio') or ''} {latest.get(user_id, '')}")
        scored = []
        for order, (post, post_tokens, weight) in enumerate(weights):
            own = post["user_id"] == user_id
            score = weight * (0.1 + similarity(profile, post_tokens))
            # Others' posts first, then score, then newest.
            scored.append((not own, score, order, post["post_id"]))
        top = heapq.nlargest(max_rec_post_len, scored)
        new_rec_matrix.append([post_id for *_, post_id in top])
    return new_rec_matrix


def rec_sys_personalized_with_trace(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    swap_rate: float = 0.1,
) -> List[List]:
    return rec_sys_personalized_twh(
        user_table, post_table, len(post_table), trace_table, rec_matrix, max_rec_post_len, 0)


# --- OASIS's pure-Python recommenders, unchanged ---------------------------------

def rec_sys_random(post_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int) -> List[List]:
    post_ids = [post["post_id"] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        return [post_ids] * len(rec_matrix)
    return [random.sample(post_ids, max_rec_post_len) for _ in range(len(rec_matrix))]


def calculate_hot_score(num_likes: int, num_dislikes: int, created_at: datetime) -> float:
    s = num_likes - num_dislikes
    order = math.log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0
    td = created_at - datetime(1970, 1, 1)
    seconds = td.days * 86400 + td.seconds + (float(td.microseconds) / 1e6) - 1134028003
    return round(sign * order + seconds / 45000, 7)


def rec_sys_reddit(post_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int) -> List[List]:
    post_ids = [post["post_id"] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        return [post_ids] * len(rec_matrix)
    all_hot_score = []
    for post in post_table:
        try:
            created_at = datetime.strptime(post["created_at"], "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            created_at = datetime.strptime(post["created_at"], "%Y-%m-%d %H:%M:%S")
        all_hot_score.append((calculate_hot_score(post["num_likes"], post["num_dislikes"], created_at), post["post_id"]))
    top_posts = heapq.nlargest(max_rec_post_len, all_hot_score, key=lambda x: x[0])
    return [[post_id for _, post_id in top_posts]] * len(rec_matrix)
