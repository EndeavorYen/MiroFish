"""Map simulation actions to structured graph facts without any model (#8).

Every action becomes ``(agent) -[ACTION_TYPE]-> (target)`` where the target
is a post, comment, user, search query or the platform feed. Posts and
comments are nodes that carry their text. The idempotency key is
``(platform, round, agent_id, action_seq)``; ``action_seq`` counts the
agent's earlier actions in the same round, in action-log order.

Post and comment text is linked to existing entities by plain name matching
(no NER model), producing ``MENTIONS`` edges.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from ..graph.store import FactNode, StructuredFact

TWITTER_ACTIONS = ("CREATE_POST", "LIKE_POST", "REPOST", "FOLLOW", "DO_NOTHING", "QUOTE_POST")
REDDIT_ACTIONS = (
    "LIKE_POST", "DISLIKE_POST", "CREATE_POST", "CREATE_COMMENT", "LIKE_COMMENT",
    "DISLIKE_COMMENT", "SEARCH_POSTS", "SEARCH_USER", "TREND", "REFRESH", "DO_NOTHING",
    "FOLLOW", "MUTE",
)
SKIPPED_ACTIONS = {"DO_NOTHING"}
MAX_TEXT = 300


def fact_key(platform: str, round_num: int, agent_id: Any, action_seq: int) -> str:
    return f"{platform}:{round_num}:{agent_id}:{action_seq}"


def find_mentions(text: str, entity_names: Iterable[str]) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(sorted({name for name in entity_names if len(name) >= 2 and name in text}))


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _agent(activity) -> FactNode:
    name = activity.agent_name or f"Agent_{activity.agent_id}"
    return FactNode(
        key=f"agent:{activity.agent_id}",
        name=name,
        label="SimAgent",
        attributes={"agent_id": activity.agent_id},
    )


def _user(name: str) -> FactNode:
    return FactNode(key=f"user:{name}", name=name, label="SimAgent")


def _post(platform: str, post_id: Any, content: str, author: str, fallback: str) -> FactNode:
    ident = post_id if post_id not in (None, "") else (
        f"h{_short_hash(content)}" if content else f"fact:{fallback}"
    )
    label = f"{author}的帖子" if author else "帖子"
    return FactNode(
        key=f"post:{platform}:{ident}",
        name=f"{label}#{ident}",
        label="Post",
        summary=(content or "")[:MAX_TEXT],
        attributes={"platform": platform, "post_id": post_id, "author": author or None},
    )


def _comment(platform: str, comment_id: Any, content: str, author: str, fallback: str) -> FactNode:
    ident = comment_id if comment_id not in (None, "") else f"fact:{fallback}"
    label = f"{author}的评论" if author else "评论"
    return FactNode(
        key=f"comment:{platform}:{ident}",
        name=f"{label}#{ident}",
        label="Comment",
        summary=(content or "")[:MAX_TEXT],
        attributes={"platform": platform, "comment_id": comment_id, "author": author or None},
    )


def activity_to_fact(activity, action_seq: int, entity_names: Iterable[str] = ()) -> StructuredFact | None:
    """Return the fact for one action, or ``None`` for DO_NOTHING."""

    action = activity.action_type
    if action in SKIPPED_ACTIONS:
        return None
    platform = (activity.platform or "").lower()
    args: dict[str, Any] = activity.action_args or {}
    key = fact_key(platform, activity.round_num, activity.agent_id, action_seq)
    agent = _agent(activity)
    extra_nodes: list[FactNode] = []
    extra_edges: list[tuple[str, str, str]] = []
    mention_text = ""

    def authored(author: str, node: FactNode) -> None:
        if author:
            author_node = _user(author)
            extra_nodes.append(author_node)
            extra_edges.append((author_node.key, "AUTHORED", node.key))

    if action == "CREATE_POST":
        content = args.get("content", "")
        target = _post(platform, args.get("post_id"), content, agent.name, key)
        mention_text = content
    elif action in ("LIKE_POST", "DISLIKE_POST"):
        author = args.get("post_author_name", "")
        target = _post(platform, args.get("post_id"), args.get("post_content", ""), author, key)
        authored(author, target)
    elif action == "REPOST":
        author = args.get("original_author_name", "")
        target = _post(
            platform,
            args.get("original_post_id"),
            args.get("original_content", ""),
            author,
            key,
        )
        authored(author, target)
        if args.get("new_post_id") not in (None, ""):
            repost = _post(platform, args["new_post_id"], args.get("original_content", ""), agent.name, key)
            extra_nodes.append(repost)
            extra_edges.append((repost.key, "REPOST_OF", target.key))
    elif action == "QUOTE_POST":
        author = args.get("original_author_name", "")
        target = _post(platform, args.get("quoted_id"), args.get("original_content", ""), author, key)
        authored(author, target)
        quote_text = args.get("quote_content") or args.get("content", "")
        quote = _post(platform, args.get("new_post_id"), quote_text, agent.name, f"{key}:quote")
        extra_nodes.append(quote)
        extra_edges.append((quote.key, "QUOTES", target.key))
        mention_text = quote_text
    elif action in ("FOLLOW", "MUTE"):
        name = args.get("target_user_name") or f"user_{args.get('follow_id') or args.get('user_id') or 'unknown'}"
        target = _user(name)
    elif action == "CREATE_COMMENT":
        content = args.get("content", "")
        target = _comment(platform, args.get("comment_id"), content, agent.name, key)
        post_author = args.get("post_author_name", "")
        post = _post(platform, args.get("post_id"), args.get("post_content", ""), post_author, f"{key}:post")
        extra_nodes.append(post)
        extra_edges.append((target.key, "REPLIES_TO", post.key))
        authored(post_author, post)
        mention_text = content
    elif action in ("LIKE_COMMENT", "DISLIKE_COMMENT"):
        author = args.get("comment_author_name", "")
        target = _comment(platform, args.get("comment_id"), args.get("comment_content", ""), author, key)
        authored(author, target)
    elif action in ("SEARCH_POSTS", "SEARCH_USER"):
        query = args.get("query") or args.get("keyword") or args.get("username") or ""
        target = FactNode(
            key=f"query:{platform}:{action}:{query}",
            name=f"搜索「{query}」" if query else "搜索",
            label="SearchQuery",
            summary=query,
        )
    else:  # TREND, REFRESH and any future action: the platform feed
        target = FactNode(key=f"feed:{platform}", name=f"{platform} feed", label="Feed")

    return StructuredFact(
        key=key,
        source=agent,
        relation=action,
        target=target,
        fact=activity.to_episode_text(),
        attributes={
            "platform": platform,
            "round": activity.round_num,
            "simulated_time": activity.timestamp,
            "agent_id": activity.agent_id,
            "action_seq": action_seq,
        },
        created_at=activity.timestamp,
        extra_nodes=tuple(extra_nodes),
        extra_edges=tuple(extra_edges),
        mentions=find_mentions(mention_text, entity_names),
    )
