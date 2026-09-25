"""Map simulation actions to structured graph facts without any model (#8).

Every action becomes ``(agent) -[ACTION_TYPE]-> (target)`` where the target
is a post, comment, user, search query or the platform feed. Posts and
comments are nodes that carry their text. The idempotency key is
``(simulation_id, platform, round, agent_id, action_seq)``; ``action_seq``
counts the agent's earlier actions in the same round, in action-log order.
Post, comment and agent keys carry the simulation id too, because every
OASIS run numbers its posts and users from 1 again.

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


def fact_key(
    platform: str, round_num: int, agent_id: Any, action_seq: int, simulation_id: str = ""
) -> str:
    return f"{simulation_id}:{platform}:{round_num}:{agent_id}:{action_seq}"


def find_mentions(text: str, entity_names: Iterable[str]) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(sorted({name for name in entity_names if len(name) >= 2 and name in text}))


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _agent(sim: str, agent_id: Any, name: str) -> FactNode:
    return FactNode(
        key=f"agent:{sim}:{agent_id}",
        name=name or f"Agent_{agent_id}",
        label="SimAgent",
        attributes={"agent_id": agent_id, "scope": sim},
    )


def _user(sim: str, name: str, user_id: Any = None, fallback: str = "") -> FactNode:
    if user_id not in (None, ""):
        return _agent(sim, user_id, name)
    if name:
        return FactNode(
            key=f"user:{sim}:{name}", name=name, label="SimAgent", attributes={"scope": sim}
        )
    # Unknown target: keep it per action rather than one shared placeholder.
    return FactNode(
        key=f"user:{sim}:unknown:{fallback}",
        name=f"未知用户#{fallback}",
        label="SimAgent",
        attributes={"scope": sim, "unknown": True},
    )


def _post(
    sim: str,
    platform: str,
    post_id: Any,
    content: str,
    author: str,
    fallback: str,
    *,
    by_content: bool = True,
) -> FactNode:
    if post_id not in (None, ""):
        ident = post_id
    elif by_content and content:
        ident = f"h{_short_hash(content)}"  # a referenced post seen only by text
    else:
        ident = f"fact:{fallback}"
    label = f"{author}的帖子" if author else "帖子"
    return FactNode(
        key=f"post:{sim}:{platform}:{ident}",
        name=f"{label}#{ident}",
        label="Post",
        summary=(content or "")[:MAX_TEXT],
        attributes={"platform": platform, "post_id": post_id, "author": author or None},
    )


def _comment(
    sim: str, platform: str, comment_id: Any, content: str, author: str, fallback: str
) -> FactNode:
    ident = comment_id if comment_id not in (None, "") else f"fact:{fallback}"
    label = f"{author}的评论" if author else "评论"
    return FactNode(
        key=f"comment:{sim}:{platform}:{ident}",
        name=f"{label}#{ident}",
        label="Comment",
        summary=(content or "")[:MAX_TEXT],
        attributes={"platform": platform, "comment_id": comment_id, "author": author or None},
    )


def activity_to_fact(
    activity,
    action_seq: int,
    entity_names: Iterable[str] = (),
    simulation_id: str = "",
) -> StructuredFact | None:
    """Return the fact for one action, or ``None`` for DO_NOTHING."""

    action = activity.action_type
    if action in SKIPPED_ACTIONS:
        return None
    platform = (activity.platform or "").lower()
    args: dict[str, Any] = activity.action_args or {}
    sim = simulation_id
    key = fact_key(platform, activity.round_num, activity.agent_id, action_seq, sim)
    agent = _agent(sim, activity.agent_id, activity.agent_name)
    extra_nodes: list[FactNode] = []
    extra_edges: list[tuple[str, str, str]] = []
    mention_text = ""

    def authored(author: str, node: FactNode) -> None:
        if author:
            author_node = _user(sim, author)
            extra_nodes.append(author_node)
            extra_edges.append((author_node.key, "AUTHORED", node.key))

    if action == "CREATE_POST":
        content = args.get("content", "")
        target = _post(sim, platform, args.get("post_id"), content, agent.name, key, by_content=False)
        mention_text = content
    elif action in ("LIKE_POST", "DISLIKE_POST"):
        author = args.get("post_author_name", "")
        target = _post(sim, platform, args.get("post_id"), args.get("post_content", ""), author, key)
        authored(author, target)
    elif action == "REPOST":
        author = args.get("original_author_name", "")
        target = _post(
            sim,
            platform,
            args.get("original_post_id"),
            args.get("original_content", ""),
            author,
            key,
        )
        authored(author, target)
        if args.get("new_post_id") not in (None, ""):
            repost = _post(sim, platform, args["new_post_id"], args.get("original_content", ""), agent.name, key)
            extra_nodes.append(repost)
            extra_edges.append((repost.key, "REPOST_OF", target.key))
    elif action == "QUOTE_POST":
        author = args.get("original_author_name", "")
        target = _post(sim, platform, args.get("quoted_id"), args.get("original_content", ""), author, key)
        authored(author, target)
        quote_text = args.get("quote_content") or args.get("content", "")
        quote = _post(
            sim, platform, args.get("new_post_id"), quote_text, agent.name, f"{key}:quote",
            by_content=False,
        )
        extra_nodes.append(quote)
        extra_edges.append((quote.key, "QUOTES", target.key))
        mention_text = quote_text
    elif action in ("FOLLOW", "MUTE"):
        target = _user(
            sim,
            args.get("target_user_name", ""),
            args.get("followee_id") if action == "FOLLOW" else args.get("mutee_id"),
            key,
        )
    elif action == "CREATE_COMMENT":
        content = args.get("content", "")
        target = _comment(sim, platform, args.get("comment_id"), content, agent.name, key)
        post_author = args.get("post_author_name", "")
        post = _post(sim, platform, args.get("post_id"), args.get("post_content", ""), post_author, f"{key}:post")
        extra_nodes.append(post)
        extra_edges.append((target.key, "REPLIES_TO", post.key))
        authored(post_author, post)
        mention_text = content
    elif action in ("LIKE_COMMENT", "DISLIKE_COMMENT"):
        author = args.get("comment_author_name", "")
        target = _comment(sim, platform, args.get("comment_id"), args.get("comment_content", ""), author, key)
        authored(author, target)
    elif action in ("SEARCH_POSTS", "SEARCH_USER"):
        query = args.get("query") or args.get("keyword") or args.get("username") or ""
        target = FactNode(
            key=f"query:{sim}:{platform}:{action}:{query}",
            name=f"搜索「{query}」" if query else "搜索",
            label="SearchQuery",
            summary=query,
        )
    else:  # TREND, REFRESH and any future action: the platform feed
        target = FactNode(key=f"feed:{sim}:{platform}", name=f"{platform} feed", label="Feed")

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
