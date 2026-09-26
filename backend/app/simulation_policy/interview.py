"""Interview context for System One agents (#10).

In ``system_one`` mode an agent's own actions were decided without text, so
the interview prompt gets a short summary of that agent's decisions
(decisions.jsonl) and emotion trajectory (agent_state.db). The existing IPC
interview flow then runs unchanged; the small model answers under the
``interview`` usage stage, outside any per-round content budget.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from .emotion import DIMENSION_LABELS, AgentStateStore

ACTION_LABELS = {
    "CREATE_POST": "发帖",
    "LIKE_POST": "点赞帖子",
    "DISLIKE_POST": "踩帖子",
    "REPOST": "转发",
    "QUOTE_POST": "引用转发",
    "FOLLOW": "关注他人",
    "MUTE": "屏蔽他人",
    "CREATE_COMMENT": "评论",
    "LIKE_COMMENT": "点赞评论",
    "DISLIKE_COMMENT": "踩评论",
    "SEARCH_POSTS": "搜索帖子",
    "SEARCH_USER": "搜索用户",
    "TREND": "看趋势",
    "REFRESH": "刷新",
    "DO_NOTHING": "旁观",
}


def _decisions(simulation_dir: str, platform: str | None, agent_id: int) -> list[dict[str, Any]]:
    path = os.path.join(simulation_dir, "decisions.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("agent_id") != agent_id or (platform is not None and row.get("platform") != platform):
                continue
            # A later action of a round that ended in an exit is not "did nothing".
            if row.get("action_index") and row.get("action") == "DO_NOTHING":
                continue
            rows.append(row)
    return rows


def interview_context(
    simulation_dir: str, agent_id: int, platform: str | None = None, max_posts: int = 5
) -> str:
    """Summary of the agent's own decisions and emotions; empty if none."""

    rows = _decisions(simulation_dir, platform, agent_id)
    lines: list[str] = []
    if rows:
        counts = Counter(row.get("action") or row.get("chosen") for row in rows)
        summary = "、".join(
            f"{ACTION_LABELS.get(action, action)} {n} 次" for action, n in counts.most_common()
        )
        lines.append(f"你在模拟中的行为：{summary}。")
        said = [
            row["args"].get("content") or row["args"].get("quote_content")
            for row in rows
            if isinstance(row.get("args"), dict)
            and (row["args"].get("content") or row["args"].get("quote_content"))
        ]
        if said:
            lines.append("你发表过的内容：" + " / ".join(said[-max_posts:]))
        stances = [row["intent"]["stance"] for row in rows if isinstance(row.get("intent"), dict)]
        if stances:
            lines.append(f"你的平均立场（0=强烈反对，1=强烈支持）：{sum(stances) / len(stances):.2f}。")
    db_path = os.path.join(simulation_dir, "agent_state.db")
    if os.path.exists(db_path):
        store = AgentStateStore(db_path)
        try:
            histories = (
                [store.history(platform, agent_id)]
                if platform
                else [store.history(p, agent_id) for p in ("twitter", "reddit")]
            )
        finally:
            store.close()
        for history in histories:
            if len(history) >= 1:
                first, last = history[0][1], history[-1][1]
                change = "，".join(
                    f"{DIMENSION_LABELS.get(d, d)} {first.get(d, 0):.2f}→{last.get(d, 0):.2f}"
                    for d in last
                )
                lines.append(f"你的情绪变化（第 {history[0][0]} 轮到第 {history[-1][0]} 轮）：{change}。")
                break
    if not lines:
        return ""
    return "【你在这次模拟中的经历】\n" + "\n".join(lines)


def augment_interview_prompt(
    prompt: str, simulation_dir: str, agent_id: int, platform: str | None = None
) -> str:
    """Prepend the decision/emotion summary when SIM_DECISION_BACKEND=system_one."""

    if os.environ.get("SIM_DECISION_BACKEND", "llm").strip().lower() != "system_one":
        return prompt
    context = interview_context(simulation_dir, agent_id, platform)
    return f"{context}\n\n{prompt}" if context else prompt
