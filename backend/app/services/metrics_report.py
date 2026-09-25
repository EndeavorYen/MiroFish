"""Deterministic metrics report plus a short summary (#12).

Computed from the simulation's structured outputs, no generation:

* emotion curves per round (all agents, and per entity type) from
  ``agent_state.db`` (System One runs);
* action distribution per round from ``<platform>/actions.jsonl``;
* the most spread posts and their repost/quote chains from the OASIS
  ``<platform>_simulation.db``;
* stance by entity type from ``decisions.jsonl`` intents and the agent
  config stance labels.

``report_metrics.json`` / ``report_metrics.md`` are written to the report
folder. A small model then writes one summary of at most 800 tokens from the
metrics JSON under the ``report`` usage stage. The same input always gives
the same metrics (sorted keys, rounded floats, no timestamps).

``REPORT_MODE=metrics|agent`` selects this or the existing ReportAgent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable

PLATFORMS = ("twitter", "reddit")
SUMMARY_MAX_TOKENS = 800
TOP_POSTS = 5


logger = logging.getLogger(__name__)


def _round(value: float) -> float:
    return round(float(value), 4)


def _load_config(sim_dir: str) -> dict[str, Any]:
    path = os.path.join(sim_dir, "simulation_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _agent_types(config: dict[str, Any]) -> dict[int, str]:
    return {
        int(a["agent_id"]): a.get("entity_type") or "Unknown"
        for a in config.get("agent_configs", [])
        if a.get("agent_id") is not None
    }


def _jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ------------------------------------------------------------------ parts


def action_distribution(sim_dir: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for platform in PLATFORMS:
        per_round: dict[int, Counter] = defaultdict(Counter)
        for row in _jsonl(os.path.join(sim_dir, platform, "actions.jsonl")):
            if "event_type" in row or not row.get("action_type"):
                continue
            per_round[int(row.get("round", 0))][row["action_type"]] += 1
        if not per_round:
            continue
        totals = Counter()
        for counts in per_round.values():
            totals.update(counts)
        result[platform] = {
            "total": dict(sorted(totals.items())),
            "by_round": {
                str(r): dict(sorted(per_round[r].items())) for r in sorted(per_round)
            },
        }
    return result


def emotion_curves(sim_dir: str, agent_types: dict[int, str]) -> dict[str, Any] | None:
    path = os.path.join(sim_dir, "agent_state.db")
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT round, platform, agent_id, state FROM emotion ORDER BY round, platform, agent_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return None  # store created but never written (no emotion table)
    finally:
        conn.close()
    overall: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    grouped: dict[str, dict[str, dict[int, list[dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for round_num, platform, agent_id, state in rows:
        values = json.loads(state)
        overall[platform][round_num].append(values)
        grouped[platform][agent_types.get(agent_id, "Unknown")][round_num].append(values)

    def mean_curve(by_round: dict[int, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
        curve = {}
        for round_num in sorted(by_round):
            states = by_round[round_num]
            dims = sorted({d for s in states for d in s})
            curve[str(round_num)] = {
                d: _round(statistics.mean(s.get(d, 0.0) for s in states)) for d in dims
            }
        return curve

    return {
        platform: {
            "overall": mean_curve(overall[platform]),
            "by_entity_type": {
                etype: mean_curve(grouped[platform][etype]) for etype in sorted(grouped[platform])
            },
        }
        for platform in sorted(overall)
    }


def spread_posts(sim_dir: str, limit: int = TOP_POSTS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for platform in PLATFORMS:
        path = os.path.join(sim_dir, f"{platform}_simulation.db")
        if not os.path.exists(path):
            continue
        conn = sqlite3.connect(path)
        try:
            names = dict(conn.execute("SELECT user_id, name FROM user").fetchall())
            posts = conn.execute(
                "SELECT post_id, user_id, original_post_id, content, quote_content, created_at, "
                "num_likes, num_shares FROM post ORDER BY post_id"
            ).fetchall()
        finally:
            conn.close()
        by_id = {p[0]: p for p in posts}

        def root_of(post_id: int) -> int:
            seen = set()
            current = post_id
            while by_id.get(current) and by_id[current][2] and current not in seen:
                seen.add(current)
                current = by_id[current][2]
            return current

        chains: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for post in posts:
            if post[2]:
                root = root_of(post[0])
                chains[root].append(
                    {
                        "post_id": post[0],
                        "by": names.get(post[1], f"User {post[1]}"),
                        "kind": "quote" if post[4] else "repost",
                        "parent": post[2],
                    }
                )
        ranked = sorted(
            (p for p in posts if not p[2]),
            key=lambda p: (-(len(chains.get(p[0], [])) + (p[6] or 0)), p[0]),
        )[:limit]
        result[platform] = [
            {
                "post_id": p[0],
                "author": names.get(p[1], f"User {p[1]}"),
                "content": (p[3] or "")[:200],
                "likes": p[6] or 0,
                "reposts_and_quotes": len(chains.get(p[0], [])),
                "chain": chains.get(p[0], []),
            }
            for p in ranked
        ]
    return result


def stance_groups(sim_dir: str, config: dict[str, Any], agent_types: dict[int, str]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in _jsonl(os.path.join(sim_dir, "decisions.jsonl")):
        intent = row.get("intent")
        if isinstance(intent, dict) and isinstance(intent.get("stance"), (int, float)):
            by_type[agent_types.get(row.get("agent_id"), "Unknown")].append(float(intent["stance"]))
    labels: dict[str, Counter] = defaultdict(Counter)
    for agent in config.get("agent_configs", []):
        labels[agent.get("entity_type") or "Unknown"][agent.get("stance") or "neutral"] += 1
    groups = {}
    for etype in sorted(set(by_type) | set(labels)):
        values = by_type.get(etype, [])
        groups[etype] = {
            "configured_stances": dict(sorted(labels.get(etype, Counter()).items())),
            "expressed_stance_mean": _round(statistics.mean(values)) if values else None,
            "expressed_stance_samples": len(values),
        }
    return groups


def compute_metrics(sim_dir: str) -> dict[str, Any]:
    config = _load_config(sim_dir)
    agent_types = _agent_types(config)
    actions = action_distribution(sim_dir)
    return {
        "simulation_id": config.get("simulation_id"),
        "agents": len(agent_types),
        "entity_types": dict(sorted(Counter(agent_types.values()).items())),
        "actions": actions,
        "emotion": emotion_curves(sim_dir, agent_types),
        "spread": spread_posts(sim_dir),
        "stance": stance_groups(sim_dir, config, agent_types),
        "hot_topics": list(config.get("event_config", {}).get("hot_topics", [])),
    }


# --------------------------------------------------------------- markdown


def _final_emotions(curve: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | None:
    if not curve:
        return None
    last = max(curve, key=int)
    return last, curve[last]


def render_markdown(metrics: dict[str, Any], summary: str = "") -> str:
    lines = ["# 模擬指標報告", ""]
    if summary:
        lines += ["## 摘要", "", summary.strip(), ""]
    lines += [
        "## 概況",
        "",
        f"- Agent 數：{metrics['agents']}",
        f"- 實體類型：{', '.join(f'{k} {v}' for k, v in metrics['entity_types'].items()) or '—'}",
        f"- 熱門話題：{', '.join(metrics['hot_topics']) or '—'}",
        "",
        "## 動作分布",
        "",
    ]
    for platform, data in metrics["actions"].items():
        total = sum(data["total"].values())
        parts = ", ".join(
            f"{a} {n}（{n / total:.0%}）" for a, n in sorted(data["total"].items(), key=lambda x: -x[1])
        )
        lines.append(f"- **{platform}**（{total} 個動作）：{parts}")
    lines += ["", "## 情緒曲線", ""]
    if metrics["emotion"]:
        for platform, data in metrics["emotion"].items():
            final = _final_emotions(data["overall"])
            first_round = min(data["overall"], key=int) if data["overall"] else None
            if final and first_round is not None:
                first = data["overall"][first_round]
                change = "，".join(
                    f"{d} {first.get(d, 0):.2f}→{v:.2f}" for d, v in sorted(final[1].items())
                )
                lines.append(f"- **{platform}** 第 {first_round}→{final[0]} 輪：{change}")
    else:
        lines.append("- 無情緒狀態資料（LLM 決策模式不記錄情緒）。")
    lines += ["", "## 擴散最廣的貼文", ""]
    for platform, posts in metrics["spread"].items():
        for post in posts:
            chain = " → ".join(step["by"] for step in post["chain"][:6])
            lines.append(
                f"- [{platform} #{post['post_id']}] {post['author']}：{post['content'][:60]}"
                f"（轉發／引用 {post['reposts_and_quotes']}，讚 {post['likes']}）"
                + (f"；擴散路徑：{chain}" if chain else "")
            )
    lines += ["", "## 立場分群", ""]
    for etype, group in metrics["stance"].items():
        mean = group["expressed_stance_mean"]
        lines.append(
            f"- **{etype}**：設定立場 {group['configured_stances'] or '—'}；"
            f"發言立場平均 {mean if mean is not None else '—'}（0=強烈反對，1=強烈支持，n={group['expressed_stance_samples']}）"
        )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- summary

SummaryFn = Callable[[str, int], str]


def summary_prompt(metrics: dict[str, Any], requirement: str) -> str:
    compact = {k: metrics[k] for k in ("agents", "entity_types", "hot_topics", "stance")}
    compact["actions_total"] = {p: d["total"] for p, d in metrics["actions"].items()}
    compact["top_posts"] = {
        p: [
            {k: post[k] for k in ("author", "content", "reposts_and_quotes", "likes")}
            for post in posts[:3]
        ]
        for p, posts in metrics["spread"].items()
    }
    if metrics["emotion"]:
        compact["final_emotion"] = {
            p: _final_emotions(d["overall"]) for p, d in metrics["emotion"].items()
        }
    return (
        f"模擬需求：{requirement}\n以下是一場社群輿論模擬的量化指標（JSON）：\n"
        f"{json.dumps(compact, ensure_ascii=False, sort_keys=True)}\n\n"
        "請用 300 字以內的繁體中文，根據這些數字寫一段摘要：主要動態、立場分布、情緒走向、"
        "擴散最廣的內容。只能根據數字，不要編造數字以外的事。"
    )


def default_summary_fn() -> SummaryFn:
    from openai import OpenAI

    from ..config import Config
    from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text

    client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL, timeout=120)

    def summarize(prompt: str, max_tokens: int) -> str:
        response = create_chat_completion(
            client,
            model=Config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return extract_chat_completion_text(response)

    return summarize


def write_metrics_report(
    sim_dir: str,
    report_dir: str,
    requirement: str,
    *,
    summary_fn: SummaryFn | None = None,
    metrics_dir: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Compute metrics, write the two files, return (metrics, markdown)."""

    from ..utils.llm_usage import usage_stage

    metrics = compute_metrics(sim_dir)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "report_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    summary = ""
    if summary_fn is not None:
        with usage_stage("report", metrics_dir=metrics_dir):
            try:
                summary = summary_fn(summary_prompt(metrics, requirement), SUMMARY_MAX_TOKENS)
            except Exception as error:  # noqa: BLE001 - the metrics report stands alone
                logger.warning("metrics report summary failed", exc_info=True)
                summary = f"（摘要產生失敗：{type(error).__name__}）"
    markdown = render_markdown(metrics, summary)
    with open(os.path.join(report_dir, "report_metrics.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    return metrics, markdown


def generate_metrics_report(
    simulation_id: str,
    graph_id: str,
    requirement: str,
    report_id: str,
    *,
    summary_fn: SummaryFn | None = None,
):
    """REPORT_MODE=metrics: build and save a Report like ReportAgent does."""

    from datetime import datetime

    from .report_agent import (
        Report,
        ReportLogger,
        ReportManager,
        ReportOutline,
        ReportSection,
        ReportStatus,
    )
    from .simulation_manager import SimulationManager

    sim_dir = SimulationManager()._get_simulation_dir(simulation_id)
    created = datetime.now().isoformat()
    report = Report(
        report_id=report_id,
        simulation_id=simulation_id,
        graph_id=graph_id,
        simulation_requirement=requirement,
        status=ReportStatus.GENERATING,
        created_at=created,
    )
    started = datetime.now()
    # The report page follows agent_log.jsonl: report_start, the outline,
    # one section_complete per section, then report_complete.
    report_logger = ReportLogger(report_id)
    report_logger.log_start(simulation_id, graph_id, requirement)
    try:
        _, markdown = write_metrics_report(
            sim_dir,
            ReportManager._get_report_folder(report_id),
            requirement,
            summary_fn=summary_fn if summary_fn is not None else default_summary_fn(),
            metrics_dir=os.path.join(sim_dir, "metrics"),
        )
        outline = markdown_outline(markdown, ReportOutline, ReportSection)
        report.outline = outline
        report_logger.log_planning_complete(outline.to_dict())
        for index, section in enumerate(outline.sections, start=1):
            report_logger.log_section_start(section.title, index)
            report_logger.log_section_full_complete(
                section.title, index, f"## {section.title}\n\n{section.content}".strip()
            )
        report.markdown_content = markdown
        report.status = ReportStatus.COMPLETED
        report_logger.log_report_complete(
            len(outline.sections), (datetime.now() - started).total_seconds()
        )
    except Exception as error:  # noqa: BLE001 - surface as a failed report
        report.status = ReportStatus.FAILED
        report.error = str(error)
        report_logger.log_error(str(error), "failed")
    report.completed_at = datetime.now().isoformat()
    ReportManager.save_report(report)
    return report


def markdown_outline(markdown: str, outline_cls, section_cls):
    """Split a rendered metrics report into a ReportOutline (# title, ## sections)."""

    title_match = re.search(r"^# (.+)$", markdown, flags=re.M)
    title = title_match.group(1).strip() if title_match else "模擬指標報告"
    parts = re.split(r"^## (.+)$", markdown, flags=re.M)
    sections = [
        section_cls(title=parts[i].strip(), content=parts[i + 1].strip())
        for i in range(1, len(parts) - 1, 2)
    ]
    summary = next((s.content.split("\n\n")[0] for s in sections if s.content), "")
    return outline_cls(title=title, summary=summary, sections=sections)
