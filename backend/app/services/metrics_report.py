"""Deterministic metrics report plus a short summary (#12).

Computed from the simulation's structured outputs, no generation:

* emotion curves per round (all agents, and per entity type) from
  ``agent_state.db`` (System One runs);
* action distribution per round from ``<platform>/actions.jsonl``;
* the most spread posts and their repost/quote chains from the OASIS
  ``<platform>_simulation.db``;
* stance by entity type from ``decisions.jsonl`` intents and the agent
  config stance labels;
* scan conclusions (main camp, overall tendency, trend, role ranking) from
  the posts' zero-decode stance readout, each with how far it can be trusted
  on the local path (#53).

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

from ..utils.locale import t

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


# ------------------------------------------------------------ scan (#53)

# How often each conclusion of the local path (System One decisions, tiered
# content) matched the LLM path on the evaluation suite: gate G5 over six
# scenarios x five seeds, docs/local-first.md. Trend and ranking held in half
# of them or fewer, so the report points to MIROFISH_PROFILE=local-llm.
SCAN_EVIDENCE = {
    "main_camp": ("high", "6/6"),
    "tendency": ("medium", "3/6"),
    "trend": ("low", "3/6"),
    "ranking": ("low", "2/6"),
}
# The same for MIROFISH_PROFILE=local-hybrid (LLM prep, local simulation);
# A and B share the prep there, so it measures whether the cheap simulation
# reproduces the LLM simulation's conclusions from one preparation (#58).
HYBRID_EVIDENCE = {
    "main_camp": ("medium", "4/6"),
    "tendency": ("high", "6/6"),
    "trend": ("high", "5/6"),
    "ranking": ("medium", "3/6"),
}
CONFIDENCE_LABEL = {"high": "高", "medium": "中", "low": "低", "none": "無法判讀"}
CAMP_LABEL = {"oppose": "反對", "neutral": "中立", "support": "支持"}
# Configured stances (0..1) spread less than this: the readout did not tell
# the roles apart, and their order in the posts is noise. Set on the
# calibration scenarios (lakemont: 0.097, ranking 0.07); on the evaluation
# suite it flags gangwan, qingpu and Northbridge, all three below 0.5.
INDISTINCT_SD = 0.10
TREND_WINDOW = 4
ScoreFn = Callable[[str, str], float]


def _posts(sim_dir: str) -> list[dict[str, Any]]:
    rows = []
    for platform in PLATFORMS:
        for row in _jsonl(os.path.join(sim_dir, platform, "actions.jsonl")):
            if "event_type" in row or row.get("action_type") not in ("CREATE_POST", "QUOTE_POST", "CREATE_COMMENT"):
                continue
            args = row.get("action_args") or {}
            text = args.get("quote_content") or args.get("content")
            if text:
                rows.append({
                    "round": int(row.get("round", 0)),
                    "agent": str(row.get("agent_name") or f"id:{row.get('agent_id')}"),
                    "text": str(text),
                })
    return rows


def _camp(stance: float) -> str:
    return "oppose" if stance < 0.4 else "support" if stance > 0.6 else "neutral"


def default_score_fn() -> ScoreFn:
    from ..simulation_policy.tiers import system_one_stance_score

    return lambda text, question: system_one_stance_score(text, question=question)


def scan_conclusions(sim_dir: str, score_fn: ScoreFn, question: str | None = None) -> dict[str, Any] | None:
    """Directional conclusions from the posts, scored on the simulated event
    in the same form as the evaluation (gate G5), with their confidence.

    ``question`` overrides the one built from the run's requirement, so runs
    of different options are scored on the same question (#56)."""

    from ..simulation_policy.tiers import detect_content_lang, event_phrase, stance_check_question

    config = _load_config(sim_dir)
    requirement = str(config.get("simulation_requirement") or "")
    question = question or stance_check_question(event_phrase(requirement), detect_content_lang(requirement))
    posts = _posts(sim_dir)
    if not posts:
        return None
    cache: dict[str, float] = {}
    for post in posts:
        if post["text"] not in cache:
            cache[post["text"]] = float(score_fn(post["text"], question))
    scored = [cache[p["text"]] for p in posts]
    by_agent: dict[str, list[float]] = defaultdict(list)
    by_window: dict[int, list[float]] = defaultdict(list)
    for post in posts:
        by_agent[post["agent"]].append(cache[post["text"]])
        by_window[post["round"] // TREND_WINDOW].append(cache[post["text"]])
    means = {agent: statistics.mean(values) for agent, values in by_agent.items()}
    counts = Counter(_camp(v) for v in means.values())
    camps = {camp: counts.get(camp, 0) for camp in ("oppose", "neutral", "support")}
    windows = sorted(by_window)
    trend = (
        statistics.mean(by_window[windows[-1]]) - statistics.mean(by_window[windows[0]])
        if len(windows) > 1 else None
    )
    configured = [
        (float(a["sentiment_bias"]) + 1) / 2 for a in config.get("agent_configs", [])
        if isinstance(a.get("sentiment_bias"), (int, float))
    ]
    spread = statistics.pstdev(configured) if len(configured) > 1 else None
    ordered = sorted(means.items(), key=lambda kv: (-kv[1], kv[0]))
    # System One runs log an intent per decision; LLM runs do not. The
    # structured prep records stance_raw; the LLM prep (hybrid) does not.
    # System One writes decisions.jsonl and the LLM path does not; a run whose
    # decisions all failed has no intents but is still a local run (#62).
    local = os.path.exists(os.path.join(sim_dir, "decisions.jsonl"))
    structured = any("stance_raw" in a for a in config.get("agent_configs", []))
    path = ("local" if structured else "hybrid") if local else "llm"

    def evidence(key: str) -> dict[str, str]:
        if path == "llm":
            return {"confidence": "reference", "evidence": "LLM 路徑，評估時的參考路徑"}
        table, label = (SCAN_EVIDENCE, "本機路徑") if path == "local" else (HYBRID_EVIDENCE, "混合模式")
        level, record = table[key]
        return {"confidence": level, "evidence": f"{label}在評估庫 {record} 個情境與 LLM 路徑一致"}

    # Flat LLM-prep stances do not make the roles indistinct: the order comes
    # from the persona text there, so the flag is for the structured prep only.
    indistinct = path == "local" and spread is not None and spread < INDISTINCT_SD
    ranking = {
        "most_supportive": [{"name": n, "stance": _round(v)} for n, v in ordered[:3]],
        "most_opposed": [{"name": n, "stance": _round(v)} for n, v in reversed(ordered[-3:])],
        "configured_spread": _round(spread) if spread is not None else None,
        "indistinct": indistinct,
        **evidence("ranking"),
    }
    if indistinct:
        ranking["confidence"] = "none"
    return {
        "question": question,
        "posts": len(posts),
        "local_path": local,
        "path": path,
        "main_camp": {
            "value": max(camps, key=lambda c: camps[c]),
            "counts": camps,
            **evidence("main_camp"),
        },
        "tendency": {"value": _round(statistics.mean(cache[p["text"]] for p in posts)), **evidence("tendency")},
        "trend": {
            "value": _round(trend) if trend is not None else None,
            "rounds": [windows[0] * TREND_WINDOW, max(p["round"] for p in posts)],
            **evidence("trend"),
        },
        "ranking": ranking,
        "by_role": {name: _round(value) for name, value in sorted(means.items())},
        "post_shares": {camp: _round(sum(1 for v in scored if _camp(v) == camp) / len(scored)) for camp in ("oppose", "neutral", "support")},
        "most_opposed_posts": [
            {"text": t, "stance": _round(cache[t])}
            for t in sorted({p["text"] for p in posts}, key=lambda t: (cache[t], t))[:3]
        ],
    }


def _scan_markdown(scan: dict[str, Any]) -> list[str]:
    def conf(block: dict[str, Any]) -> str:
        return f"可信度：{CONFIDENCE_LABEL.get(block['confidence'], '參考')}（{block['evidence']}）"

    lines = [
        f"## {t('metrics.scanConclusion')}",
        "",
        f"由 {scan['posts']} 則貼文的立場讀出計算（題目：{scan['question']}；0 = 強烈反對，1 = 強烈支持）。",
        "",
    ]
    camp = scan["main_camp"]
    counts = "、".join(f"{CAMP_LABEL[c]} {n}" for c, n in camp["counts"].items())
    lines.append(f"- **主要陣營**：{CAMP_LABEL[camp['value']]}（角色數：{counts}）。{conf(camp)}")
    lines.append(f"- **整體傾向**：{scan['tendency']['value']:.2f}。{conf(scan['tendency'])}")
    trend = scan["trend"]
    if trend["value"] is not None:
        lines.append(f"- **走向**：第 {trend['rounds'][0]}–{trend['rounds'][1]} 輪 {trend['value']:+.2f}。{conf(trend)}")
    ranking = scan["ranking"]
    if ranking["indistinct"]:
        lines.append(
            f"- **角色排序**：無法區分。準備階段給各角色的立場幾乎相同（標準差 {ranking['configured_spread']:.3f}），"
            "貼文之間的差異主要是雜訊。"
        )
    else:
        top = "、".join(f"{r['name']} {r['stance']:.2f}" for r in ranking["most_supportive"])
        bottom = "、".join(f"{r['name']} {r['stance']:.2f}" for r in ranking["most_opposed"])
        lines.append(f"- **角色排序**：最支持 {top}；最反對 {bottom}。{conf(ranking)}")
    if scan.get("path", "local" if scan["local_path"] else "llm") == "local":
        lines += [
            "",
            "本機路徑的主要陣營可信；走向與角色排序在評估中只有約一半的情境和 LLM 路徑一致。"
            "要依這兩項下判斷，請用 `MIROFISH_PROFILE=local-hybrid` 或 `local-llm` 重跑確認。",
        ]
    elif scan.get("path") == "hybrid":
        lines += [
            "",
            "混合模式（LLM 準備、本機模擬）的整體傾向與走向在評估中大多和 LLM 路徑一致，角色排序約一半；"
            "主要陣營在「多數中立」的情境會偏中立（LLM 路徑偏支持）。重要的判斷請用 `MIROFISH_PROFILE=local-llm` 重跑確認。",
        ]
    return lines + [""]


# --------------------------------------------------------------- markdown


def _final_emotions(curve: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | None:
    if not curve:
        return None
    last = max(curve, key=int)
    return last, curve[last]


def decision_error_rate(sim_dir: str) -> float | None:
    """Share of System One decisions that failed (``error`` rows); None when
    the run logged no decisions (the LLM path)."""

    rows = _jsonl(os.path.join(sim_dir, "decisions.jsonl"))
    if not rows:
        return None
    return sum(1 for row in rows if row.get("error")) / len(rows)


ERROR_RATE_WARNING = 0.2


def render_markdown(metrics: dict[str, Any], summary: str = "") -> str:
    lines = [f"# {t('metrics.reportTitle')}", ""]
    rate = metrics.get("decision_error_rate")
    if rate is not None and rate > ERROR_RATE_WARNING:
        lines += [f"> ⚠️ {t('metrics.decisionErrorWarning', rate=f'{rate:.0%}')}", ""]
    if summary:
        lines += [f"## {t('metrics.summary')}", "", summary.strip(), ""]
    lines += [
        f"## {t('metrics.overview')}",
        "",
        f"- Agent 數：{metrics['agents']}",
        f"- 實體類型：{', '.join(f'{k} {v}' for k, v in metrics['entity_types'].items()) or '—'}",
        f"- 熱門話題：{', '.join(metrics['hot_topics']) or '—'}",
        "",
    ]
    if metrics.get("scan"):
        lines += _scan_markdown(metrics["scan"])
    elif metrics.get("scan_error"):
        lines += [
            f"## {t('metrics.scanConclusion')}",
            "",
            t("metrics.scanUnavailable", reason=metrics["scan_error"]),
            "",
        ]
    lines += [
        f"## {t('metrics.actionDistribution')}",
        "",
    ]
    for platform, data in metrics["actions"].items():
        total = sum(data["total"].values())
        parts = ", ".join(
            f"{a} {n}（{n / total:.0%}）" for a, n in sorted(data["total"].items(), key=lambda x: -x[1])
        )
        lines.append(f"- **{platform}**（{total} 個動作）：{parts}")
    lines += ["", f"## {t('metrics.emotionCurve')}", ""]
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
    lines += ["", f"## {t('metrics.topPosts')}", ""]
    for platform, posts in metrics["spread"].items():
        for post in posts:
            chain = " → ".join(step["by"] for step in post["chain"][:6])
            lines.append(
                f"- [{platform} #{post['post_id']}] {post['author']}：{post['content'][:60]}"
                f"（轉發／引用 {post['reposts_and_quotes']}，讚 {post['likes']}）"
                + (f"；擴散路徑：{chain}" if chain else "")
            )
    lines += ["", f"## {t('metrics.stanceClustering')}", ""]
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
    if metrics.get("scan"):
        compact["scan"] = metrics["scan"]
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
    score_fn: ScoreFn | None = None,
) -> tuple[dict[str, Any], str]:
    """Compute metrics, write the two files, return (metrics, markdown).

    With ``score_fn`` the posts are scored for the scan conclusions; when that
    fails (no model server) the section states the reason."""

    from ..utils.llm_usage import usage_stage

    metrics = compute_metrics(sim_dir)
    metrics["decision_error_rate"] = decision_error_rate(sim_dir)
    if score_fn is not None:
        try:
            metrics["scan"] = scan_conclusions(sim_dir, score_fn)
            if metrics["scan"] is None:
                metrics["scan_error"] = "這次模擬沒有可評分的貼文"
        except Exception as error:  # noqa: BLE001 - the metrics report stands alone
            logger.warning("scan conclusions failed", exc_info=True)
            metrics["scan"] = None
            # Say why instead of leaving the section out (#62).
            metrics["scan_error"] = f"{type(error).__name__}: {error}"
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
    score_fn: ScoreFn | None = None,
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
            score_fn=score_fn if score_fn is not None else default_score_fn(),
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
    title = title_match.group(1).strip() if title_match else t("metrics.reportTitle")
    parts = re.split(r"^## (.+)$", markdown, flags=re.M)
    sections = [
        section_cls(title=parts[i].strip(), content=parts[i + 1].strip())
        for i in range(1, len(parts) - 1, 2)
    ]
    summary = next((s.content.split("\n\n")[0] for s in sections if s.content), "")
    return outline_cls(title=title, summary=summary, sections=sections)
