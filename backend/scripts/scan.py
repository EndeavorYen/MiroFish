"""Scan prepared scenarios on the local path and say which to confirm (#53).

The local path (System One decisions, tiered content) costs 5-10% of the LLM
path's decode, so a scenario can be run on several seeds for the price of
one LLM run. This runs each prepared scenario on ``--seeds``, computes the
metrics report's scan conclusions for every run, and reports per scenario:

* the main camp and how many seeds agree on it;
* overall tendency and trend, mean and spread over the seeds, and whether
  the trend keeps its sign;
* the role ranking averaged over seeds, and how stable it is (Spearman
  between the per-role means of two halves of the seeds);
* whether to confirm with ``MIROFISH_PROFILE=local-llm``: seeds disagree on
  the main camp or the trend's sign, the roles cannot be told apart, the
  ranking is unstable, or (always) when trend or ranking will be used, since
  those matched the LLM path in only about half of the evaluation scenarios.

Seed agreement shows the local path's own noise; it does not show that the
local path agrees with the LLM path (a scenario can be stable and still
ranked differently, chengchuan in #53).

Usage:
    uv run python scripts/scan.py --work <prepared dir> [<prepared dir> ...] \\
        --out <dir> [--seeds 1 2 3] [--rounds 24]

A ``<prepared dir>`` is ``golden_pipeline.py prepare --prep-mode template``
output. Finished runs are reused.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

CAMP_LABEL = {"oppose": "反對", "neutral": "中立", "support": "支持"}
STABLE_RANKING = 0.5


def spearman(x: list[float], y: list[float]) -> float | None:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                result[order[k]] = (i + j) / 2
            i = j + 1
        return result

    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def role_means(scans: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for scan in scans:
        for name, stance in scan["by_role"].items():
            values.setdefault(name, []).append(stance)
    return {name: statistics.mean(v) for name, v in values.items()}


def aggregate(name: str, scans: list[dict[str, Any]]) -> dict[str, Any]:
    """One scenario's conclusions over its seeds (``scans`` from scan_conclusions)."""

    camps = Counter(scan["main_camp"]["value"] for scan in scans)
    camp, agree = camps.most_common(1)[0]
    tendencies = [scan["tendency"]["value"] for scan in scans]
    trends = [scan["trend"]["value"] for scan in scans if scan["trend"]["value"] is not None]
    signs = Counter((t > 0) - (t < 0) for t in trends)
    means = role_means(scans)
    half = len(scans) // 2
    stability = None
    if half:
        first, second = role_means(scans[:half]), role_means(scans[half:])
        common = sorted(set(first) & set(second))
        stability = spearman([first[n] for n in common], [second[n] for n in common])
    indistinct = any(scan["ranking"]["indistinct"] for scan in scans)
    reasons = []
    if agree < len(scans):
        reasons.append(f"主要陣營只有 {agree}/{len(scans)} 個 seed 一致")
    if trends and max(signs.values()) < len(trends):
        reasons.append("走向的正負在 seed 之間不一致")
    if indistinct:
        reasons.append("角色立場無法區分")
    elif stability is not None and stability < STABLE_RANKING:
        reasons.append(f"角色排序在 seed 之間不穩（{stability:.2f}）")
    ordered = sorted(means.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "scenario": name,
        "seeds": len(scans),
        "main_camp": {"value": camp, "agree": agree},
        "tendency": {"mean": round(statistics.mean(tendencies), 4), "sd": round(statistics.pstdev(tendencies), 4)},
        "trend": {
            "mean": round(statistics.mean(trends), 4) if trends else None,
            "sd": round(statistics.pstdev(trends), 4) if trends else None,
        },
        "ranking": {
            "most_supportive": [n for n, _ in ordered[:3]],
            "most_opposed": [n for n, _ in reversed(ordered[-3:])],
            "stability": round(stability, 4) if stability is not None else None,
            "indistinct": indistinct,
        },
        "confirm": reasons,
    }


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 本機掃描",
        "",
        "| 情境 | 主要陣營（一致 seed） | 整體傾向 | 走向 | 角色排序穩定度 | 建議 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        trend = row["trend"]
        trend_text = "—" if trend["mean"] is None else f"{trend['mean']:+.2f} ± {trend['sd']:.2f}"
        ranking = row["ranking"]
        stable = "無法區分" if ranking["indistinct"] else (
            "—" if ranking["stability"] is None else f"{ranking['stability']:.2f}"
        )
        advice = (
            "用 local-llm 確認：" + "；".join(row["confirm"]) if row["confirm"]
            else "seed 間一致；主要陣營、整體傾向可用（走向、角色排序仍以 local-llm 為準）"
        )
        lines.append(
            f"| {row['scenario']} | {CAMP_LABEL[row['main_camp']['value']]}（{row['main_camp']['agree']}/{row['seeds']}） "
            f"| {row['tendency']['mean']:.2f} ± {row['tendency']['sd']:.2f} | {trend_text} | {stable} | {advice} |"
        )
    lines += [
        "",
        "seed 之間一致，只代表本機路徑自己穩定，不代表和 LLM 路徑一致。評估庫上，本機路徑的主要陣營 6/6 與 LLM 路徑一致，"
        "整體傾向與走向約一半，角色排序 2/6（docs/local-first.md 的 G5）。要依走向或角色排序下判斷，"
        "請用 `MIROFISH_PROFILE=local-llm` 重跑確認。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=24)
    args = parser.parse_args(argv)

    import ab_eval
    import golden_pipeline as gp

    if gp.dotenv_files():
        parser.error("a .env file exists; move it aside (see golden_pipeline.py)")
    rows = []
    for work in args.work:
        work = work.resolve()
        scans = []
        for seed in args.seeds:
            run = args.out.resolve() / work.name / f"seed{seed}"
            ab_eval.run_simulation(work, run, "system_one", seed, args.rounds, "tiered")
            gp.force_local_config(work)
            from app.services.metrics_report import default_score_fn, scan_conclusions

            scan = scan_conclusions(str(run / "sim"), default_score_fn())
            if scan:
                scans.append(scan)
        if scans:
            rows.append(aggregate(work.name, scans))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scan.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render(rows)
    (args.out / "scan.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
