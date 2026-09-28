"""Scan-tool gates G5 over finished A/B reports (#43).

The local path is a cheap scanning tool, so G5 checks the directional
conclusions a scan is used for, with absolute tolerances (thresholds fixed
in docs/local-first.md before any G5 number was computed):

* persona_rank: Spearman of per-persona mean stance, B vs A, >= 0.5
  (undecidable below 12 shared personas);
* camp: the largest camp (oppose < 0.4 <= neutral <= 0.6 < support) matches;
* lean: mean post stance within 0.10;
* trend: last minus first 4-round window (windows both groups cover),
  within 0.10;
* cost: G4's decode ratio and VRAM gates.

Action JS and stance-distribution JS are reported only. Reads
``ab_report.json``; no simulation or scoring is rerun.

Usage:
    uv run python scripts/scan_gates.py <suite-dir> [--out scan_report.md]
    uv run python scripts/scan_gates.py --report <ab_report.json> ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

MIN_PERSONAS = 12
RANK_MIN = 0.5
LEAN_TOL = 0.10
TREND_TOL = 0.10
SCAN_READY_SHARE = 0.8
REPORT_JS = 0.05


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = _ranks(list(x)), _ranks(list(y))
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if not sx or not sy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def camp(persona_means: dict[str, float]) -> str:
    counts = {"oppose": 0, "neutral": 0, "support": 0}
    for value in persona_means.values():
        counts["oppose" if value < 0.4 else "support" if value > 0.6 else "neutral"] += 1
    return max(counts, key=lambda k: (counts[k], k == "neutral"))


def _mean_personas(runs: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for run in runs:
        for name, value in (run.get("stance_by_persona") or {}).items():
            by.setdefault(name, []).append(float(value))
    return {name: statistics.mean(v) for name, v in by.items()}


def _lean(runs: list[dict]) -> float | None:
    leans = []
    for run in runs:
        counts = list((run.get("stance_levels") or {}).values())
        total = sum(counts)
        if total:
            leans.append(sum(i * c for i, c in enumerate(counts)) / (4 * total))
    return statistics.mean(leans) if leans else None


def _windows(runs: list[dict]) -> list[float | None]:
    curves = [run.get("stance_curve_windowed") or [] for run in runs]
    width = max((len(c) for c in curves), default=0)
    windows = []
    for i in range(width):
        values = [c[i] for c in curves if i < len(c) and c[i] is not None]
        windows.append(statistics.mean(values) if values else None)
    return windows


def _trends(a_runs: list[dict], b_runs: list[dict]) -> tuple[float | None, float | None]:
    """Last minus first window, over the windows both groups have posts in.
    A group whose first posts land a window earlier would otherwise be
    compared from a different starting point."""

    wa, wb = _windows(a_runs), _windows(b_runs)
    both = [i for i in range(min(len(wa), len(wb))) if wa[i] is not None and wb[i] is not None]
    if len(both) < 2:
        return None, None
    return wa[both[-1]] - wa[both[0]], wb[both[-1]] - wb[both[0]]


def _status(ok: bool | None) -> str:
    return "undecidable" if ok is None else ("pass" if ok else "fail")


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    a_runs = list(report["groups"]["A"].values())
    b_runs = list(report["groups"]["B"].values())
    pa, pb = _mean_personas(a_runs), _mean_personas(b_runs)
    common = sorted(set(pa) & set(pb))
    rank = spearman([pa[n] for n in common], [pb[n] for n in common]) if len(common) >= 2 else None
    rank_ok = None if len(common) < MIN_PERSONAS or rank is None else rank >= RANK_MIN
    camp_a, camp_b = camp({n: pa[n] for n in common}), camp({n: pb[n] for n in common})
    lean_a, lean_b = _lean(a_runs), _lean(b_runs)
    trend_a, trend_b = _trends(a_runs, b_runs)
    g4 = report.get("gates", {})
    cost_ok = bool(g4.get("decode_ratio", {}).get("passed")) and bool(g4.get("vram", {}).get("passed"))
    action_js = (g4.get("action_js") or {}).get("b_vs_a")
    levels_js = (g4.get("stance_distribution") or {}).get("b_vs_a")
    return {
        "persona_rank": {"value": rank, "common_personas": len(common), "threshold": RANK_MIN, "status": _status(rank_ok)},
        "camp": {"a": camp_a, "b": camp_b, "status": _status(camp_a == camp_b)},
        "lean": {"a": lean_a, "b": lean_b, "tolerance": LEAN_TOL,
                 "status": _status(None if None in (lean_a, lean_b) else abs(lean_a - lean_b) <= LEAN_TOL)},
        "trend": {"a": trend_a, "b": trend_b, "tolerance": TREND_TOL,
                  "status": _status(None if None in (trend_a, trend_b) else abs(trend_a - trend_b) <= TREND_TOL)},
        "cost": {"decode_ratio": g4.get("decode_ratio", {}).get("value"),
                 "vram_mib": g4.get("vram", {}).get("peak_mib"), "status": _status(cost_ok)},
        "report_only": {"action_js": action_js, "stance_distribution_js": levels_js,
                        "action_js_within": action_js is not None and action_js <= REPORT_JS,
                        "stance_distribution_within": levels_js is not None and levels_js <= REPORT_JS},
    }


GATES = ("persona_rank", "camp", "lean", "trend", "cost")


def scenario_status(gates: dict[str, Any]) -> str:
    return "fail" if any(gates[g]["status"] == "fail" for g in GATES) else "pass"


def suite_verdict(statuses: list[str]) -> str:
    if not statuses:
        return "not_ready"
    share = sum(s == "pass" for s in statuses) / len(statuses)
    return "scan_ready" if share >= SCAN_READY_SHARE else "not_ready"


def _f(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


MARK = {"pass": "✅", "fail": "❌", "undecidable": "無法判讀"}


def render(rows: list[tuple[str, dict[str, Any]]], verdict: str) -> str:
    lines = [
        "| 情境 | 結果 | 角色排序 | 主要陣營 B／A | 整體傾向 B／A | 走向 B／A | 成本 | 動作 JS | 立場分布 JS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, g in rows:
        lines.append(
            f"| {name} | {MARK[scenario_status(g)]} "
            f"| {_f(g['persona_rank']['value'])} {MARK[g['persona_rank']['status']]} "
            f"| {g['camp']['b']}／{g['camp']['a']} {MARK[g['camp']['status']]} "
            f"| {_f(g['lean']['b'])}／{_f(g['lean']['a'])} {MARK[g['lean']['status']]} "
            f"| {_f(g['trend']['b'])}／{_f(g['trend']['a'])} {MARK[g['trend']['status']]} "
            f"| {_f(g['cost']['decode_ratio'])} {MARK[g['cost']['status']]} "
            f"| {_f(g['report_only']['action_js'])} | {_f(g['report_only']['stance_distribution_js'])} |"
        )
    passed = sum(scenario_status(g) == "pass" for _, g in rows)
    lines += ["", f"通過 {passed}/{len(rows)}，判定：{verdict}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", nargs="?", type=Path, help="dir with <scenario>/runs/ab_report.json")
    parser.add_argument("--report", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    paths = list(args.report)
    if args.suite:
        paths += sorted(args.suite.glob("*/runs/ab_report.json"))
    rows = [(p.parent.parent.name, evaluate(json.loads(p.read_text(encoding="utf-8")))) for p in paths]
    verdict = suite_verdict([scenario_status(g) for _, g in rows])
    text = render(rows, verdict)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps({"verdict": verdict, "scenarios": dict(rows)}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
