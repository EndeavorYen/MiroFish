"""A/B fidelity evaluation on the golden scenario (#13).

Group A (baseline): LLM decisions on the LLM-prepared scenario.
Group B (local):    System One decisions + tiered content on the
                    template/structured-prepared scenario.
Both run on the local graph and the local model server (no external API).

For each group and seed, ``golden_pipeline.py simulate`` is run (or an
existing run is reused), then this script computes:

* decode tokens per round, round latency, VRAM peak;
* Jensen-Shannon divergence of action-type distributions: A seed vs A seed
  (the noise floor) and B vs A;
* stance curves: every post of every run is scored with the same System One
  ``score`` question, averaged per round; B's mean curve is correlated with
  A's (and A seeds with each other);
* distinct-2 and seed-entity mention rate of posts;
* extraction recall from #7 (passed in).

It writes ``ab_report.json`` and ``ab_report.md`` with the G4 gate result and
a recommendation (switch defaults, switch conditionally, or keep llm).

Runs are reused when their summary.json exists, so each group can first be
simulated under its own model-server settings with ``--simulate-only A|B``
(OASIS LLM agents in A need about 32K context per slot).

Usage:
    uv run python scripts/ab_eval.py --work-a <llm prepared> --work-b <template prepared> \
        --out <dir> --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import golden_pipeline as gp  # noqa: E402  (captures the pre-.env environment)

STANCE_LEVELS = ["強烈反對", "反對", "中立", "支持", "強烈支持"]
STANCE_QUESTION = "這則貼文對「東海市無人駕駛空中計程車試點」的立場是什麼？"
VRAM_BUDGET_MIB = 10 * 1024


# ------------------------------------------------------------------ maths


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, 0..1) of two count dicts."""

    keys = sorted(set(p) | set(q))
    ps = sum(p.values()) or 1.0
    qs = sum(q.values()) or 1.0
    pv = [p.get(k, 0) / ps for k in keys]
    qv = [q.get(k, 0) / qs for k in keys]
    mv = [(a + b) / 2 for a, b in zip(pv, qv)]

    def kl(x, y):
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0)

    return 0.5 * kl(pv, mv) + 0.5 * kl(qv, mv)


def pearson(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy)


# ------------------------------------------------------------------- runs


def run_simulation(work: Path, out: Path, backend: str, seed: int, rounds: int, content: str | None) -> dict:
    summary = out / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    command = [
        sys.executable, str(BACKEND_DIR / "scripts" / "golden_pipeline.py"), "simulate",
        "--work", str(work), "--out", str(out), "--decision-backend", backend,
        "--seed", str(seed), "--max-rounds", str(rounds),
    ]
    if content:
        command += ["--content-mode", content]
    subprocess.run(command, cwd=str(BACKEND_DIR), check=True, stdout=subprocess.DEVNULL)
    return json.loads(summary.read_text(encoding="utf-8"))


def llm_errors(run: Path) -> int:
    """Model-server errors (context overflow, 5xx) the simulation logged.

    A run with errors lost agent turns, so its numbers are not comparable.
    """

    path = run / "simulation.log"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if re.search(r"Error code: [45]\d\d", line))


def action_counts(run: Path) -> Counter:
    counts: Counter = Counter()
    for platform, per_type in gp.action_counts(run / "sim").items():
        for action, n in per_type.items():
            counts[f"{platform}:{action}"] += n
    return counts


def posts_by_round(run: Path) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for platform in ("twitter", "reddit"):
        path = run / "sim" / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "event_type" in row or row.get("action_type") not in ("CREATE_POST", "QUOTE_POST", "CREATE_COMMENT"):
                continue
            args = row.get("action_args") or {}
            text = args.get("quote_content") or args.get("content")
            if text:
                result.setdefault(int(row.get("round", 0)), []).append(str(text))
    return result


def stance_curve(client, run: Path, rounds: int, cache: dict[str, float]) -> list[float | None]:
    from app.system_one.models import ScoreQuestion, SystemOneRequest

    curve: list[float | None] = []
    by_round = posts_by_round(run)
    for r in range(1, rounds + 1):
        scores = []
        for text in by_round.get(r, []):
            if text not in cache:
                answer = client.ask(
                    SystemOneRequest(
                        state=f"貼文：{text[:300]}",
                        questions={"s": ScoreQuestion(instructions=STANCE_QUESTION, criteria=STANCE_LEVELS)},
                    )
                ).answers["s"]
                cache[text] = answer.score / (len(STANCE_LEVELS) - 1)
            scores.append(cache[text])
        curve.append(statistics.mean(scores) if scores else None)
    return curve


def mean_curve(curves: list[list[float | None]]) -> list[float | None]:
    out = []
    for values in zip(*curves):
        present = [v for v in values if v is not None]
        out.append(statistics.mean(present) if present else None)
    return out


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-a", required=True, help="golden_pipeline prepare --prep-mode llm")
    parser.add_argument("--work-b", required=True, help="golden_pipeline prepare --prep-mode template")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--simulate-only", choices=["A", "B"], default=None,
                        help="only run this group's simulations (e.g. to switch server settings between groups)")
    parser.add_argument("--extraction-recall", type=float, default=None,
                        help="entity recall from #7 (scripts/eval_local_extraction.py)")
    args = parser.parse_args(argv)
    if gp.dotenv_files():
        parser.error("a .env file exists; move it aside (see golden_pipeline.py)")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = {
        "A": {"work": Path(args.work_a).resolve(), "backend": "llm", "content": None},
        "B": {"work": Path(args.work_b).resolve(), "backend": "system_one", "content": "tiered"},
    }
    runs: dict[str, dict[int, dict[str, Any]]] = {"A": {}, "B": {}}
    for name, group in groups.items():
        if args.simulate_only and name != args.simulate_only:
            continue
        for seed in args.seeds:
            run_dir = out / f"{name}_seed{seed}"
            summary = run_simulation(group["work"], run_dir, group["backend"], seed, args.rounds, group["content"])
            runs[name][seed] = {"dir": run_dir, "summary": summary}
    if args.simulate_only:
        return 0

    gp.force_local_config(out)
    from app.system_one.client import get_system_one_client

    client = get_system_one_client()
    cache: dict[str, float] = {}
    per_run: dict[str, dict[int, dict[str, Any]]] = {"A": {}, "B": {}}
    for name in runs:
        work = groups[name]["work"]
        prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
        names = gp.entity_names(work, prepared)
        for seed, info in runs[name].items():
            summary = info["summary"]
            rounds = max(summary.get("rounds_logged", {}).values() or [0]) or args.rounds
            texts = [t for ts in posts_by_round(info["dir"]).values() for t in ts]
            latencies = [
                v.get("mean") for v in summary.get("round_latency_s", {}).values() if v.get("mean") is not None
            ]
            per_run[name][seed] = {
                "decode_tokens": summary["simulation_decode_tokens"],
                "decode_per_round": summary["simulation_decode_tokens"] / args.rounds,
                "active_rounds": rounds,
                "round_latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
                "elapsed_s": summary["elapsed_s"],
                "llm_errors": llm_errors(info["dir"]),
                "vram_peak_mib": summary["vram_mib"]["peak"],
                "actions": action_counts(info["dir"]),
                "content": gp.content_stats(texts, names),
                "stance_curve": stance_curve(client, info["dir"], args.rounds, cache),
            }

    a_runs = list(per_run["A"].values())
    b_runs = list(per_run["B"].values())
    js_aa = [js_divergence(x["actions"], y["actions"]) for x, y in itertools.combinations(a_runs, 2)]
    js_ba = [js_divergence(b["actions"], a["actions"]) for b in b_runs for a in a_runs]
    a_curve = mean_curve([r["stance_curve"] for r in a_runs])
    b_curve = mean_curve([r["stance_curve"] for r in b_runs])
    corr_aa = [pearson(x["stance_curve"], y["stance_curve"]) for x, y in itertools.combinations(a_runs, 2)]
    corr_aa = [c for c in corr_aa if c is not None]
    corr_ab = pearson(a_curve, b_curve)

    def mean_of(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(values) if values else None

    a_decode = mean_of(a_runs, "decode_per_round")
    b_decode = mean_of(b_runs, "decode_per_round")
    js_noise = statistics.mean(js_aa) if js_aa else None
    js_b = statistics.mean(js_ba) if js_ba else None
    # The gate is about the local path fitting a 10 GB card; A may run on a
    # server with larger per-slot contexts (OASIS LLM agents need ~32K).
    vram_peak = max(r["vram_peak_mib"] or 0 for r in b_runs)
    vram_peak_a = max(r["vram_peak_mib"] or 0 for r in a_runs)
    gates = {
        "decode_ratio": {
            "value": (b_decode / a_decode) if a_decode else None,
            "threshold": 0.10,
            "passed": a_decode is not None and b_decode <= 0.10 * a_decode,
        },
        "action_js": {
            "b_vs_a": js_b,
            "a_seed_noise": js_noise,
            "threshold": "≤ 2 × A noise",
            "passed": js_b is not None and js_noise is not None and js_b <= 2 * js_noise,
        },
        "stance_correlation": {
            "value": corr_ab,
            "a_seed_pairs_mean": statistics.mean(corr_aa) if corr_aa else None,
            "threshold": 0.5,
            "passed": corr_ab is not None and corr_ab >= 0.5,
        },
        "vram": {
            "peak_mib": vram_peak,
            "a_peak_mib": vram_peak_a,
            "budget_mib": VRAM_BUDGET_MIB,
            "passed": vram_peak <= VRAM_BUDGET_MIB,
        },
    }
    passed = [k for k, v in gates.items() if v["passed"]]
    if len(passed) == len(gates):
        recommendation = "switch"
    elif gates["decode_ratio"]["passed"] and gates["vram"]["passed"] and (
        gates["action_js"]["passed"] or gates["stance_correlation"]["passed"]
    ):
        recommendation = "conditional"
    else:
        recommendation = "keep_llm"

    def strip(rows):
        return {
            seed: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in row.items()}
            for seed, row in rows.items()
        }

    report = {
        "seeds": args.seeds,
        "rounds": args.rounds,
        "groups": {name: strip(rows) for name, rows in per_run.items()},
        "summary": {
            "A": {
                "decode_per_round": a_decode,
                "round_latency_mean_s": mean_of(a_runs, "round_latency_mean_s"),
                "distinct_2": statistics.mean(r["content"]["distinct_2"] for r in a_runs),
                "entity_mention_rate": statistics.mean(r["content"]["entity_mention_rate"] or 0 for r in a_runs),
            },
            "B": {
                "decode_per_round": b_decode,
                "round_latency_mean_s": mean_of(b_runs, "round_latency_mean_s"),
                "distinct_2": statistics.mean(r["content"]["distinct_2"] for r in b_runs),
                "entity_mention_rate": statistics.mean(r["content"]["entity_mention_rate"] or 0 for r in b_runs),
            },
            "stance_curve_A": a_curve,
            "stance_curve_B": b_curve,
            "extraction_recall": args.extraction_recall,
        },
        "gates": gates,
        "runs_with_llm_errors": sorted(
            f"{name}_seed{seed}" for name, rows in per_run.items() for seed, row in rows.items() if row["llm_errors"]
        ),
        "recommendation": recommendation,
    }
    (out / "ab_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "ab_report.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(report: dict[str, Any]) -> str:
    s, g = report["summary"], report["gates"]
    labels = {"switch": "切換預設值", "conditional": "有條件切換", "keep_llm": "維持 llm"}
    lines = [
        "# A/B 忠實度評估（golden scenario）",
        "",
        f"- seeds：{', '.join(map(str, report['seeds']))}；每組 {len(report['seeds'])} 次；每次 {report['rounds']} 回合",
        "- A：LLM 決策（LLM 準備）；B：System One 決策＋分層內容（模板／結構化準備）；兩組都用本機圖譜與本機模型",
        "",
        "## 指標",
        "",
        "| 指標 | A | B |",
        "| --- | --- | --- |",
        f"| 每回合 decode tokens | {_fmt(s['A']['decode_per_round'], 1)} | {_fmt(s['B']['decode_per_round'], 1)} |",
        f"| 有動作回合平均延遲 (s) | {_fmt(s['A']['round_latency_mean_s'], 1)} | {_fmt(s['B']['round_latency_mean_s'], 1)} |",
        f"| distinct-2 | {_fmt(s['A']['distinct_2'])} | {_fmt(s['B']['distinct_2'])} |",
        f"| 種子實體出現率 | {_fmt(s['A']['entity_mention_rate'])} | {_fmt(s['B']['entity_mention_rate'])} |",
        f"| 抽取實體召回（#7） | — | {_fmt(s['extraction_recall'])} |",
        "",
        "## 閘門 G4",
        "",
        "| 條件 | 數值 | 門檻 | 結果 |",
        "| --- | --- | --- | --- |",
        f"| B／A 每回合 decode | {_fmt(g['decode_ratio']['value'])} | ≤ 0.10 | {'✅' if g['decode_ratio']['passed'] else '❌'} |",
        f"| 動作分布 JS（B vs A） | {_fmt(g['action_js']['b_vs_a'])}（A 組間 {_fmt(g['action_js']['a_seed_noise'])}） | ≤ 2 × A 組間 | {'✅' if g['action_js']['passed'] else '❌'} |",
        f"| 立場曲線相關係數（B vs A） | {_fmt(g['stance_correlation']['value'])}（A 組間平均 {_fmt(g['stance_correlation']['a_seed_pairs_mean'])}） | ≥ 0.5 | {'✅' if g['stance_correlation']['passed'] else '❌'} |",
        f"| VRAM 峰值（B） | {g['vram']['peak_mib']} MiB（A {g['vram'].get('a_peak_mib', '—')} MiB） | ≤ {g['vram']['budget_mib']} MiB | {'✅' if g['vram']['passed'] else '❌'} |",
        "",
        f"## 建議：{labels[report['recommendation']]}",
        "",
    ]
    if report.get("runs_with_llm_errors"):
        lines += [
            f"⚠️ 下列 run 有模型伺服器錯誤，數字不可比：{', '.join(report['runs_with_llm_errors'])}",
            "",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
