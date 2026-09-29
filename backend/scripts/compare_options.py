"""Compare options on one scenario by the reaction they get (#56).

A decision is rarely "how many will oppose" and usually "which of A, B, C
draws the least opposition". Every option runs with the same seeds, and its
posts are scored on the same question (the base scenario's event), so the
model's own biases largely cancel between options and the per-seed
differences can be paired.

Each option becomes a scenario of its own: the base seed document plus the
option text, and a requirement that names the option. It goes through the
zero-decode template prep, so the option reaches the graph, the prep
stances and the initial posts, then runs on the local path.

``options.json``::

    [{"name": "baseline", "text": "..."}, {"name": "free_trial", "text": "..."}]

The first option is the baseline. Per option the report gives the overall
tendency and the share of opposing posts, the paired difference to the
baseline (mean, spread and how many seeds agree on its sign), and the most
opposed posts. A difference whose sign the seeds do not agree on is
reported as indistinguishable.

Usage:
    uv run python scripts/compare_options.py --fixture tests/fixtures/golden_scenario \\
        --options options.json --out <dir> [--seeds 1 2 3] [--rounds 24]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def load_options(path: Path) -> list[dict[str, str]]:
    options = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(options, list) or len(options) < 2:
        raise SystemExit("options.json needs at least two options; the first is the baseline")
    names = [str(o.get("name") or "") for o in options]
    if any(not n or SLUG_RE.search(n) for n in names) or len(set(names)) != len(names):
        raise SystemExit("option names must be unique and use only letters, digits, _ and -")
    if any(not str(o.get("text") or "").strip() for o in options):
        raise SystemExit("every option needs a text")
    return [{"name": str(o["name"]), "text": str(o["text"]).strip()} for o in options]


def option_fixture(base: Path, option: dict[str, str], out: Path) -> Path:
    """The base scenario with the option added to the seed and the requirement."""

    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.is_file():
            shutil.copy(item, out / item.name)
    seed = (base / "news_seed.txt").read_text(encoding="utf-8").rstrip()
    requirement = (base / "simulation_requirement.txt").read_text(encoding="utf-8").strip()
    english = bool(re.search(r"[A-Za-z]{4,}", requirement)) and not re.search(r"[一-鿿]", requirement)
    label = "Announced plan" if english else "公布的方案"
    (out / "news_seed.txt").write_text(f"{seed}\n\n{label}：{option['text']}\n", encoding="utf-8")
    (out / "simulation_requirement.txt").write_text(
        f"{requirement} {label}：{option['text']}" if english else f"{requirement}{label}：{option['text']}",
        encoding="utf-8",
    )
    return out


def run_measures(scan: dict[str, Any]) -> dict[str, float]:
    return {"tendency": scan["tendency"]["value"], "oppose_share": scan["post_shares"]["oppose"]}


def paired(base: dict[int, dict[str, float]], other: dict[int, dict[str, float]], key: str) -> dict[str, Any]:
    """Per-seed differences other - base on the seeds both have."""

    diffs = [other[s][key] - base[s][key] for s in sorted(set(base) & set(other))]
    if not diffs:
        return {"mean": None, "sd": None, "seeds": 0, "same_sign": 0, "distinct": False}
    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    same = max(positive, negative)
    return {
        "mean": round(statistics.mean(diffs), 4),
        "sd": round(statistics.pstdev(diffs), 4),
        "seeds": len(diffs),
        "same_sign": same,
        # Every seed moves the same way; with 3 seeds that is a 1-in-4 chance
        # under no effect, so use 5 seeds before acting on a small difference.
        "distinct": len(diffs) >= 2 and same == len(diffs),
    }


def compare(results: dict[str, dict[int, dict[str, Any]]], names: list[str]) -> list[dict[str, Any]]:
    """Rows per option from ``results[name][seed] = scan``."""

    measures = {n: {s: run_measures(scan) for s, scan in results[n].items()} for n in names}
    base = names[0]
    rows = []
    for name in names:
        runs = measures[name]
        row: dict[str, Any] = {
            "option": name,
            "seeds": sorted(runs),
            "tendency": round(statistics.mean(r["tendency"] for r in runs.values()), 4),
            "oppose_share": round(statistics.mean(r["oppose_share"] for r in runs.values()), 4),
            "most_opposed_posts": sorted(
                (p for scan in results[name].values() for p in scan["most_opposed_posts"]),
                key=lambda p: (p["stance"], p["text"]),
            )[:3],
        }
        if name != base:
            row["vs_baseline"] = {
                "tendency": paired(measures[base], runs, "tendency"),
                "oppose_share": paired(measures[base], runs, "oppose_share"),
            }
        rows.append(row)
    return rows


def _diff_text(block: dict[str, Any], scale: float = 1.0, unit: str = "") -> str:
    if block["mean"] is None:
        return "—"
    text = f"{block['mean'] * scale:+.2f}{unit}（{block['same_sign']}/{block['seeds']} seed 同向）"
    return text if block["distinct"] else text + "，無法區分"


def render(rows: list[dict[str, Any]], question: str) -> str:
    lines = [
        "# 方案比較",
        "",
        f"每個方案用同一組 seed，貼文都用同一題評分：{question}（0 = 強烈反對，1 = 強烈支持）。",
        "",
        "| 方案 | 整體傾向 | 反對貼文比例 | 傾向差（對基準） | 反對比例差（對基準） |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        vs = row.get("vs_baseline")
        lines.append(
            f"| {row['option']}{'（基準）' if not vs else ''} | {row['tendency']:.2f} | {row['oppose_share']:.0%} "
            f"| {_diff_text(vs['tendency']) if vs else '—'} | {_diff_text(vs['oppose_share'], 100, ' 個百分點') if vs else '—'} |"
        )
    lines += ["", "## 各方案最反對的貼文", ""]
    for row in rows:
        lines.append(f"- **{row['option']}**：" + "；".join(f"「{p['text'][:60]}」（{p['stance']:.2f}）" for p in row["most_opposed_posts"]))
    lines += [
        "",
        "差距只有在每個 seed 都同向時才算可區分。這是本機路徑的結果：主要陣營與整體傾向在評估庫上比較可靠，"
        "要依結果做決定，請把前兩名用 `MIROFISH_PROFILE=local-llm` 重跑確認。模擬結果不是對真實世界的預測。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", type=Path, required=True, help="base scenario (news_seed.txt, simulation_requirement.txt)")
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=24)
    args = parser.parse_args(argv)

    import ab_eval
    import golden_pipeline as gp

    if gp.dotenv_files():
        parser.error("a .env file exists; move it aside (see golden_pipeline.py)")
    options = load_options(args.options)
    base = args.fixture.resolve()
    out = args.out.resolve()
    from app.simulation_policy.tiers import detect_content_lang, event_phrase, stance_check_question

    requirement = (base / "simulation_requirement.txt").read_text(encoding="utf-8").strip()
    question = stance_check_question(event_phrase(requirement), detect_content_lang(requirement))
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for option in options:
        fixture = option_fixture(base, option, out / option["name"] / "fixture")
        work = out / option["name"] / "work"
        if not (work / "prepared.json").exists():
            subprocess.run(
                [sys.executable, str(BACKEND_DIR / "scripts" / "golden_pipeline.py"), "prepare",
                 "--work", str(work), "--prep-mode", "template", "--fixture", str(fixture)],
                cwd=str(BACKEND_DIR), check=True, stdout=subprocess.DEVNULL,
            )
        results[option["name"]] = {}
        for seed in args.seeds:
            run = out / option["name"] / f"seed{seed}"
            ab_eval.run_simulation(work, run, "system_one", seed, args.rounds, "tiered")
            gp.force_local_config(work)
            from app.services.metrics_report import default_score_fn, scan_conclusions

            scan = scan_conclusions(str(run / "sim"), default_score_fn(), question=question)
            if scan:
                results[option["name"]][seed] = scan
    rows = compare(results, [o["name"] for o in options])
    (out / "compare.json").write_text(
        json.dumps({"question": question, "options": options, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text = render(rows, question)
    (out / "compare.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
