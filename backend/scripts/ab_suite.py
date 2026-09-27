"""Batch A/B over the frozen scenario library (#44).

For each manifest entry: reuse or prepare group A (llm) and group B
(template), run ``ab_eval.py``, then write ``suite_report.json`` and
``suite_report.md``. A finished simulation is reused by ``ab_eval``;
a prepared dir is reused when its ``fixture_digest`` matches.

``--quick`` is ``--seeds 1 2 3 --rounds 12``.

Usage:
    uv run python scripts/ab_suite.py --out <dir>
    uv run python scripts/ab_suite.py --out <dir> --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import ab_eval  # noqa: E402
import golden_pipeline as gp  # noqa: E402

MANIFEST = BACKEND_DIR / "tests" / "fixtures" / "scenarios" / "suite.json"
GATE_KEYS = ("decode_ratio", "action_js", "stance_by_persona", "stance_distribution", "vram")


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["scenarios"])


def fixture_dir(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path


def verify_digest(entry: dict[str, Any]) -> Path:
    """Fixture directory. SystemExit when the recorded digest does not match."""

    fixture = fixture_dir(entry)
    digest = gp.scenario_digest(fixture)
    if digest != entry.get("digest"):
        raise SystemExit(f"{entry.get('name')} digest mismatch: {digest} != {entry.get('digest')}")
    return fixture


def ensure_prepared(work: Path, fixture: Path, mode: str, digest: str) -> dict[str, Any]:
    """Reuse ``prepared.json`` when the digest matches; refuse a different one."""

    path = work / "prepared.json"
    if path.exists():
        prepared = json.loads(path.read_text(encoding="utf-8"))
        if prepared.get("fixture_digest") != digest:
            raise SystemExit(
                f"{work} was prepared from digest {prepared.get('fixture_digest')}, not {digest}"
            )
        return prepared
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(BACKEND_DIR / "scripts" / "golden_pipeline.py"), "prepare",
            "--work", str(work), "--prep-mode", mode, "--fixture", str(fixture),
        ],
        cwd=str(BACKEND_DIR),
        check=True,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_status(gates: dict[str, dict]) -> str:
    """``pass`` when every gate is ``pass`` or ``undecidable``."""

    for gate in gates.values():
        status = gate.get("status")
        if status is None:
            status = "pass" if gate.get("passed") else "fail"
        if status == "undecidable":
            continue
        if status != "pass":
            return "fail"
    return "pass"


def suite_recommendation(statuses: list[str]) -> str:
    """``switch`` at >= 80% scenarios passing, ``conditional`` at >= 50%."""

    if not statuses:
        return "keep_llm"
    ratio = sum(status == "pass" for status in statuses) / len(statuses)
    if ratio >= 0.8:
        return "switch"
    if ratio >= 0.5:
        return "conditional"
    return "keep_llm"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(bounds: Any) -> str:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return "—"
    return f"[{float(bounds[0]):.3f}, {float(bounds[1]):.3f}]"


def _gate_status(gate: dict[str, Any] | None) -> str:
    if not gate:
        return "—"
    status = gate.get("status")
    if status:
        return str(status)
    if gate.get("passed") is None:
        return "—"
    return "pass" if gate.get("passed") else "fail"


def render_suite(rows: list[dict[str, Any]], recommendation: str, elapsed_s: float) -> str:
    passed = sum(1 for row in rows if row.get("status") == "pass")
    lines = [
        "# 情境庫 A/B 彙總（G4 v2）",
        "",
        "| 情境 | 結果 | 動作 JS | 動作 JS 95% CI | 立場相關 | 立場分布 JS | 共同角色 | 耗時 (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        gates = row.get("gates") or {}
        action = gates.get("action_js") or {}
        persona = gates.get("stance_by_persona") or {}
        levels = gates.get("stance_distribution") or {}
        ci = row.get("b_vs_a_ci")
        if ci is None:
            ci = action.get("b_vs_a_ci")
        lines.append(
            "| {name} | {status} | {js} | {ci} | {persona} | {levels} | {common} | {elapsed} |".format(
                name=row.get("name", "—"),
                status=row.get("status", "—"),
                js=_fmt(action.get("b_vs_a")),
                ci=_ci(ci),
                persona=_fmt(persona.get("value")),
                levels=_fmt(levels.get("b_vs_a")),
                common=persona.get("common_personas", row.get("common_personas", "—")),
                elapsed=_fmt(row.get("elapsed_s"), 1),
            )
        )
    lines += [
        "",
        "## 總表",
        "",
        f"- 通過情境：{passed}/{len(rows)}",
        f"- 建議：{recommendation}",
        f"- 總耗時：{elapsed_s:.0f}s",
        "",
        "閘門狀態（pass / fail / undecidable）：",
        "",
    ]
    for row in rows:
        gates = row.get("gates") or {}
        bits = " ".join(f"{key}={_gate_status(gates.get(key))}" for key in GATE_KEYS)
        lines.append(f"- {row.get('name', '—')}：{bits}")
    lines.append("")
    return "\n".join(lines)


def paired_difference_interval(
    baseline: list[float],
    new: list[float],
    *,
    higher_is_better: bool = True,
    reps: int = 1000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """95% interval of the mean paired difference, new minus baseline."""

    import random

    diffs = [float(after) - float(before) for before, after in zip(baseline, new)]
    if not higher_is_better:
        diffs = [-diff for diff in diffs]
    rng = random.Random(rng_seed)
    means = [
        statistics.mean(rng.choice(diffs) for _ in diffs)
        for _ in range(reps)
    ]
    return ab_eval.bootstrap_ci(means)


def screen_decision(lo: float, hi: float) -> str:
    """One of improve, worsen, inconclusive. Zero inside the interval is inconclusive."""

    if lo > 0:
        return "improve"
    if hi < 0:
        return "worsen"
    return "inconclusive"


def screen_rows(path: Path) -> str:
    """Decisions for saved per-seed metric rows. One line per row: ``name decision``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = []
    for row in payload["rows"]:
        lo, hi = paired_difference_interval(
            row["baseline"],
            row["new"],
            higher_is_better=bool(row.get("higher_is_better", True)),
        )
        lines.append(f"{row['name']} {screen_decision(lo, hi)}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--screen-rows",
        default=None,
        help="JSON of saved baseline/new metric rows; print improve, worsen, or inconclusive",
    )
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=ab_eval.BOOTSTRAP_REPS)
    parser.add_argument("--delta", type=float, default=ab_eval.PERSONA_DELTA)
    args = parser.parse_args(argv)
    if not args.screen_rows and not args.out:
        parser.error("--out is required")
    if args.quick:
        args.seeds = [1, 2, 3]
        args.rounds = 12
    return args


def _row_from_report(name: str, report: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    gates = report["gates"]
    return {
        "name": name,
        "status": scenario_status(gates),
        "gates": gates,
        "common_personas": gates.get("stance_by_persona", {}).get("common_personas"),
        "elapsed_s": round(elapsed_s, 1),
        "b_vs_a_ci": (gates.get("action_js") or {}).get("b_vs_a_ci"),
        "runs_with_llm_errors": report.get("runs_with_llm_errors") or [],
        "recommendation": report.get("recommendation"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.screen_rows:
        print(screen_rows(Path(args.screen_rows)), end="")
        return 0
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()
    if args.scenarios:
        wanted = set(args.scenarios)
        entries = [entry for entry in entries if entry["name"] in wanted]
        missing = wanted - {entry["name"] for entry in entries}
        if missing:
            raise SystemExit(f"unknown scenarios: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for entry in entries:
        fixture = verify_digest(entry)
        name = entry["name"]
        digest = entry["digest"]
        ensure_prepared(out / name / "A", fixture, "llm", digest)
        ensure_prepared(out / name / "B", fixture, "template", digest)
        run_out = out / name / "runs"
        scenario_started = time.perf_counter()
        ab_eval.main([
            "--work-a", str(out / name / "A"),
            "--work-b", str(out / name / "B"),
            "--out", str(run_out),
            "--seeds", *[str(seed) for seed in args.seeds],
            "--rounds", str(args.rounds),
            "--bootstrap-reps", str(args.bootstrap_reps),
            "--delta", str(args.delta),
        ])
        report = json.loads((run_out / "ab_report.json").read_text(encoding="utf-8"))
        rows.append(_row_from_report(name, report, time.perf_counter() - scenario_started))
    elapsed = time.perf_counter() - started
    recommendation = suite_recommendation([row["status"] for row in rows])
    payload = {
        "recommendation": recommendation,
        "elapsed_s": round(elapsed, 1),
        "passed": sum(1 for row in rows if row["status"] == "pass"),
        "total": len(rows),
        "scenarios": rows,
    }
    (out / "suite_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = render_suite(rows, recommendation, elapsed)
    (out / "suite_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
