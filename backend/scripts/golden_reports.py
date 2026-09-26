"""Generate metrics-mode and ReportAgent reports for a golden run (#12).

For a run made by ``golden_pipeline.py simulate``:

1. replay the run's actions into its local graph (structured facts, so the
   ReportAgent's graph tools see what happened in the simulation);
2. ``--mode metrics``: deterministic metrics report + one short summary;
3. ``--mode agent``: the existing ReportAgent on the local model.

Each mode's llm_usage rows go to ``<run>/reports/<mode>/metrics`` so the
``report`` stage decode can be compared.

Usage:
    uv run python scripts/golden_reports.py --work <work> --run <run> --mode metrics
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import golden_pipeline as gp  # noqa: E402  (captures the pre-.env environment)

sys.path.insert(0, str(BACKEND_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=["metrics", "agent"], required=True)
    parser.add_argument("--skip-writeback", action="store_true")
    args = parser.parse_args(argv)
    if gp.dotenv_files():
        parser.error("a .env file exists; move it aside (see golden_pipeline.py)")

    work = Path(args.work).resolve()
    # A bare run name is relative to the work dir, not the cwd.
    run = Path(args.run) if Path(args.run).is_absolute() else work / args.run
    run = run.resolve()
    if not (run / "sim").is_dir():
        parser.error(f"{run} has no sim/ directory; is it a golden_pipeline run?")
    prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
    fixture = Path(prepared.get("fixture") or gp.FIXTURE)
    requirement = (fixture / "simulation_requirement.txt").read_text(encoding="utf-8").strip()
    out = run / "reports" / args.mode
    if out.exists():
        shutil.move(str(out), f"{out}.old{int(time.time())}")
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True)

    gp.force_local_config(run)
    writeback = None
    marker = run / "reports" / "writeback.json"
    if not args.skip_writeback and not marker.exists():
        writeback = gp.replay_graph_writeback(run, prepared)
        marker.write_text(json.dumps(writeback, ensure_ascii=False, default=str), encoding="utf-8")

    # All usage of this report goes to its own metrics dir.
    os.environ["MIROFISH_METRICS_DIR"] = str(metrics_dir)
    started = time.perf_counter()
    if args.mode == "metrics":
        from app.services.metrics_report import default_summary_fn, write_metrics_report

        _, markdown = write_metrics_report(
            str(run / "sim"),
            str(out),
            requirement,
            summary_fn=default_summary_fn(),
            metrics_dir=str(metrics_dir),
        )
        (out / "report.md").write_text(markdown, encoding="utf-8")
    else:
        from app.config import Config
        from app.services.report_agent import ReportAgent, ReportManager

        Config.UPLOAD_FOLDER = str(out / "uploads")
        ReportManager.REPORTS_DIR = str(out / "uploads" / "reports")
        agent = ReportAgent(
            graph_id=prepared["graph_id"],
            simulation_id=prepared["simulation_id"],
            simulation_requirement=requirement,
        )
        report = agent.generate_report()
        (out / "report.md").write_text(report.markdown_content or f"(failed: {report.error})", encoding="utf-8")
    elapsed = time.perf_counter() - started

    rows = gp.usage_rows(metrics_dir)
    summary = {
        "mode": args.mode,
        "seconds": round(elapsed, 1),
        "usage": gp.stage_totals(rows),
        "report_chars": len((out / "report.md").read_text(encoding="utf-8")),
        "writeback": writeback,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
