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
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

from app.services.seed_consistency import (  # noqa: E402  (re-exported for callers)
    CAMP_LABEL, STABLE_RANKING, aggregate, render, role_means, spearman,
)


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
