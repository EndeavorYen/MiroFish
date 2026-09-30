"""Run the A/B simulations of several prepared scenario dirs in one pool.

Each ``<dir>`` holds ``A/`` and ``B/`` (prepared work dirs) and ``runs/``,
as ``ab_suite.py`` lays them out. All missing simulations of all dirs go
into one pool of ``--jobs`` workers, so the model server is kept busy
across scenarios; then ``ab_eval.py`` scores each dir, reusing the runs.
Arguments after ``--`` go to every ``ab_eval.py`` call (e.g. ``--scorer``).

Usage:
    uv run python scripts/ab_pool.py --dirs <d1> <d2> ... --seeds 1 2 3 --jobs 12 [-- --scorer phi]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
import ab_eval  # noqa: E402
import ab_suite  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    dirs = [d.resolve() for d in args.dirs]
    started = time.perf_counter()
    tasks = [
        task for d in dirs
        for task in ab_suite.suite_tasks(d.parent, [d.name], args.seeds, args.rounds)
    ]
    ab_eval.simulate_many(tasks, args.jobs)
    simulated = time.perf_counter() - started
    for d in dirs:
        ab_eval.main([
            "--work-a", str(d / "A"), "--work-b", str(d / "B"), "--out", str(d / "runs"),
            "--seeds", *map(str, args.seeds), "--rounds", str(args.rounds), *extra,
        ])
    print(f"{len(tasks)} simulations in {simulated:.0f} s, scored in {time.perf_counter() - started - simulated:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
