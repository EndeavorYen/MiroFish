"""Peak memory and start-up time of one simulation process (#65).

Copies a prepared simulation's inputs to a scratch directory, runs
``run_parallel_simulation.py`` on them for a few rounds and samples the
resident memory of the process tree. Prints one JSON line:

    peak_rss_mb      highest resident memory of the process tree
    first_round_s    launch -> the first platform logs ``round_start`` 1
    wall_s           launch -> exit
    returncode

The model service the simulation uses must be up (a local profile needs
llama-server on :8000). ``SIM_RECSYS`` is passed through, so

    SIM_RECSYS=light uv run python scripts/bench_memory.py uploads/simulations/<id>
    SIM_RECSYS=oasis uv run python scripts/bench_memory.py uploads/simulations/<id>

compare the two recommenders.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

BACKEND_DIR = Path(__file__).resolve().parents[1]
INPUTS = ("simulation_config.json", "twitter_profiles.csv", "reddit_profiles.json")


def tree_rss(process: psutil.Process) -> int:
    total = 0
    for proc in [process, *process.children(recursive=True)]:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass
    return total


def first_round_logged(sim_dir: Path) -> bool:
    for platform in ("twitter", "reddit"):
        path = sim_dir / platform / "actions.jsonl"
        if path.exists() and '"round": 1, ' in path.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def bench(source: Path, rounds: int, seed: int, interval: float = 0.1) -> dict:
    work = Path(tempfile.mkdtemp(prefix="bench_memory_"))
    try:
        for name in INPUTS:
            if (source / name).exists():
                shutil.copy2(source / name, work / name)
        command = [
            sys.executable, str(BACKEND_DIR / "scripts" / "run_parallel_simulation.py"),
            "--config", str(work / "simulation_config.json"),
            "--max-rounds", str(rounds), "--no-wait", "--seed", str(seed),
        ]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        started = time.perf_counter()
        with open(work / "bench.log", "w", encoding="utf-8") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(BACKEND_DIR))
            process = psutil.Process(child.pid)
            peak, first_round = 0, None
            while child.poll() is None:
                peak = max(peak, tree_rss(process))
                if first_round is None and first_round_logged(work):
                    first_round = time.perf_counter() - started
                time.sleep(interval)
        wall = time.perf_counter() - started
        if first_round is None and first_round_logged(work):
            first_round = wall
        tail = (work / "bench.log").read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        return {
            "recsys": os.environ.get("SIM_RECSYS") or "oasis",
            "peak_rss_mb": round(peak / 2**20),
            "first_round_s": None if first_round is None else round(first_round, 1),
            "wall_s": round(wall, 1),
            "rounds": rounds,
            "returncode": child.returncode,
            **({} if child.returncode == 0 else {"log_tail": tail}),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("simulation_dir", type=Path, help="a prepared simulation (its config and profiles)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = bench(args.simulation_dir.resolve(), args.rounds, args.seed)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
