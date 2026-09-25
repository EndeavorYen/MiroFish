"""Run the golden scenario end to end on local services and measure it.

Subcommands:

    prepare   ontology -> local graph -> profiles + simulation config
    simulate  run run_parallel_simulation.py on a prepared scenario with a
              decision backend and seed; summarise decode tokens, per-round
              latency, VRAM peak and actions

Everything talks to local servers only (LLM_BASE_URL, SYSTEM_ONE_BASE_URL,
EMBED_BASE_URL); GRAPH_BACKEND is forced to local. All LLM usage lands in
<dir>/metrics/llm_usage.jsonl through MIROFISH_METRICS_DIR.

Example:
    uv run python scripts/golden_pipeline.py prepare --work runs/golden
    uv run python scripts/golden_pipeline.py simulate --work runs/golden \
        --decision-backend system_one --seed 42 --out runs/golden/sim_s1_42
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
# Environment before app.config loads any .env; local settings are derived
# from this snapshot, never from values a .env injected later. Valid only
# when this module is imported before ``app`` (command-line use).
_INITIAL_ENV = dict(os.environ)
FORCED_LOCAL = {
    "GRAPH_BACKEND": "local",
    "GRAPH_EXTRACTOR": "local",
    "EXTRACT_SUMMARY_LLM": "0",
    "SYSTEM_ONE_BACKEND": "local",
}
DROPPED_KEYS = (
    "SYSTEM_ONE_API_KEY",
    "EMBED_API_KEY",
    "ZEP_API_KEY",
    # run_parallel_simulation sends Reddit agents to the boost endpoint when set.
    "LLM_BOOST_API_KEY",
    "LLM_BOOST_BASE_URL",
    "LLM_BOOST_MODEL_NAME",
)


def dotenv_files() -> list[Path]:
    """Every .env that app.config or the simulation scripts could load.

    python-dotenv's ``load_dotenv()`` searches upward from backend/app, and
    the simulation scripts load backend/.env explicitly.
    """

    found = []
    for directory in [BACKEND_DIR / "app", *(BACKEND_DIR / "app").parents]:
        candidate = directory / ".env"
        if candidate.exists():
            found.append(candidate)
    return found
FIXTURE = BACKEND_DIR / "tests" / "fixtures" / "golden_scenario"
LOCAL_DEFAULTS = {
    "LLM_API_KEY": "local",
    "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
    "LLM_MODEL_NAME": "qwen3.5-4b",
    "SYSTEM_ONE_BASE_URL": "http://127.0.0.1:8000/v1",
    "SYSTEM_ONE_MODEL": "qwen3.5-4b",
    "EMBED_BASE_URL": "http://127.0.0.1:8001/v1",
    "EMBED_MODEL_NAME": "intfloat/multilingual-e5-small",
    "GRAPH_EMBEDDER": "http",
}


def force_local_config(work: Path) -> None:
    """Re-assert local settings after ``app.config`` loaded any .env file.

    ``app/config.py`` calls ``load_dotenv(override=True)``, so a repo .env
    could otherwise switch this process back to Zep or a cloud LLM.
    """

    sys.path.insert(0, str(BACKEND_DIR))
    from app.config import Config

    env = local_env(work)
    os.environ.update(env)
    Config.GRAPH_BACKEND = "local"
    Config.GRAPH_EXTRACTOR = "local"
    Config.EXTRACT_SUMMARY_LLM = False
    Config.SYSTEM_ONE_BACKEND = "local"
    Config.SYSTEM_ONE_API_KEY = None
    Config.EMBED_API_KEY = None
    Config.GRAPH_DATA_DIR = env["GRAPH_DATA_DIR"]
    Config.GRAPH_EMBEDDER = env["GRAPH_EMBEDDER"]
    Config.EMBED_BASE_URL = env["EMBED_BASE_URL"]
    Config.EMBED_MODEL_NAME = env["EMBED_MODEL_NAME"]
    Config.LLM_API_KEY = env["LLM_API_KEY"]
    Config.LLM_BASE_URL = env["LLM_BASE_URL"]
    Config.LLM_MODEL_NAME = env["LLM_MODEL_NAME"]
    Config.SYSTEM_ONE_BASE_URL = env["SYSTEM_ONE_BASE_URL"]
    Config.SYSTEM_ONE_MODEL = env["SYSTEM_ONE_MODEL"]


def local_env(work: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Local settings on top of the pre-.env environment.

    A value the caller exported explicitly (e.g. another local port) wins
    over the default; a value that only came from a .env file does not.
    """

    env = dict(os.environ)
    for key, value in LOCAL_DEFAULTS.items():
        env[key] = _INITIAL_ENV.get(key, value)
    env.update(FORCED_LOCAL)
    for key in DROPPED_KEYS:
        env.pop(key, None)
    env["GRAPH_DATA_DIR"] = str(work / "graphs")
    env["MIROFISH_METRICS_DIR"] = str(work / "metrics")
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra or {})
    return env


class VramSampler:
    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def read() -> int | None:
        if not shutil.which("nvidia-smi"):
            return None
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.read()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def usage_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    path = metrics_dir / "llm_usage.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stage_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        stage = totals.setdefault(row["stage"], {"calls": 0, "prompt_tokens": 0, "decode_tokens": 0})
        stage["calls"] += 1
        stage["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        stage["decode_tokens"] += int(row.get("completion_tokens") or 0)
    return totals


# ---------------------------------------------------------------- prepare


def cmd_prepare(args: argparse.Namespace) -> int:
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(local_env(work, {"ONTOLOGY_MODE": args.prep_mode, "PROFILE_MODE": args.prep_mode, "SIM_CONFIG_MODE": args.prep_mode}))
    force_local_config(work)

    from app.graph.store import TextEpisode  # noqa: F401
    from app.services.graph_builder import GraphBuilderService
    from app.services.ontology_generator import OntologyGenerator
    from app.services.simulation_manager import SimulationManager
    from app.services.text_processor import TextProcessor
    from app.utils.llm_usage import usage_stage

    seed_text = (FIXTURE / "news_seed.txt").read_text(encoding="utf-8")
    requirement = (FIXTURE / "simulation_requirement.txt").read_text(encoding="utf-8").strip()
    timings: dict[str, float] = {}

    started = time.perf_counter()
    ontology = OntologyGenerator().generate(
        document_texts=[seed_text], simulation_requirement=requirement
    )
    timings["ontology_s"] = round(time.perf_counter() - started, 1)
    (work / "ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.perf_counter()
    with usage_stage("graph_build"):
        builder = GraphBuilderService()
        graph_id = builder.create_graph("golden")
        builder.set_ontology(graph_id, ontology)
        chunks = TextProcessor.split_text(seed_text, chunk_size=500, overlap=50)
        builder._wait_for_batch(builder.add_text_batches(graph_id, chunks))
        graph = builder.get_graph_data(graph_id)
    timings["graph_build_s"] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    manager = SimulationManager()
    state = manager.create_simulation("proj_golden_scenario", graph_id)
    manager.prepare_simulation(
        state.simulation_id,
        simulation_requirement=requirement,
        document_text=seed_text,
        use_llm_for_profiles=args.prep_mode == "llm",
    )
    timings["prepare_s"] = round(time.perf_counter() - started, 1)
    sim_dir = Path(manager._get_simulation_dir(state.simulation_id))
    shutil.copytree(sim_dir, work / "prepared", dirs_exist_ok=True)

    prepared = {
        "prep_mode": args.prep_mode,
        "graph_id": graph_id,
        "simulation_id": state.simulation_id,
        "prepared_dir": str(work / "prepared"),
        "nodes": graph.get("node_count"),
        "edges": graph.get("edge_count"),
        "timings": timings,
        "usage": stage_totals(usage_rows(work / "metrics")),
        "created": datetime.now().isoformat(),
    }
    (work / "prepared.json").write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------- simulate


def round_latencies(sim_dir: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for platform in ("twitter", "reddit"):
        path = sim_dir / platform / "actions.jsonl"
        if not path.exists():
            continue
        starts: dict[int, datetime] = {}
        durations = []
        # Round 0 is the initial posts; rounds with no active agent are
        # skipped by the runner. Neither is a decision round.
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event_type")
            if event not in ("round_start", "round_end"):
                continue
            stamp = datetime.fromisoformat(row["timestamp"])
            if event == "round_start":
                starts[row["round"]] = stamp
            elif row["round"] in starts and row["round"] > 0 and row.get("actions_count", 1) > 0:
                durations.append((stamp - starts[row["round"]]).total_seconds())
        result[platform] = durations
    return result


def action_counts(sim_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for platform in ("twitter", "reddit"):
        path = sim_dir / platform / "actions.jsonl"
        if not path.exists():
            continue
        platform_counts: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event_type" in row:
                continue
            platform_counts[row.get("action_type", "?")] = platform_counts.get(row.get("action_type", "?"), 0) + 1
        counts[platform] = platform_counts
    return counts


TEXT_KEYS = ("content", "quote_content")


def post_texts(sim_dir: Path) -> list[str]:
    """Text the agents wrote (posts, quotes, comments) from actions.jsonl."""

    texts = []
    for platform in ("twitter", "reddit"):
        path = sim_dir / platform / "actions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event_type" in row or row.get("action_type") not in (
                "CREATE_POST", "QUOTE_POST", "CREATE_COMMENT"
            ):
                continue
            args = row.get("action_args") or {}
            text = args.get("quote_content") or args.get("content")
            if text:
                texts.append(str(text))
    return texts


def entity_names(work: Path, prepared: dict[str, Any]) -> list[str]:
    """Seed entities: ontology-typed nodes of the prepared graph (+ aliases)."""

    import sqlite3

    db = work / "graphs" / f"{prepared['graph_id']}.sqlite"
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT name, labels, attributes FROM nodes").fetchall()
    finally:
        conn.close()
    names = []
    for name, labels, attributes in rows:
        if "Entity" not in json.loads(labels):
            continue
        names.append(name)
        names.extend(json.loads(attributes).get("aliases") or [])
    return [n for n in names if len(n) >= 2]


def content_stats(texts: list[str], names: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(BACKEND_DIR))
    from app.simulation_policy.tiers import distinct_2

    with_entity = sum(1 for t in texts if any(n in t for n in names))
    return {
        "posts": len(texts),
        "distinct_2": round(distinct_2(texts), 4),
        "entity_mention_rate": round(with_entity / len(texts), 4) if texts else None,
    }


def content_tiers(sim_dir: Path) -> dict[str, Any]:
    tiers = {"template": 0, "shared": 0, "full": 0}
    decode = 0
    rounds = 0
    for path in sim_dir.glob("content_metrics_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rounds += 1
            decode += row.get("decode_tokens_total", 0)
            for tier, n in row.get("tiers", {}).items():
                tiers[tier] = tiers.get(tier, 0) + n
    total = sum(tiers.values())
    return {
        "tiers": tiers,
        "tier_share": {k: round(v / total, 4) for k, v in tiers.items()} if total else {},
        "content_decode_tokens": decode,
        "platform_rounds": rounds,
    }


def replay_graph_writeback(run: Path, prepared: dict[str, Any]) -> dict[str, Any]:
    """Feed the run's actions to the memory updater on the local graph."""

    force_local_config(run)
    from app.services.zep_graph_memory_updater import ZepGraphMemoryUpdater
    from app.utils.llm_usage import usage_stage

    before = len(usage_rows(run / "metrics"))
    updater = ZepGraphMemoryUpdater(prepared["graph_id"], simulation_id=prepared["simulation_id"])
    updater._running = True
    fed = 0
    with usage_stage("simulation", metrics_dir=str(run / "metrics")):
        for platform in ("twitter", "reddit"):
            path = run / "sim" / platform / "actions.jsonl"
            if not path.exists():
                continue
            batch = []
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if "event_type" in row:
                    continue
                updater.add_activity_from_dict(row, platform)
                fed += 1
            while not updater._activity_queue.empty():
                batch.append(updater._activity_queue.get_nowait())
            updater._send_batch_activities(batch, platform)
    updater._running = False
    after = len(usage_rows(run / "metrics"))
    return {"actions_fed": fed, "llm_usage_rows_added": after - before, "stats": updater.get_stats()}


def cmd_simulate(args: argparse.Namespace) -> int:
    work = Path(args.work).resolve()
    prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
    run = Path(args.out).resolve()
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(prepared["prepared_dir"], run / "sim")
    # A private copy of the graph so writeback replays do not mix runs.
    shutil.copytree(work / "graphs", run / "graphs")
    env = local_env(run, {"SIM_DECISION_BACKEND": args.decision_backend})
    if args.content_mode:
        env["CONTENT_MODE"] = args.content_mode
    if args.content_budget is not None:
        env["CONTENT_DECODE_BUDGET_PER_ROUND"] = str(args.content_budget)
    config = run / "sim" / "simulation_config.json"
    command = [
        sys.executable,
        str(BACKEND_DIR / "scripts" / "run_parallel_simulation.py"),
        "--config", str(config),
        "--max-rounds", str(args.max_rounds),
        "--no-wait",
        "--seed", str(args.seed),
        "--decision-backend", args.decision_backend,
    ]
    vram_before = VramSampler.read()
    started = time.perf_counter()
    with VramSampler() as vram, open(run / "simulation.log", "w", encoding="utf-8") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(BACKEND_DIR))
    elapsed = time.perf_counter() - started

    rows = usage_rows(run / "metrics")
    totals = stage_totals(rows)
    latencies = round_latencies(run / "sim")
    rounds = max((len(v) for v in latencies.values()), default=0)
    sim_decode = totals.get("simulation", {}).get("decode_tokens", 0)
    summary: dict[str, Any] = {
        "decision_backend": args.decision_backend,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "exit_code": completed.returncode,
        "elapsed_s": round(elapsed, 1),
        "usage": totals,
        "simulation_decode_tokens": sim_decode,
        "simulation_decode_per_round": round(sim_decode / rounds, 1) if rounds else None,
        "rounds_logged": {k: len(v) for k, v in latencies.items()},
        "round_latency_s": {
            k: {
                "mean": round(statistics.mean(v), 2) if v else None,
                "p50": round(statistics.median(v), 2) if v else None,
                "max": round(max(v), 2) if v else None,
            }
            for k, v in latencies.items()
        },
        "actions": action_counts(run / "sim"),
        "content": content_stats(post_texts(run / "sim"), entity_names(work, prepared)),
        "content_tiers": content_tiers(run / "sim"),
        "vram_mib": {
            "before": vram_before,
            "peak": max(vram.samples) if vram.samples else None,
        },
    }
    if args.graph_writeback:
        summary["graph_writeback"] = replay_graph_writeback(run, prepared)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if completed.returncode == 0 else completed.returncode


def cmd_summarize(args: argparse.Namespace) -> int:
    """Recompute content stats for an existing run (e.g. a baseline)."""

    work = Path(args.work).resolve()
    prepared = json.loads((work / "prepared.json").read_text(encoding="utf-8"))
    run = Path(args.out).resolve()
    result = {
        "content": content_stats(post_texts(run / "sim"), entity_names(work, prepared)),
        "content_tiers": content_tiers(run / "sim"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--allow-dotenv",
        action="store_true",
        help="run even if a repo .env exists (the simulation subprocess re-loads it)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--work", required=True)
    prep.add_argument("--prep-mode", choices=["llm", "template"], default="llm")
    sim = sub.add_parser("simulate")
    sim.add_argument("--work", required=True)
    sim.add_argument("--out", required=True)
    sim.add_argument("--decision-backend", choices=["llm", "system_one"], default="llm")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--max-rounds", type=int, default=12)
    sim.add_argument("--content-mode", default=None, help="tiered | template (system_one only)")
    sim.add_argument("--content-budget", type=int, default=None)
    sim.add_argument("--graph-writeback", action="store_true")
    summ = sub.add_parser("summarize")
    summ.add_argument("--work", required=True)
    summ.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    dotenvs = dotenv_files()
    if dotenvs and not args.allow_dotenv:
        # app.config loads .env with override=True in every process,
        # including the simulation subprocess; refuse rather than risk cloud
        # endpoints.
        parser.error(
            f"{', '.join(map(str, dotenvs))} exists; move it aside or pass --allow-dotenv"
        )
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "summarize":
        return cmd_summarize(args)
    return cmd_simulate(args)


if __name__ == "__main__":
    raise SystemExit(main())
