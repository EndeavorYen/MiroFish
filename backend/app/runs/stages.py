"""The real stages: the endpoints the UI calls, in-process (#63).

Each stage calls the existing API through the app's test client, exactly as
the frontend does (same validation, locks, background tasks and payloads),
then polls the matching status endpoint. A non-2xx answer or
``success: false`` stops the run with the endpoint's error. A deliberate
seam: moving the endpoints' logic into service functions is a later #68
step, not this one.
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime
from typing import Any, Callable

from .orchestrator import StageContext, StageFailed

POLL_S = 2.0


def _client(ctx: StageContext):
    return ctx.extra["app"].test_client()


def _headers(ctx: StageContext) -> dict[str, str]:
    locale = ctx.params.get("locale")
    return {"Accept-Language": locale} if locale else {}


def _request(ctx: StageContext, method: str, path: str, **kwargs) -> tuple[int, dict[str, Any]]:
    response = getattr(_client(ctx), method)(path, headers=_headers(ctx), **kwargs)
    return response.status_code, response.get_json(silent=True) or {}


def _call(ctx: StageContext, method: str, path: str, **kwargs) -> dict[str, Any]:
    status, body = _request(ctx, method, path, **kwargs)
    if status >= 400 or not body.get("success", False):
        raise StageFailed(body.get("error") or f"{method.upper()} {path}: HTTP {status}")
    return body.get("data") or {}


def _default_progress(data: dict[str, Any]) -> tuple[Any, Any]:
    return data.get("progress"), data.get("message")


def _poll(ctx: StageContext, fetch: Callable[[], dict[str, Any]], done: Callable[[dict], bool],
          failed: Callable[[dict], str | None], timeout_s: float, what: str,
          progress_of: Callable[[dict], tuple[Any, Any]] = _default_progress) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_progress = None
    interval = ctx.extra.get("poll_s", POLL_S)
    while True:
        data = fetch()
        reason = failed(data)
        if reason:
            raise StageFailed(reason)
        if done(data):
            return data
        progress = progress_of(data)
        if progress != last_progress and any(progress):
            ctx.emit("progress", {"progress": progress[0], "message": progress[1]})
            last_progress = progress
        if time.monotonic() > deadline:
            raise StageFailed(f"{what} did not finish within {int(timeout_s)} s")
        time.sleep(interval)


def ontology(ctx: StageContext) -> dict[str, Any]:
    name = ctx.params.get("project_name") or "Run " + ctx.run_id
    earlier = ctx.artifacts.get("ontology_started_at")
    if earlier:  # cut off before: reuse the project it finished, delete the rest
        found = _projects_since(ctx, name, earlier)
        keep = next((p for p in found if p.get("status") == "ontology_generated"), None)
        for project in found:
            if project is not keep:
                _request(ctx, "delete", f"/api/graph/project/{project['project_id']}")
        if keep:
            return {"project_id": keep["project_id"]}
    # Project.created_at is local time; recorded before the call, so a resume finds what it created.
    started = datetime.now().isoformat()
    ctx.store.update(ctx.run_id, artifacts={"ontology_started_at": started})
    with open(ctx.params["document_path"], "rb") as f:
        document = f.read()
    data = _call(ctx, "post", "/api/graph/ontology/generate", data={
        "files": (io.BytesIO(document), os.path.basename(ctx.params["document_path"])),
        "simulation_requirement": ctx.params["simulation_requirement"],
        "project_name": name,
    }, content_type="multipart/form-data")
    return {"project_id": data["project_id"]}


def _projects_since(ctx: StageContext, name: str, since: str) -> list[dict[str, Any]]:
    """Projects named ``name`` created at or after ``since``, newest first."""

    projects = _call(ctx, "get", "/api/graph/project/list?limit=500")
    projects = projects if isinstance(projects, list) else []
    found = [p for p in projects if p.get("name") == name and (p.get("created_at") or "") >= since]
    return sorted(found, key=lambda p: p.get("created_at") or "", reverse=True)


def graph(ctx: StageContext) -> dict[str, Any]:
    project_id = ctx.artifacts["project_id"]
    status, body = _request(ctx, "post", "/api/graph/build", json={"project_id": project_id})
    if status == 409 and body.get("recoverable"):
        # A restart lost the build task; the endpoint wants an explicit rebuild.
        status, body = _request(ctx, "post", "/api/graph/build", json={"project_id": project_id, "force": True})
    if status >= 400 or not body.get("success", False):
        raise StageFailed(body.get("error") or f"POST /api/graph/build: HTTP {status}")
    data = body.get("data") or {}
    if data.get("reused"):
        # Already built: the old task id may be gone after a restart.
        project = _call(ctx, "get", f"/api/graph/project/{project_id}")
        if project.get("status") == "graph_completed" and project.get("graph_id"):
            return {"graph_id": project["graph_id"]}
    task_id = data["task_id"]
    _poll(ctx, lambda: _call(ctx, "get", f"/api/graph/task/{task_id}"),
          done=lambda d: d.get("status") == "completed",
          failed=lambda d: (d.get("error") or d.get("message") or "graph build failed") if d.get("status") == "failed" else None,
          timeout_s=ctx.extra.get("graph_timeout_s", 3600), what="graph build")
    graph_id = _call(ctx, "get", f"/api/graph/project/{project_id}").get("graph_id")
    if not graph_id:
        raise StageFailed("graph build finished without a graph id")
    return {"graph_id": graph_id}


def prepare(ctx: StageContext) -> dict[str, Any]:
    simulation_id = ctx.artifacts.get("simulation_id") or _call(ctx, "post", "/api/simulation/create", json={
        "project_id": ctx.artifacts["project_id"],
        "graph_id": ctx.artifacts["graph_id"],
        "enable_twitter": True,
        "enable_reddit": True,
    })["simulation_id"]
    ctx.store.update(ctx.run_id, artifacts={"simulation_id": simulation_id})  # a resume reuses it
    data = _call(ctx, "post", "/api/simulation/prepare", json={
        "simulation_id": simulation_id, "use_llm_for_profiles": True, "parallel_profile_count": 5})
    if not data.get("already_prepared"):
        task_id = data.get("task_id")
        _poll(ctx, lambda: _call(ctx, "post", "/api/simulation/prepare/status",
                                 json={"task_id": task_id, "simulation_id": simulation_id}),
              done=lambda d: d.get("status") in ("completed", "ready"),
              failed=lambda d: (d.get("error") or d.get("message") or "preparation failed") if d.get("status") == "failed" else None,
              timeout_s=ctx.extra.get("prepare_timeout_s", 3600), what="preparation")
    return {"simulation_id": simulation_id}


def seed_values(params: dict[str, Any]) -> list[int]:
    first = params.get("seed")
    first = 1 if first is None else int(first)
    return [first + i for i in range(int(params.get("seeds") or 1))]


def parallel_limit(count: int) -> int:
    """How many seed simulations may run at once: each takes about
    ``RUNS_SIM_MEMORY_MB`` of host memory, and ``RUNS_MEMORY_RESERVE_MB`` is
    left for the model service, the OS and the browser."""

    per_sim = float(os.environ.get("RUNS_SIM_MEMORY_MB", 500))
    reserve = float(os.environ.get("RUNS_MEMORY_RESERVE_MB", 2048))
    try:
        import psutil

        available = psutil.virtual_memory().available / 2**20
    except Exception:  # unknown: one at a time
        return 1
    return max(1, min(count, int((available - reserve) // per_sim)))


def simulate(ctx: StageContext) -> dict[str, Any]:
    """One simulation per seed. The first seed uses the prepared simulation;
    the others are copies of it, so preparation runs once."""

    seeds = seed_values(ctx.params)
    runs = list(ctx.artifacts.get("seed_simulations") or [])
    known = {r["seed"] for r in runs}
    for seed in seeds:
        if seed in known:
            continue
        simulation_id = ctx.artifacts["simulation_id"] if not runs else _call(
            ctx, "post", "/api/simulation/copy", json={"simulation_id": ctx.artifacts["simulation_id"]})["simulation_id"]
        runs.append({"seed": seed, "simulation_id": simulation_id})
        ctx.store.update(ctx.run_id, artifacts={"seed_simulations": runs})  # a resume reuses the copies

    def status(run):
        return _call(ctx, "get", f"/api/simulation/{run['simulation_id']}/run-status")

    pending = [r for r in runs if status(r).get("runner_status") != "completed"]  # a resume skips finished seeds
    limit = parallel_limit(len(pending))
    if pending:
        ctx.emit("progress", {"progress": 0, "message": f"{len(runs)} seeds, {limit} at a time"})
    going: list[dict[str, Any]] = []
    deadline = time.monotonic() + ctx.extra.get("simulate_timeout_s", 6 * 3600)
    interval = ctx.extra.get("poll_s", POLL_S)
    last_progress = None
    while pending or going:
        while pending and len(going) < limit:
            run = pending.pop(0)
            _start_simulation(ctx, run)
            going.append(run)
        states = {r["simulation_id"]: status(r) for r in runs}
        for run in list(going):
            state = states[run["simulation_id"]]
            if state.get("runner_status") in ("failed", "stopped"):
                raise StageFailed(f"seed {run['seed']}: " + (state.get("error") or f"simulation {state.get('runner_status')}"))
            if state.get("runner_status") == "completed":
                going.remove(run)
        progress = _seeds_progress(list(states.values()))
        if progress != last_progress and any(progress):
            ctx.emit("progress", {"progress": progress[0], "message": progress[1]})
            last_progress = progress
        if not (pending or going):
            break
        if time.monotonic() > deadline:
            raise StageFailed(f"simulation did not finish within {int(ctx.extra.get('simulate_timeout_s', 6 * 3600))} s")
        time.sleep(interval)
    return {"seed_simulations": runs, "rounds": states[runs[0]["simulation_id"]].get("current_round") if runs else None}


def _start_simulation(ctx: StageContext, run: dict[str, Any]) -> None:
    body = {"simulation_id": run["simulation_id"], "platform": "parallel", "force": True,
            # Parallel seeds must not write agents back into one graph; the
            # metrics report reads the simulation's own files.
            "enable_graph_memory_update": False, "seed": run["seed"]}
    if ctx.params.get("max_rounds"):
        body["max_rounds"] = int(ctx.params["max_rounds"])
    _call(ctx, "post", "/api/simulation/start", json=body)


def _seeds_progress(states: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Rounds over all seeds; with one seed, the same as ``_round_progress``."""

    if len(states) == 1:
        return _round_progress(states[0])
    current = sum(s.get("current_round") or 0 for s in states)
    total = sum(s.get("total_rounds") or 0 for s in states)
    if not total:
        return None, None
    done = sum(s.get("runner_status") == "completed" for s in states)
    return round(100 * current / total), f"round {current}/{total} over {len(states)} seeds, {done} done"


def _round_progress(data: dict[str, Any]) -> tuple[Any, Any]:
    """run-status reports rounds, not a percentage."""

    current, total = data.get("current_round") or 0, data.get("total_rounds") or 0
    if not total:
        return None, None
    return round(100 * current / total), f"round {current}/{total}"


def report(ctx: StageContext) -> dict[str, Any]:
    simulation_id = ctx.artifacts["simulation_id"]
    data = _call(ctx, "post", "/api/report/generate", json={"simulation_id": simulation_id, "force_regenerate": True})
    report_id, task_id = data.get("report_id"), data.get("task_id")
    if task_id:
        _poll(ctx, lambda: _call(ctx, "post", "/api/report/generate/status",
                                 json={"task_id": task_id, "simulation_id": simulation_id}),
              done=lambda d: d.get("status") == "completed",
              failed=lambda d: (d.get("error") or d.get("message") or "report failed") if d.get("status") == "failed" else None,
              timeout_s=ctx.extra.get("report_timeout_s", 3600), what="report")
    return {"report_id": report_id}


def consistency(ctx: StageContext) -> dict[str, Any]:
    """How far the seeds agree (main camp, tendency, trend, role ranking);
    ``scripts/scan.py``'s conclusions, for this run. One seed: nothing to
    compare."""

    runs = ctx.artifacts.get("seed_simulations") or []
    if len(runs) < 2:
        return {"consistency": None}
    from ..services.metrics_report import default_score_fn, scan_conclusions
    from ..services.seed_consistency import aggregate
    from ..services.simulation_manager import SimulationManager

    score_fn, scans = default_score_fn(), []
    for i, run in enumerate(runs, 1):
        ctx.emit("progress", {"progress": round(100 * (i - 1) / len(runs)), "message": f"seed {i}/{len(runs)}"})
        scan = scan_conclusions(os.path.join(SimulationManager.SIMULATION_DATA_DIR, run["simulation_id"]), score_fn)
        if scan:
            scans.append(scan)
    if len(scans) < 2:
        raise StageFailed(f"only {len(scans)} of {len(runs)} seeds have posts to compare")
    return {"consistency": aggregate(ctx.run_id, scans)}


REAL_STAGES = {"ontology": ontology, "graph": graph, "prepare": prepare, "simulate": simulate, "report": report,
               "consistency": consistency}
