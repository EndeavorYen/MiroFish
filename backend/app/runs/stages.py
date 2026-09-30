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
from typing import Any, Callable

from .orchestrator import StageContext, StageFailed

POLL_S = 2.0


def _client(ctx: StageContext):
    return ctx.extra["app"].test_client()


def _headers(ctx: StageContext) -> dict[str, str]:
    locale = ctx.params.get("locale")
    return {"Accept-Language": locale} if locale else {}


def _call(ctx: StageContext, method: str, path: str, **kwargs) -> dict[str, Any]:
    response = getattr(_client(ctx), method)(path, headers=_headers(ctx), **kwargs)
    body = response.get_json(silent=True) or {}
    if response.status_code >= 400 or not body.get("success", False):
        raise StageFailed(body.get("error") or f"{method.upper()} {path}: HTTP {response.status_code}")
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
    with open(ctx.params["document_path"], "rb") as f:
        document = f.read()
    data = _call(ctx, "post", "/api/graph/ontology/generate", data={
        "files": (io.BytesIO(document), os.path.basename(ctx.params["document_path"])),
        "simulation_requirement": ctx.params["simulation_requirement"],
        "project_name": ctx.params.get("project_name") or "Run " + ctx.run_id,
    }, content_type="multipart/form-data")
    return {"project_id": data["project_id"]}


def graph(ctx: StageContext) -> dict[str, Any]:
    project_id = ctx.artifacts["project_id"]
    task_id = _call(ctx, "post", "/api/graph/build", json={"project_id": project_id})["task_id"]
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


def simulate(ctx: StageContext) -> dict[str, Any]:
    simulation_id = ctx.artifacts["simulation_id"]
    body = {"simulation_id": simulation_id, "platform": "parallel", "force": True,
            # Parallel seeds must not write agents back into one graph; the
            # metrics report reads the simulation's own files.
            "enable_graph_memory_update": False}
    if ctx.params.get("max_rounds"):
        body["max_rounds"] = int(ctx.params["max_rounds"])
    if ctx.params.get("seed") is not None:
        body["seed"] = int(ctx.params["seed"])
    _call(ctx, "post", "/api/simulation/start", json=body)
    data = _poll(ctx, lambda: _call(ctx, "get", f"/api/simulation/{simulation_id}/run-status"),
                 done=lambda d: d.get("runner_status") == "completed",
                 failed=lambda d: (d.get("error") or f"simulation {d.get('runner_status')}")
                 if d.get("runner_status") in ("failed", "stopped") else None,
                 timeout_s=ctx.extra.get("simulate_timeout_s", 6 * 3600), what="simulation",
                 progress_of=_round_progress)
    return {"rounds": data.get("current_round")}


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


REAL_STAGES = {"ontology": ontology, "graph": graph, "prepare": prepare, "simulate": simulate, "report": report}
