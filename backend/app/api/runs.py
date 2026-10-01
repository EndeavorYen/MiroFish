"""Runs API: one call for the whole flow, progress over SSE (#63).

POST /api/runs                 start a run (document + requirement) -> 202 {run_id}
GET  /api/runs                 recent runs
GET  /api/runs/<id>            status, stage, artifacts, error
GET  /api/runs/<id>/events     Server-Sent Events: stage_start, progress,
                               stage_done, run_done, run_failed, interrupted
POST /api/runs/<id>/resume     continue a failed or interrupted run from the
                               first stage it did not finish
POST /api/runs/<id>/confirm    the roles are confirmed: continue a run paused
                               with ``confirm_roles``
POST /api/runs/<id>/rerun      the same input in another mode, to confirm a run
"""

from __future__ import annotations

import json
import os
import shutil
import time

from flask import Response, current_app, request, stream_with_context

from . import runs_bp
from .. import run_mode
from ..config import Config
from ..runs.orchestrator import RunBusy
from ..runs.service import get_runs
from ..runs.store import AWAITING, FAILED, INTERRUPTED, TERMINAL
from ..utils.locale import t
from ..utils.logger import get_logger
from ..utils.model_health import slot_context

logger = get_logger("mirofish.api.runs")

HEARTBEAT_S = 15.0
ALLOWED_SUFFIXES = (".txt", ".md", ".markdown", ".pdf")
DEFAULT_SEEDS, MAX_SEEDS = 3, 8
LLM_SLOT_CONTEXT = 8192


def _json(payload, status=200):
    return current_app.response_class(json.dumps(payload, ensure_ascii=False), status=status,
                                      mimetype="application/json")


def _error(message: str, status: int):
    return _json({"success": False, "error": message}, status)


def _integer(value) -> int:
    """An int, or the digits of a form field; not a float or a bool."""

    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(value)
    return int(value)


def _run_dir(run_id: str) -> str:
    path = os.path.join(Config.UPLOAD_FOLDER, "runs", run_id)
    os.makedirs(path, exist_ok=True)
    return path


@runs_bp.route("", methods=["POST"])
def create_run():
    """multipart: ``file`` or ``document_text``; ``simulation_requirement``
    (required); optional ``max_rounds`` (default 24), ``seeds`` (how many,
    default 3), ``seed`` (the first one, default 1), ``confirm_roles``
    (pause after the graph), ``project_name``, ``mode`` (local, local-hybrid
    or local-llm; default the backend's ``MIROFISH_PROFILE``)."""

    form = request.form if request.files or request.form else (request.get_json(silent=True) or {})
    requirement = form.get("simulation_requirement") or ""
    text = form.get("document_text")
    if not isinstance(requirement, str) or not isinstance(text, (str, type(None))):
        return _error("simulation_requirement and document_text must be strings", 400)
    requirement = requirement.strip()
    if not requirement:
        return _error("simulation_requirement is required", 400)
    upload = request.files.get("file")
    if not upload and not (text and text.strip()):
        return _error("a file or document_text is required", 400)
    if upload and not upload.filename.lower().endswith(ALLOWED_SUFFIXES):
        return _error(f"file must be one of {', '.join(ALLOWED_SUFFIXES)}", 400)
    try:
        max_rounds = _integer(form.get("max_rounds") or 24)
        seed = None if form.get("seed") in (None, "") else _integer(form.get("seed"))
        seeds = DEFAULT_SEEDS if form.get("seeds") in (None, "") else _integer(form.get("seeds"))
    except (TypeError, ValueError):
        return _error("max_rounds, seeds and seed must be integers", 400)
    if max_rounds <= 0:
        return _error("max_rounds must be positive", 400)
    if not 1 <= seeds <= MAX_SEEDS:
        return _error(f"seeds must be between 1 and {MAX_SEEDS}", 400)
    confirm_roles = form.get("confirm_roles")
    if isinstance(confirm_roles, str):
        confirm_roles = confirm_roles.lower() in ("1", "true", "yes")
    if not isinstance(confirm_roles, (bool, type(None))):
        return _error("confirm_roles must be a boolean", 400)
    mode = form.get("mode") or os.environ.get("MIROFISH_PROFILE", "")
    problem = _mode_problem(mode) if form.get("mode") else None
    if problem:
        return _error(*problem)

    params = {
        "simulation_requirement": requirement,
        "max_rounds": max_rounds,
        "seed": seed,
        "seeds": seeds,
        "confirm_roles": bool(confirm_roles),
        "project_name": form.get("project_name") or "",
        "profile": mode,
        "locale": request.headers.get("Accept-Language", ""),
    }

    def write(folder: str) -> str:
        if upload:
            path = os.path.join(folder, "document" + os.path.splitext(upload.filename)[1].lower())
            upload.save(path)
        else:
            path = os.path.join(folder, "document.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return path

    return _start(params, write)


def _mode_problem(mode: str) -> tuple[str, int] | None:
    """Why a run cannot use ``mode`` now, as (message, HTTP status)."""

    if mode not in run_mode.profiles():
        return f"mode must be one of {', '.join(run_mode.profiles())}", 400
    if mode == "local-llm":
        # LLM agents need 8K per slot (docs/local-first.md); the local path 4K.
        context = slot_context(Config.LLM_BASE_URL)
        if context is not None and context < LLM_SLOT_CONTEXT:
            return t("api.runModeContext", mode=mode, need=LLM_SLOT_CONTEXT, have=context), 409
    return None


def _start(params: dict, write_document) -> Response:
    """Create the run, save its document (``write_document(folder) -> path``)
    and start it; a failure marks the run failed instead of leaving it queued."""

    runs = get_runs(current_app._get_current_object())
    run_id = runs.store.create(params)
    try:
        params["document_path"] = write_document(_run_dir(run_id))  # a resume uploads it again
        runs.store.set_params(run_id, params)
        runs.orchestrator.start(run_id)
    except Exception as error:  # not left "queued" forever
        logger.exception("run %s could not start", run_id)
        runs.store.update(run_id, status=FAILED, error=f"{type(error).__name__}: {error}")
        return _error(f"run could not start: {error}", 500)
    logger.info("run %s started (mode %s)", run_id, params.get("profile") or "-")
    return _json({"success": True, "data": {"run_id": run_id}}, 202)


@runs_bp.route("/<run_id>/rerun", methods=["POST"])
def rerun(run_id: str):
    """The same document and settings in another mode (``{"mode": ...}``), to
    confirm a run's conclusions: the new run records ``confirms``."""

    record = get_runs(current_app._get_current_object()).store.get(run_id)
    if record is None:
        return _error(f"run not found: {run_id}", 404)
    mode = (request.get_json(silent=True) or {}).get("mode")
    if not isinstance(mode, str):
        return _error("mode is required", 400)
    problem = _mode_problem(mode)
    if problem:
        return _error(*problem)
    params = {**record["params"], "profile": mode, "confirms": run_id}
    source = params.pop("document_path", None)
    if not source or not os.path.exists(source):
        return _error("the run's document is gone", 409)

    def copy(folder: str) -> str:
        path = os.path.join(folder, os.path.basename(source))
        shutil.copyfile(source, path)
        return path

    return _start(params, copy)


@runs_bp.route("", methods=["GET"])
def list_runs():
    runs = get_runs(current_app._get_current_object())
    limit = request.args.get("limit", 50, type=int)
    return _json({"success": True, "data": runs.store.list(min(max(limit, 1), 500))})


@runs_bp.route("/<run_id>", methods=["GET"])
def get_run(run_id: str):
    record = get_runs(current_app._get_current_object()).store.get(run_id)
    if record is None:
        return _error(f"run not found: {run_id}", 404)
    record["params"].pop("document_path", None)
    return _json({"success": True, "data": record})


@runs_bp.route("/<run_id>/resume", methods=["POST"])
def resume_run(run_id: str):
    runs = get_runs(current_app._get_current_object())
    record = runs.store.get(run_id)
    if record is None:
        return _error(f"run not found: {run_id}", 404)
    try:
        runs.orchestrator.start(run_id, from_statuses=(FAILED, INTERRUPTED))
    except RunBusy as error:
        return _error(str(error), 409)
    return _json({"success": True, "data": {"run_id": run_id}}, 202)


@runs_bp.route("/<run_id>/confirm", methods=["POST"])
def confirm_roles(run_id: str):
    """Edit the roles first with GET/POST /api/graph/<graph_id>/roles (#64)."""

    runs = get_runs(current_app._get_current_object())
    if runs.store.get(run_id) is None:
        return _error(f"run not found: {run_id}", 404)
    try:
        runs.orchestrator.start(run_id, from_statuses=(AWAITING,), artifacts={"roles_confirmed": True})
    except RunBusy as error:
        return _error(str(error), 409)
    return _json({"success": True, "data": {"run_id": run_id}}, 202)


@runs_bp.route("/<run_id>/events", methods=["GET"])
def run_events(run_id: str):
    runs = get_runs(current_app._get_current_object())
    if runs.store.get(run_id) is None:
        return _error(f"run not found: {run_id}", 404)
    try:
        cursor = int(request.headers.get("Last-Event-ID") or request.args.get("after") or 0)
    except ValueError:
        cursor = 0
    poll_s = float(current_app.config.get("RUNS_SSE_POLL_S", 0.5))

    def stream():
        nonlocal cursor
        last_sent = time.monotonic()
        while True:
            for event in runs.store.events(run_id, cursor):
                cursor = event["id"]
                last_sent = time.monotonic()
                yield f"id: {event['id']}\nevent: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            record = runs.store.get(run_id)
            if record["status"] in TERMINAL and not runs.orchestrator.is_active(run_id):
                if not runs.store.events(run_id, cursor):
                    return
                continue
            if time.monotonic() - last_sent > HEARTBEAT_S:
                last_sent = time.monotonic()
                yield ": keep-alive\n\n"
            time.sleep(poll_s)

    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
