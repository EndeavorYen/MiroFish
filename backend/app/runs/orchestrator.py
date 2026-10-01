"""Run the stages of a run in order, one thread per run (#63).

A stage is a function ``stage(ctx) -> dict`` returning the artifacts it
produced (ids of the project, graph, simulation, report); it raises
``StageFailed`` with a reason to stop the run. Finished stages are listed in
``artifacts["done_stages"]`` so a resumed run skips them.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from ..utils.logger import get_logger
from .store import AWAITING, COMPLETED, FAILED, RUNNING, RunStore

logger = get_logger("mirofish.runs")

# Fixed stage codes; display text belongs to the frontend's i18n.
STAGES = ("ontology", "graph", "prepare", "simulate", "report", "consistency")

# With ``params["confirm_roles"]`` the run pauses before this stage until
# ``artifacts["roles_confirmed"]`` (POST /api/runs/<id>/confirm).
CONFIRM_BEFORE = "prepare"


class StageFailed(Exception):
    """A stage could not finish; the message is shown as the run's reason."""


class RunBusy(Exception):
    """The run is going, or is not in a status it can start from."""


@dataclass
class StageContext:
    run_id: str
    params: dict[str, Any]
    artifacts: dict[str, Any]
    store: RunStore
    stage: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self.store.add_event(self.run_id, self.stage, kind, payload)


Stage = Callable[[StageContext], dict]


class RunOrchestrator:
    def __init__(self, store: RunStore, stages: dict[str, Stage], extra: dict[str, Any] | None = None) -> None:
        missing = [s for s in STAGES if s not in stages]
        if missing:
            raise ValueError(f"missing stages: {missing}")
        self.store = store
        self.stages = stages
        self.extra = extra or {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(run_id)
            return bool(thread and thread.is_alive())

    def start(self, run_id: str, from_statuses: tuple[str, ...] | None = None,
              artifacts: dict[str, Any] | None = None) -> threading.Thread:
        """Start the run's thread. With ``from_statuses`` the run must be in
        one of them; the check and the start share the lock, so two resumes
        at once start one thread and the other gets ``RunBusy``."""

        with self._lock:
            for done_id in [i for i, t in self._threads.items() if not t.is_alive()]:
                del self._threads[done_id]
            if run_id in self._threads:
                raise RunBusy("run is still going")
            if from_statuses is not None:
                status = (self.store.get(run_id) or {}).get("status")
                if status not in from_statuses:
                    raise RunBusy(f"the run must be {' or '.join(from_statuses)} (status: {status})")
            self.store.update(run_id, status=RUNNING, clear_error=True, artifacts=artifacts)
            thread = threading.Thread(target=self.run, args=(run_id,), name=f"run-{run_id}", daemon=True)
            self._threads[run_id] = thread
            thread.start()
        return thread

    def run(self, run_id: str) -> None:
        """All remaining stages, in order; stops at the first failure."""

        try:
            self._run(run_id)
        except Exception as error:  # e.g. the database: never leave the run "running"
            logger.error("run %s crashed: %s", run_id, traceback.format_exc())
            try:
                self._fail(run_id, None, f"{type(error).__name__}: {error}")
            except Exception:
                logger.error("run %s: could not record the failure: %s", run_id, traceback.format_exc())

    def _run(self, run_id: str) -> None:
        record = self.store.get(run_id)
        artifacts = dict(record["artifacts"])
        done = list(artifacts.get("done_stages", []))
        self.store.add_event(run_id, None, "run_start", {"resume_from": next((s for s in STAGES if s not in done), None)})
        for stage in STAGES:
            if stage in done:
                continue
            if stage == CONFIRM_BEFORE and record["params"].get("confirm_roles") and not artifacts.get("roles_confirmed"):
                self.store.update(run_id, status=AWAITING, stage=stage)
                self.store.add_event(run_id, stage, "awaiting_confirmation", {"graph_id": artifacts.get("graph_id")})
                return
            self.store.update(run_id, stage=stage)
            ctx = StageContext(run_id, record["params"], artifacts, self.store, stage, self.extra)
            ctx.emit("stage_start")
            started = time.perf_counter()
            try:
                produced = self.stages[stage](ctx) or {}
            except StageFailed as error:
                self._fail(run_id, stage, str(error))
                return
            except Exception as error:  # a bug in a stage must not leave the run "running"
                logger.error("run %s stage %s crashed: %s", run_id, stage, traceback.format_exc())
                self._fail(run_id, stage, f"{type(error).__name__}: {error}")
                return
            artifacts.update(produced)
            done.append(stage)
            artifacts["done_stages"] = done
            self.store.update(run_id, artifacts={**produced, "done_stages": done})
            ctx.emit("stage_done", {"artifacts": produced, "seconds": round(time.perf_counter() - started, 1)})
        self.store.update(run_id, status=COMPLETED)
        self.store.add_event(run_id, None, "run_done", {"artifacts": artifacts})

    def _fail(self, run_id: str, stage: str | None, reason: str) -> None:
        self.store.update(run_id, status=FAILED, error=reason)
        self.store.add_event(run_id, stage, "run_failed", {"reason": reason})
