"""The runs store and orchestrator of an app (#63)."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask

from .orchestrator import RunOrchestrator
from .stages import REAL_STAGES
from .store import RunStore, default_db_path

EXTENSION = "mirofish_runs"


class Runs:
    def __init__(self, store: RunStore, orchestrator: RunOrchestrator) -> None:
        self.store = store
        self.orchestrator = orchestrator


def get_runs(app: Flask) -> Runs:
    runs = app.extensions.get(EXTENSION)
    if runs is None:
        store = RunStore()
        runs = Runs(store, RunOrchestrator(store, REAL_STAGES, extra={"app": app}))
        app.extensions[EXTENSION] = runs
    return runs


def set_runs(app: Flask, runs: Runs) -> None:
    app.extensions[EXTENSION] = runs


def mark_interrupted_at_startup(logger: Any) -> None:
    """Runs that were going when the backend stopped lost their thread."""

    path = default_db_path()
    if not os.path.exists(path):
        return  # never used: do not create the database at start-up
    store = RunStore(path)
    try:
        ids = store.mark_interrupted()
    finally:
        store.close()
    if ids:
        logger.warning("runs interrupted by the restart (POST /api/runs/<id>/resume): %s", ", ".join(ids))
