"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _model_service_answers(monkeypatch):
    """API tests must not depend on a live model server: the start/prepare
    health check (#62) answers "up" unless a test replaces it."""

    monkeypatch.setattr("app.api.simulation._shared.check_model_service", lambda url: None)


@pytest.fixture(autouse=True)
def _runs_database_in_tmp(monkeypatch, tmp_path_factory):
    """The runs store (#63) must never touch the real uploads/runs database."""

    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path_factory.mktemp("runs") / "runs.sqlite"))
