"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _model_service_answers(monkeypatch):
    """API tests must not depend on a live model server: the start/prepare
    health check (#62) answers "up" unless a test replaces it."""

    monkeypatch.setattr("app.api.simulation.check_model_service", lambda url: None)
