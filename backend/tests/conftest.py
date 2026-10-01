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


@pytest.fixture(autouse=True)
def _uploads_in_tmp(monkeypatch, tmp_path_factory):
    """No test writes into the real backend/uploads: loading or saving a
    simulation, project or report creates its folder, so every data folder
    points into a temporary one. A test may still point one elsewhere."""

    from app.config import Config
    from app.models.project import ProjectManager
    from app.services.report_manager import ReportManager
    from app.services.simulation_manager import SimulationManager
    from app.services.simulation_runner import SimulationRunner

    uploads = tmp_path_factory.mktemp("uploads")
    simulations = str(uploads / "simulations")
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(uploads))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", simulations)
    monkeypatch.setattr(Config, "GRAPH_DATA_DIR", str(uploads / "graphs"))
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(uploads / "projects"))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(uploads / "reports"))
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", simulations)
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", simulations)
