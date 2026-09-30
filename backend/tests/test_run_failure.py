"""A run whose model service fails is reported as failed, with the reason (#62)."""

import json

import httpx
import pytest

from app.services.simulation_runner import SimulationRunner


def test_failure_file_is_the_error_of_a_failed_run(tmp_path):
    (tmp_path / "simulation.log").write_text("line\n" * 5 + "Traceback ... ModelServiceUnavailable", encoding="utf-8")
    (tmp_path / "failure.json").write_text(
        json.dumps({"reason": "twitter: every decision failed in 2 consecutive rounds; last error: ConnectError"}),
        encoding="utf-8",
    )
    reason = SimulationRunner._failure_reason(str(tmp_path), 3)
    assert reason.startswith("twitter: every decision failed")
    (tmp_path / "failure.json").unlink()
    assert "ModelServiceUnavailable" in SimulationRunner._failure_reason(str(tmp_path), 1)
    assert "1" in SimulationRunner._failure_reason(str(tmp_path), 1)


def test_force_restart_cleanup_removes_decisions_and_agent_state(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    sim = tmp_path / "sim-1"
    (sim / "twitter").mkdir(parents=True)
    for name in ("decisions.jsonl", "agent_state.db", "content_metrics_twitter.jsonl",
                 "content_metrics_reddit.jsonl", "failure.json", "simulation_config.json"):
        (sim / name).write_text("x", encoding="utf-8")
    result = SimulationRunner.cleanup_simulation_logs("sim-1")
    assert result["success"], result
    left = sorted(p.name for p in sim.iterdir() if p.is_file())
    assert left == ["simulation_config.json"]  # the config stays


def test_model_service_check_names_the_endpoint(monkeypatch):
    from app.utils import model_health

    def refused(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(model_health.httpx, "get", refused)
    reason = model_health.check_model_service("http://127.0.0.1:8000/v1")
    assert reason and "127.0.0.1:8000" in reason

    class Ok:
        status_code = 200

    monkeypatch.setattr(model_health.httpx, "get", lambda url, **kwargs: Ok())
    assert model_health.check_model_service("http://127.0.0.1:8000/v1") is None

    class Down:
        status_code = 503

    monkeypatch.setattr(model_health.httpx, "get", lambda url, **kwargs: Down())
    assert "503" in model_health.check_model_service("http://127.0.0.1:8000/v1")
    class Keyed:
        status_code = 401

    monkeypatch.setattr(model_health.httpx, "get", lambda url, **kwargs: Keyed())
    assert model_health.check_model_service("http://127.0.0.1:8000/v1") is None  # reachable, wants a key
    assert model_health.check_model_service("") is None  # no endpoint configured: nothing to check


def test_start_refuses_when_the_model_service_is_down(monkeypatch):
    from flask import Flask

    from app.api import simulation as simulation_api

    class State:
        status = None

    monkeypatch.setattr(simulation_api.SimulationManager, "get_simulation", lambda self, sid: State())
    monkeypatch.setattr(simulation_api, "check_model_service", lambda url: "模型服務無回應：http://127.0.0.1:8000/v1（ConnectError）")
    cleaned = []
    monkeypatch.setattr(simulation_api.SimulationRunner, "cleanup_simulation_logs",
                        classmethod(lambda cls, sid: cleaned.append(sid)))
    app = Flask(__name__)
    with app.test_request_context("/api/simulation/start", method="POST", json={"simulation_id": "sim-1", "force": True}):
        response, code = simulation_api.start_simulation()
    assert code == 503
    assert "127.0.0.1:8000" in response.get_json()["error"]
    assert cleaned == []  # the previous run's logs are kept


def test_record_failure_writes_the_reason_the_runner_reads(tmp_path):
    from app.simulation_policy.oasis_bridge import ModelServiceUnavailable, record_failure

    record_failure(str(tmp_path), ModelServiceUnavailable("reddit: every decision failed in 2 consecutive rounds"))
    assert SimulationRunner._failure_reason(str(tmp_path), 3).startswith("reddit: every decision failed")


def test_the_check_probes_the_services_the_step_uses(monkeypatch):
    from app.api import simulation as simulation_api

    probed = []
    monkeypatch.setattr(simulation_api, "check_model_service", lambda url: probed.append(url))
    monkeypatch.setattr(simulation_api.Config, "LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(simulation_api.Config, "SYSTEM_ONE_BASE_URL", "http://localhost:8000/v1")

    monkeypatch.setenv("SIM_DECISION_BACKEND", "llm")
    simulation_api._model_service_problem(run=True)
    simulation_api._model_service_problem()
    assert probed == ["https://api.example.com/v1"] * 2  # a hosted-LLM setup never probes :8000

    probed.clear()
    monkeypatch.setenv("SIM_DECISION_BACKEND", "system_one")
    simulation_api._model_service_problem(run=True)
    assert probed == ["http://localhost:8000/v1", "https://api.example.com/v1"]
