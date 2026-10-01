"""A run's own mode (#66): local, local-hybrid or local-llm per run."""

import threading

import pytest

from app import run_mode
from app.config import Config
from app.runs.orchestrator import STAGES, RunOrchestrator
from app.runs.service import Runs, set_runs
from app.runs.store import COMPLETED, RunStore


def test_a_run_mode_overrides_the_process_settings_only_inside_the_run(monkeypatch):
    monkeypatch.setattr(Config, "PROFILE_MODE", "llm")
    monkeypatch.setattr(Config, "REPORT_MODE", "agent")
    with run_mode.use("local"):
        assert (Config.ONTOLOGY_MODE, Config.PROFILE_MODE, Config.SIM_CONFIG_MODE) == ("template", "structured", "structured")
        assert Config.REPORT_MODE == "metrics"
        assert run_mode.setting("SIM_DECISION_BACKEND", "llm") == "system_one"
    assert (Config.PROFILE_MODE, Config.REPORT_MODE) == ("llm", "agent")  # back to the process's


def test_no_or_an_unknown_profile_leaves_the_process_settings(monkeypatch):
    monkeypatch.setattr(Config, "PROFILE_MODE", "llm")
    for profile in (None, "", "cloud"):
        with run_mode.use(profile):
            assert Config.PROFILE_MODE == "llm" and not run_mode.active()


def test_the_mode_follows_a_bound_thread_and_reaches_the_subprocess():
    seen = []
    with run_mode.use("local-llm"):
        thread = threading.Thread(target=run_mode.bind(lambda: seen.append(Config.PROFILE_MODE)))
        env = run_mode.subprocess_env({"CONTENT_MODE": "tiered", "PATH": "x"})
    thread.start()
    thread.join()
    assert seen == ["llm"]
    assert env["SIM_DECISION_BACKEND"] == "llm" and env["SIM_AGENT_CONTEXT_TOKENS"] == "3072"
    assert env["MIROFISH_PROFILE"] == "local-llm"  # the simulation process applies it again
    assert "CONTENT_MODE" not in env and env["PATH"] == "x"  # local-llm leaves it unset


def test_two_runs_in_different_modes_each_see_their_own(tmp_path):
    store, seen, gate = RunStore(str(tmp_path / "r.sqlite")), {}, threading.Barrier(2)

    def stage(ctx):
        if ctx.stage == "ontology":
            gate.wait(5)  # both runs are in their stage at the same time
        seen.setdefault(ctx.params["profile"], set()).add(Config.PROFILE_MODE)
        return {}

    orchestrator = RunOrchestrator(store, {name: stage for name in STAGES})
    ids = [store.create({"profile": "local"}), store.create({"profile": "local-hybrid"})]
    threads = [orchestrator.start(run_id) for run_id in ids]
    for thread in threads:
        thread.join(5)
    assert all(store.get(run_id)["status"] == COMPLETED for run_id in ids)
    assert seen == {"local": {"structured"}, "local-hybrid": {"llm"}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import create_app

    monkeypatch.setenv("MIROFISH_PROFILE", "local")  # a run may choose its mode only here

    app = create_app()
    store = RunStore(str(tmp_path / "api.sqlite"))
    started = []

    class Recorder(RunOrchestrator):
        def start(self, run_id, from_statuses=None, artifacts=None):
            started.append(run_id)
            return threading.Thread(target=lambda: None)

    set_runs(app, Runs(store, Recorder(store, {name: (lambda ctx: {}) for name in STAGES})))
    client = app.test_client()
    client.store, client.started = store, started
    return client


def _create(client, **extra):
    return client.post("/api/runs", data={"document_text": "東海市宣佈…", "simulation_requirement": "預測反應", **extra})


def test_the_api_takes_a_mode(client, monkeypatch):
    monkeypatch.setenv("MIROFISH_PROFILE", "local")
    assert _create(client, mode="cloud").status_code == 400
    default = _create(client).get_json()["data"]["run_id"]
    hybrid = _create(client, mode="local-hybrid").get_json()["data"]["run_id"]
    assert client.store.get(default)["params"]["profile"] == "local"
    assert client.store.get(hybrid)["params"]["profile"] == "local-hybrid"


def test_local_llm_needs_an_8k_slot(client, monkeypatch):
    monkeypatch.setattr("app.api.runs.slot_context", lambda url: 4096)
    response = _create(client, mode="local-llm")
    assert response.status_code == 409 and "65536" in response.get_json()["error"]
    monkeypatch.setattr("app.api.runs.slot_context", lambda url: 8192)
    assert _create(client, mode="local-llm").status_code == 202
    monkeypatch.setattr("app.api.runs.slot_context", lambda url: None)  # not llama.cpp: not checked
    assert _create(client, mode="local-llm").status_code == 202


def test_rerun_runs_the_same_input_in_another_mode(client, monkeypatch):
    monkeypatch.setattr("app.api.runs.slot_context", lambda url: None)
    original = _create(client, mode="local", seeds="2", max_rounds="12").get_json()["data"]["run_id"]
    assert client.post(f"/api/runs/{original}/rerun", json={}).status_code == 400
    assert client.post(f"/api/runs/{original}/rerun", json={"mode": "cloud"}).status_code == 400
    assert client.post("/api/runs/run_nope/rerun", json={"mode": "local-llm"}).status_code == 404

    response = client.post(f"/api/runs/{original}/rerun", json={"mode": "local-llm"})
    assert response.status_code == 202
    again = response.get_json()["data"]["run_id"]
    first, second = client.store.get(original)["params"], client.store.get(again)["params"]
    assert (second["profile"], second["confirms"]) == ("local-llm", original)
    assert (second["seeds"], second["max_rounds"], second["simulation_requirement"]) == (2, 12, "預測反應")
    assert second["document_path"] != first["document_path"]  # its own copy
    with open(second["document_path"], encoding="utf-8") as f:
        assert f.read() == "東海市宣佈…"
    assert client.started[-1] == again


def test_the_backend_profile_keeps_its_env_overrides(monkeypatch):
    """A run in the backend's own mode uses the process settings, so .env
    overrides of that profile (e.g. SIM_AGENT_CONTEXT_TOKENS=off) still apply."""

    monkeypatch.setenv("MIROFISH_PROFILE", "Local-LLM ")
    monkeypatch.setenv("SIM_AGENT_CONTEXT_TOKENS", "off")
    with run_mode.use("local-llm"):
        assert not run_mode.active()
        assert run_mode.subprocess_env({"SIM_AGENT_CONTEXT_TOKENS": "off"}) == {"SIM_AGENT_CONTEXT_TOKENS": "off"}
    with run_mode.use("local"):  # another mode: its own settings
        assert run_mode.active()


def test_a_bound_thread_carries_the_mode_and_nothing_else():
    import contextvars

    other = contextvars.ContextVar("other", default="unset")
    seen = []
    with run_mode.use("local-llm"):
        other.set("request of the caller")
        target = run_mode.bind(lambda: seen.append((Config.PROFILE_MODE, other.get())))
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    assert seen == [("llm", "unset")]


def test_a_backend_without_a_local_profile_refuses_a_mode(client, monkeypatch):
    monkeypatch.delenv("MIROFISH_PROFILE")
    response = _create(client, mode="local")
    assert response.status_code == 409 and "MIROFISH_PROFILE" in response.get_json()["error"]
    assert _create(client).status_code == 202  # no mode: the backend's own settings


def test_resume_checks_the_mode_again(client, monkeypatch):
    from app.runs.store import FAILED

    monkeypatch.setattr("app.api.runs.slot_context", lambda url: 8192)
    run_id = _create(client, mode="local-llm").get_json()["data"]["run_id"]
    client.store.update(run_id, status=FAILED)
    monkeypatch.setattr("app.api.runs.slot_context", lambda url: 4096)  # llama-server back with 4K slots
    assert client.post(f"/api/runs/{run_id}/resume").status_code == 409


def test_rerun_needs_a_json_object(client):
    run_id = _create(client).get_json()["data"]["run_id"]
    assert client.post(f"/api/runs/{run_id}/rerun", json=["local-llm"]).status_code == 400


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000"])
def test_slot_context_reads_llama_props(monkeypatch, base_url):
    from app.utils import model_health

    asked = []

    class Answer:
        def json(self):
            return {"default_generation_settings": {"n_ctx": 8192}, "total_slots": 8}

    monkeypatch.setattr(model_health.httpx, "get", lambda url, **kw: asked.append(url) or Answer())
    assert model_health.slot_context(base_url) == 8192
    assert asked == ["http://127.0.0.1:8000/props"]
    assert model_health.slot_context("https://api.example.com/v1") is None  # not local: not asked
