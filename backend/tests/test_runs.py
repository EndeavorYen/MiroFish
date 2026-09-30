"""Runs: store, orchestrator and API (#63). No model service needed."""

import json
import time

import pytest

from app.runs.orchestrator import STAGES, RunOrchestrator, StageFailed
from app.runs.service import Runs, set_runs
from app.runs.store import COMPLETED, FAILED, INTERRUPTED, RUNNING, RunStore


def _fake_stages(log, fail_at=None, crash_at=None):
    def make(name):
        def stage(ctx):
            log.append((name, dict(ctx.artifacts)))
            ctx.emit("progress", {"progress": 50, "message": f"{name} half way"})
            if name == fail_at:
                raise StageFailed(f"{name} could not finish: 模型服務無回應")
            if name == crash_at:
                raise KeyError("boom")
            return {f"{name}_id": f"{name}-1"}
        return stage
    return {name: make(name) for name in STAGES}


def _wait(store, run_id, timeout=5):
    deadline = time.monotonic() + timeout
    while store.get(run_id)["status"] in ("queued", RUNNING) and time.monotonic() < deadline:
        time.sleep(0.02)
    return store.get(run_id)


def test_the_store_keeps_runs_and_events_across_instances(tmp_path):
    path = tmp_path / "runs.sqlite"
    first = RunStore(str(path))
    run_id = first.create({"simulation_requirement": "OpenAI 漲價"})
    first.update(run_id, status=RUNNING, stage="graph", artifacts={"project_id": "p1"})
    first.add_event(run_id, "graph", "stage_start")
    first.add_event(run_id, "graph", "progress", {"progress": 40})
    first.close()

    second = RunStore(str(path))  # a new backend process
    record = second.get(run_id)
    assert (record["status"], record["stage"], record["artifacts"]) == (RUNNING, "graph", {"project_id": "p1"})
    assert record["params"]["simulation_requirement"] == "OpenAI 漲價"
    assert [e["kind"] for e in second.events(run_id)] == ["stage_start", "progress"]
    assert [e["kind"] for e in second.events(run_id, after_id=1)] == ["progress"]
    assert second.mark_interrupted() == [run_id]
    assert second.get(run_id)["status"] == INTERRUPTED


def test_stages_run_in_order_and_pass_their_artifacts(tmp_path):
    store, log = RunStore(str(tmp_path / "r.sqlite")), []
    orchestrator = RunOrchestrator(store, _fake_stages(log))
    run_id = store.create({})
    orchestrator.start(run_id).join(5)
    record = store.get(run_id)
    assert record["status"] == COMPLETED
    assert [name for name, _ in log] == list(STAGES)
    assert log[-1][1]["simulate_id"] == "simulate-1"  # later stages see earlier artifacts
    kinds = [e["kind"] for e in store.events(run_id)]
    assert kinds[0] == "run_start" and kinds[-1] == "run_done"
    assert kinds.count("stage_start") == kinds.count("stage_done") == len(STAGES)


@pytest.mark.parametrize("how", ["fail", "crash"])
def test_a_failing_stage_stops_the_run_with_its_reason(tmp_path, how):
    store, log = RunStore(str(tmp_path / "r.sqlite")), []
    stages = _fake_stages(log, fail_at="prepare") if how == "fail" else _fake_stages(log, crash_at="prepare")
    orchestrator = RunOrchestrator(store, stages)
    run_id = store.create({})
    orchestrator.start(run_id).join(5)
    record = store.get(run_id)
    assert record["status"] == FAILED and record["stage"] == "prepare"
    assert ("模型服務無回應" in record["error"]) if how == "fail" else ("KeyError" in record["error"])
    assert [name for name, _ in log] == ["ontology", "graph", "prepare"]  # simulate never ran
    assert store.events(run_id)[-1]["kind"] == "run_failed"


def test_a_resumed_run_skips_the_stages_it_finished(tmp_path):
    store, log = RunStore(str(tmp_path / "r.sqlite")), []
    orchestrator = RunOrchestrator(store, _fake_stages(log, fail_at="simulate"))
    run_id = store.create({})
    orchestrator.start(run_id).join(5)
    assert store.get(run_id)["status"] == FAILED

    log.clear()
    RunOrchestrator(store, _fake_stages(log)).start(run_id).join(5)  # e.g. after a restart
    record = store.get(run_id)
    assert record["status"] == COMPLETED and record["error"] is None
    assert [name for name, _ in log] == ["simulate", "report"]
    assert log[0][1]["prepare_id"] == "prepare-1"  # artifacts of the finished stages carried over


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import create_app
    from app.config import Config

    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    app = create_app()
    app.config["RUNS_SSE_POLL_S"] = 0.02
    log = []
    store = RunStore(str(tmp_path / "api-runs.sqlite"))
    set_runs(app, Runs(store, RunOrchestrator(store, _fake_stages(log))))
    client = app.test_client()
    client.log, client.store = log, store
    return client


def _events(client, run_id, headers=None):
    response = client.get(f"/api/runs/{run_id}/events", headers=headers or {})
    assert response.mimetype == "text/event-stream"
    text = b"".join(response.response).decode("utf-8")
    return [json.loads(block.split("data: ", 1)[1]) for block in text.split("\n\n") if "data: " in block]


def test_the_api_runs_a_document_and_streams_its_progress(client):
    response = client.post("/api/runs", data={"document_text": "東海市宣佈…", "simulation_requirement": "預測反應",
                                              "max_rounds": "6"})
    assert response.status_code == 202
    run_id = response.get_json()["data"]["run_id"]
    assert _wait(client.store, run_id)["status"] == COMPLETED
    record = client.get(f"/api/runs/{run_id}").get_json()["data"]
    assert record["params"]["max_rounds"] == 6 and "document_path" not in record["params"]
    events = _events(client, run_id)  # the stream ends once the run is done
    assert events[0]["kind"] == "run_start" and events[-1]["kind"] == "run_done"
    assert [e["stage"] for e in events if e["kind"] == "stage_start"] == list(STAGES)
    # Last-Event-ID continues the stream after that event.
    later = _events(client, run_id, headers={"Last-Event-ID": str(events[-2]["id"])})
    assert [e["kind"] for e in later] == ["run_done"]


def test_the_api_validates_and_resumes(client):
    assert client.post("/api/runs", data={"document_text": "x"}).status_code == 400
    assert client.post("/api/runs", data={"simulation_requirement": "y"}).status_code == 400
    assert client.post("/api/runs", data={"document_text": "x", "simulation_requirement": "y",
                                          "max_rounds": "zero"}).status_code == 400
    assert client.get("/api/runs/run_nope").status_code == 404

    run_id = client.post("/api/runs", data={"document_text": "x", "simulation_requirement": "y"}).get_json()["data"]["run_id"]
    _wait(client.store, run_id)
    assert client.post(f"/api/runs/{run_id}/resume").status_code == 409  # completed runs do not resume
    client.store.update(run_id, status=INTERRUPTED)
    assert client.post(f"/api/runs/{run_id}/resume").status_code == 202
    assert _wait(client.store, run_id)["status"] == COMPLETED


def test_runs_going_at_a_restart_are_marked_interrupted(tmp_path, monkeypatch):
    from app import create_app

    path = tmp_path / "restart.sqlite"
    monkeypatch.setenv("RUNS_DB_PATH", str(path))
    store = RunStore(str(path))
    run_id = store.create({})
    store.update(run_id, status=RUNNING, stage="simulate")
    store.close()
    create_app()
    reopened = RunStore(str(path))
    assert reopened.get(run_id)["status"] == INTERRUPTED
    assert reopened.events(run_id)[-1]["kind"] == "interrupted"


def test_the_simulation_stage_reports_rounds_as_progress():
    from app.runs.stages import _round_progress

    assert _round_progress({"current_round": 6, "total_rounds": 24}) == (25, "round 6/24")
    assert _round_progress({"current_round": 0, "total_rounds": 0}) == (None, None)
