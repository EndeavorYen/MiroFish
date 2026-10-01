"""A run that compares options (#56, #66)."""

import json
import os
import threading
import time

import pytest

from app.runs.orchestrator import STAGES, RunOrchestrator, StageContext
from app.runs.service import Runs, set_runs
from app.runs.store import RunStore
from app.services import option_compare

OPTIONS = [{"name": "維持現狀", "text": "水費不調整。"}, {"name": "分階段", "text": "三年分三次調漲。"}]


def test_options_are_validated():
    assert option_compare.validate([{"name": " A ", "text": " x "}, {"name": "B", "text": "y"}]) == [
        {"name": "A", "text": "x"}, {"name": "B", "text": "y"}]
    for bad in ([OPTIONS[0]], OPTIONS * 3, [OPTIONS[0], OPTIONS[0]], [OPTIONS[0], {"name": "B"}],
                [OPTIONS[0], {"name": "x" * 41, "text": "t"}], "not a list", [OPTIONS[0], "B"]):
        with pytest.raises(ValueError):
            option_compare.validate(bad)


def test_an_option_is_announced_in_the_requirement_and_the_document():
    # First in both: the local prep reads the requirement's first 300
    # characters and opens with the document's first sentences.
    long_requirement = "模擬北港市調漲水費後的輿論。" + "各方的反應" * 80
    requirement, document = option_compare.with_option(long_requirement, "北港市擬調漲水費。\n", OPTIONS[1])
    assert requirement.startswith("公布的方案：三年分三次調漲。\n模擬北港市")
    assert "三年分三次調漲" in requirement[:300]
    assert document == "公布的方案：三年分三次調漲。\n\n北港市擬調漲水費。\n"
    english = "Simulate how residents react after the city announces higher water prices for every household."
    requirement, _ = option_compare.with_option(english, "Text", {"name": "a", "text": "A free first month"})
    assert requirement.startswith("Announced plan: A free first month.\n")
    with pytest.raises(ValueError):
        option_compare.validate_one({"name": "a", "text": "x" * 301})


def _scan(tendency, oppose):
    return {"tendency": {"value": tendency}, "post_shares": {"oppose": oppose},
            "most_opposed_posts": [{"stance": oppose, "text": f"t{tendency}"}]}


def test_options_are_compared_to_the_baseline_seed_by_seed():
    results = {
        "維持現狀": {1: _scan(0.6, 0.2), 1001: _scan(0.5, 0.3), 2001: _scan(0.55, 0.25)},
        "分階段": {1: _scan(0.7, 0.1), 1001: _scan(0.65, 0.15), 2001: _scan(0.6, 0.2)},
    }
    rows = option_compare.compare(results, ["維持現狀", "分階段"])
    assert [r["option"] for r in rows] == ["維持現狀", "分階段"]
    vs = rows[1]["vs_baseline"]["tendency"]
    assert (round(vs["mean"], 4), vs["same_sign"], vs["seeds"], vs["distinct"]) == (0.1, 3, 3, True)
    # Two seeds agreeing is a coin flip under no effect: never "distinct".
    two = {name: {s: r for s, r in runs.items() if s < 2001} for name, runs in results.items()}
    assert option_compare.compare(two, ["維持現狀", "分階段"])[1]["vs_baseline"]["tendency"]["distinct"] is False
    assert option_compare.compare({"分階段": results["分階段"]}, ["維持現狀", "分階段"]) == []  # no baseline


def test_the_prepare_endpoint_announces_the_option(monkeypatch):
    from app import create_app
    from app.api.simulation import prepare as prepare_api
    from app.services.simulation_manager import SimulationManager

    seen = {}

    class Project:
        simulation_requirement = "預測反應。"

    simulation_id = SimulationManager().create_simulation("proj_1", "g1").simulation_id  # data in tmp (conftest)
    monkeypatch.setattr(SimulationManager, "prepare_simulation", lambda self, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(prepare_api.ProjectManager, "get_project", lambda pid: Project())
    monkeypatch.setattr(prepare_api.ProjectManager, "get_extracted_text", lambda pid: "北港市擬調漲水費。")
    monkeypatch.setattr(prepare_api, "EntityReader", lambda: (_ for _ in ()).throw(RuntimeError("skip the preview")))
    client = create_app().test_client()

    bad = client.post("/api/simulation/prepare", json={"simulation_id": simulation_id, "option": {"name": "x"}})
    assert bad.status_code == 400
    response = client.post("/api/simulation/prepare", json={"simulation_id": simulation_id, "option": OPTIONS[1]})
    assert response.status_code == 200, response.get_json()
    for _ in range(100):  # the preparation runs in a background thread
        if seen:
            break
        time.sleep(0.05)
    assert seen["simulation_requirement"].startswith("公布的方案：三年分三次調漲。")
    assert seen["document_text"].startswith("公布的方案：三年分三次調漲。")


class _Fake:
    """The simulation endpoints the stages call."""

    def __init__(self):
        self.created, self.prepared, self.copied, self.started = 0, [], [], []

    def __call__(self, ctx, method, path, **kwargs):
        body = kwargs.get("json") or {}
        ok = lambda data: (200, {"success": True, "data": data})  # noqa: E731
        if path == "/api/simulation/create":
            self.created += 1
            return ok({"simulation_id": f"sim_p{self.created}"})
        if path == "/api/simulation/prepare":
            self.prepared.append((body["simulation_id"], (body.get("option") or {}).get("name")))
            return ok({"already_prepared": True})
        if path == "/api/simulation/copy":
            self.copied.append(body["simulation_id"])
            return ok({"simulation_id": f"{body['simulation_id']}_c{len(self.copied)}"})
        if path == "/api/simulation/start":
            self.started.append(body["simulation_id"])
            return ok({})
        if path == "/api/simulation/close-env":
            return ok({})
        if path.endswith("/run-status"):
            simulation_id = path.split("/")[3]
            done = simulation_id in self.started
            return ok({"runner_status": "completed" if done else "idle", "current_round": 6, "total_rounds": 6})
        raise AssertionError(path)


def _ctx(tmp_path, params, artifacts=None):
    store = RunStore(str(tmp_path / "o.sqlite"))
    run_id = store.create(params)
    return StageContext(run_id, params, {"project_id": "proj_1", "graph_id": "g1", **(artifacts or {})},
                        store, "prepare", {"poll_s": 0})


def test_each_option_is_prepared_and_runs_on_the_same_seeds(tmp_path, monkeypatch):
    from app.runs import stages

    fake = _Fake()
    monkeypatch.setattr(stages, "_request", fake)
    monkeypatch.setattr(stages, "parallel_limit", lambda count: 4)
    ctx = _ctx(tmp_path, {"options": OPTIONS, "seeds": 2})
    prepared = stages.prepare(ctx)
    assert prepared["option_simulations"] == [
        {"option": "維持現狀", "simulation_id": "sim_p1"}, {"option": "分階段", "simulation_id": "sim_p2"}]
    assert fake.prepared == [("sim_p1", "維持現狀"), ("sim_p2", "分階段")]
    assert prepared["simulation_id"] == "sim_p1"  # the baseline's: the report reads it

    ctx.artifacts.update(prepared)
    runs = stages.simulate(ctx)["seed_simulations"]
    assert [(r["option"], r["seed"]) for r in runs] == [("維持現狀", 1), ("維持現狀", 1001), ("分階段", 1), ("分階段", 1001)]
    assert fake.copied == ["sim_p1", "sim_p2"]  # each option's second seed copies that option
    assert runs[0]["simulation_id"] == "sim_p1" and runs[2]["simulation_id"] == "sim_p2"


def test_a_resumed_option_preparation_reuses_what_it_made(tmp_path, monkeypatch):
    from app.runs import stages

    fake = _Fake()
    monkeypatch.setattr(stages, "_request", fake)
    ctx = _ctx(tmp_path, {"options": OPTIONS}, {"option_simulations": [{"option": "維持現狀", "simulation_id": "sim_old"}]})
    prepared = stages.prepare(ctx)
    assert fake.created == 1  # only the second option's simulation is new
    assert [p["simulation_id"] for p in prepared["option_simulations"]] == ["sim_old", "sim_p1"]


def test_the_comparison_scores_every_option_on_the_base_question(tmp_path, monkeypatch):
    from app.runs import stages
    from app.services import metrics_report

    questions = set()
    values = {"sim_a1": (0.6, 0.2), "sim_a2": (0.5, 0.3), "sim_b1": (0.7, 0.1), "sim_b2": (0.66, 0.12)}

    def scan(sim_dir, score_fn, question=None):
        questions.add(question)
        tendency, oppose = values[os.path.basename(sim_dir)]
        return {**_scan(tendency, oppose), "main_camp": {"value": "support"}, "trend": {"value": 0.0},
                "ranking": {"indistinct": False}, "by_role": {"甲": tendency}}

    monkeypatch.setattr(metrics_report, "default_score_fn", lambda: None)
    monkeypatch.setattr(metrics_report, "scan_conclusions", scan)
    runs = [{"option": "維持現狀", "seed": 1, "simulation_id": "sim_a1"}, {"option": "維持現狀", "seed": 1001, "simulation_id": "sim_a2"},
            {"option": "分階段", "seed": 1, "simulation_id": "sim_b1"}, {"option": "分階段", "seed": 1001, "simulation_id": "sim_b2"}]
    ctx = _ctx(tmp_path, {"options": OPTIONS, "seeds": 2, "simulation_requirement": "模擬北港市調漲水費後的反應。"},
               {"seed_simulations": runs})
    out = stages.consistency(ctx)
    assert len(questions) == 1 and None not in questions  # one question for every option
    assert out["consistency"]["seeds"] == 2  # the baseline's seeds
    rows = out["options_comparison"]
    assert [r["option"] for r in rows] == ["維持現狀", "分階段"]
    assert rows[1]["vs_baseline"]["tendency"]["same_sign"] == 2  # two seeds: shown, not "distinct"


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import create_app

    app = create_app()
    store = RunStore(str(tmp_path / "api.sqlite"))

    class Recorder(RunOrchestrator):
        def start(self, run_id, from_statuses=None, artifacts=None):
            return threading.Thread(target=lambda: None)

    set_runs(app, Runs(store, Recorder(store, {name: (lambda ctx: {}) for name in STAGES})))
    client = app.test_client()
    client.store = store
    return client


def test_the_api_takes_options(client):
    data = {"document_text": "文件", "simulation_requirement": "需求"}
    assert client.post("/api/runs", data={**data, "options": "not json"}).status_code == 400
    assert client.post("/api/runs", data={**data, "options": json.dumps(OPTIONS[:1])}).status_code == 400
    form = client.post("/api/runs", data={**data, "options": json.dumps(OPTIONS)})
    assert form.status_code == 202
    assert client.store.get(form.get_json()["data"]["run_id"])["params"]["options"] == OPTIONS
    as_json = client.post("/api/runs", json={**data, "options": OPTIONS})
    assert client.store.get(as_json.get_json()["data"]["run_id"])["params"]["options"] == OPTIONS
    plain = client.post("/api/runs", data=data)
    assert client.store.get(plain.get_json()["data"]["run_id"])["params"]["options"] is None


def test_a_prepared_option_is_recorded_and_scored_on_the_base_question(tmp_path):
    from app.services.metrics_report import scan_conclusions

    config_path = tmp_path / "simulation_config.json"
    requirement, _ = option_compare.with_option("模擬北港市調漲水費後的反應。", "文件", OPTIONS[1])
    config_path.write_text(json.dumps({"simulation_requirement": requirement, "agent_configs": []}, ensure_ascii=False),
                           encoding="utf-8")
    option_compare.record(str(config_path), "模擬北港市調漲水費後的反應。", OPTIONS[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["option"] == OPTIONS[1] and config["base_requirement"] == "模擬北港市調漲水費後的反應。"

    (tmp_path / "twitter").mkdir()
    rows = [{"round": 1, "agent_id": 1, "agent_name": "甲", "action_type": "CREATE_POST", "action_args": {"content": "太貴了"}}]
    (tmp_path / "twitter" / "actions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    asked = set()
    scan_conclusions(str(tmp_path), score_fn=lambda text, question: asked.add(question) or 0.5)
    assert asked and all("北港市調漲水費" in q and "公布的方案" not in q for q in asked)


def test_a_bad_option_is_refused_even_on_a_prepared_simulation(monkeypatch):
    from app import create_app
    from app.api.simulation import prepare as prepare_api

    monkeypatch.setattr(prepare_api, "_check_simulation_prepared", lambda sid: (True, {}))
    from app.services.simulation_manager import SimulationManager

    simulation_id = SimulationManager().create_simulation("proj_1", "g1").simulation_id
    response = create_app().test_client().post(
        "/api/simulation/prepare", json={"simulation_id": simulation_id, "option": {"name": "only a name"}})
    assert response.status_code == 400


def test_one_run_that_cannot_be_scored_does_not_lose_the_comparison(tmp_path, monkeypatch):
    from app.runs import stages
    from app.services import metrics_report

    def scan(sim_dir, score_fn, question=None):
        name = os.path.basename(sim_dir)
        if name == "sim_b2":
            raise ValueError("broken posts file")
        tendency = {"sim_a1": 0.6, "sim_a2": 0.5, "sim_b1": 0.7}[name]
        return {**_scan(tendency, 0.2), "main_camp": {"value": "support"}, "trend": {"value": 0.0},
                "ranking": {"indistinct": False}, "by_role": {"甲": tendency}}

    monkeypatch.setattr(metrics_report, "default_score_fn", lambda: None)
    monkeypatch.setattr(metrics_report, "scan_conclusions", scan)
    runs = [{"option": "維持現狀", "seed": 1, "simulation_id": "sim_a1"}, {"option": "維持現狀", "seed": 1001, "simulation_id": "sim_a2"},
            {"option": "分階段", "seed": 1, "simulation_id": "sim_b1"}, {"option": "分階段", "seed": 1001, "simulation_id": "sim_b2"}]
    ctx = _ctx(tmp_path, {"options": OPTIONS, "seeds": 2, "simulation_requirement": "需求"}, {"seed_simulations": runs})
    out = stages.consistency(ctx)
    assert [r["option"] for r in out["options_comparison"]] == ["維持現狀", "分階段"]
    assert out["options_comparison"][1]["seeds"] == [1]
    assert "broken posts file" in out["scoring_errors"][0]


def test_a_resumed_preparation_keeps_every_known_option(tmp_path, monkeypatch):
    from app.runs import stages

    fake = _Fake()
    monkeypatch.setattr(stages, "_request", fake)
    known = [{"option": "維持現狀", "simulation_id": "sim_old1"}, {"option": "分階段", "simulation_id": "sim_old2"}]
    ctx = _ctx(tmp_path, {"options": OPTIONS}, {"option_simulations": known})
    stored = []
    original = ctx.store.update

    def watch(run_id, **kwargs):
        if "artifacts" in kwargs:
            stored.append(kwargs["artifacts"]["option_simulations"])
        return original(run_id, **kwargs)

    monkeypatch.setattr(ctx.store, "update", watch)
    stages.prepare(ctx)
    assert fake.created == 0
    assert all(len(entry) == 2 for entry in stored)  # never shrinks to the options seen so far


def test_the_api_limits_and_normalises_options(client):
    data = {"document_text": "文件", "simulation_requirement": "需求"}
    for empty in ("null", "[]", ""):
        response = client.post("/api/runs", data={**data, "options": empty})
        assert client.store.get(response.get_json()["data"]["run_id"])["params"]["options"] is None
    too_many = [{"name": f"o{i}", "text": "t"} for i in range(4)]
    assert client.post("/api/runs", data={**data, "options": json.dumps(too_many), "seeds": "5"}).status_code == 400
    assert client.post("/api/runs", data={**data, "options": json.dumps(too_many), "seeds": "4"}).status_code == 202
