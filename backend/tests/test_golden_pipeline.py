"""Local-only guards of scripts/golden_pipeline.py (no services needed)."""

from pathlib import Path

import pytest

import scripts.golden_pipeline as gp


def test_local_env_forces_local_backends_and_drops_cloud_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(gp, "_INITIAL_ENV", {})
    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "http")
    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "secret")
    monkeypatch.setenv("ZEP_API_KEY", "z")
    monkeypatch.setenv("LLM_BOOST_API_KEY", "boost")
    monkeypatch.setenv("LLM_BOOST_BASE_URL", "https://boost.example/v1")
    monkeypatch.setenv("LLM_BASE_URL", "https://cloud.example/v1")  # e.g. from a .env
    env = gp.local_env(tmp_path)
    assert env["SYSTEM_ONE_BACKEND"] == "local"
    assert env["GRAPH_BACKEND"] == "local" and env["GRAPH_EXTRACTOR"] == "local"
    assert env["LLM_BASE_URL"] == gp.LOCAL_DEFAULTS["LLM_BASE_URL"]
    for key in gp.DROPPED_KEYS:
        assert key not in env
    assert env["GRAPH_DATA_DIR"] == str(tmp_path / "graphs")


def test_explicitly_exported_local_port_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(gp, "_INITIAL_ENV", {"LLM_BASE_URL": "http://127.0.0.1:9000/v1"})
    assert gp.local_env(tmp_path)["LLM_BASE_URL"] == "http://127.0.0.1:9000/v1"


def test_refuses_to_run_with_any_dotenv(monkeypatch, tmp_path):
    fake = tmp_path / ".env"
    fake.write_text("X=1", encoding="utf-8")
    monkeypatch.setattr(gp, "dotenv_files", lambda: [fake])
    with pytest.raises(SystemExit) as excinfo:
        gp.main(["prepare", "--work", str(tmp_path)])
    assert excinfo.value.code == 2


def test_dotenv_search_covers_backend_and_parents():
    searched = [gp.BACKEND_DIR / "app", *(gp.BACKEND_DIR / "app").parents]
    assert gp.BACKEND_DIR in searched and gp.BACKEND_DIR.parent in searched
    assert isinstance(gp.dotenv_files(), list)


def test_llm_error_count(tmp_path):
    log = tmp_path / "simulation.log"
    log.write_text("ok\nError code: 400 - context\nopenai.InternalServerError: Error code: 500\n", encoding="utf-8")
    assert gp.llm_error_count(log) == 2
    assert gp.llm_error_count(tmp_path / "missing.log") == 0


def test_local_defaults_budget_agent_memory():
    assert int(gp.LOCAL_DEFAULTS["SIM_AGENT_CONTEXT_TOKENS"]) < 8192



def _scenario(tmp_path, name, seed="種子", req="需求"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "news_seed.txt").write_text(seed, encoding="utf-8")
    (d / "simulation_requirement.txt").write_text(req, encoding="utf-8")
    return d


def test_scenario_guard_compares_content_and_defaults_to_golden(tmp_path):
    a = _scenario(tmp_path / "x", "wind")
    b = _scenario(tmp_path / "y", "wind")  # same scenario from another checkout
    c = _scenario(tmp_path / "z", "wind", req="改過的需求")  # same name, edited
    assert gp.check_same_scenario({"fixture": str(a)}, {"fixture": str(b)}) == "wind"
    with pytest.raises(SystemExit):
        gp.check_same_scenario({"fixture": str(a)}, {"fixture": str(c)})
    # A prepared.json from before --fixture is the golden scenario.
    assert gp.check_same_scenario({}, {"fixture": str(gp.FIXTURE)}) == "golden_scenario"
    with pytest.raises(SystemExit):
        gp.check_same_scenario({}, {"fixture": str(a)})


def test_requirement_prefers_the_copy_in_the_work_dir(tmp_path):
    fixture = _scenario(tmp_path, "wind", req="原始需求")
    work = tmp_path / "work"
    work.mkdir()
    assert gp.requirement_for(work, {"fixture": str(fixture)}) == "原始需求"
    (work / "simulation_requirement.txt").write_text("隨工作目錄的需求", encoding="utf-8")
    assert gp.requirement_for(work, {"fixture": str(tmp_path / "gone")}) == "隨工作目錄的需求"
    assert gp.requirement_for(tmp_path / "empty", {}) == (gp.FIXTURE / "simulation_requirement.txt").read_text(encoding="utf-8").strip()
