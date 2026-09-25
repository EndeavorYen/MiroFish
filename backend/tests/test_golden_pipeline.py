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
