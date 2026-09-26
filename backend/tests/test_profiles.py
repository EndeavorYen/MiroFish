import importlib
import re
from pathlib import Path

import pytest

from app.profiles import DROPPED, LOCAL_EMBED, LOCAL_LLM, PROFILES, apply_profile

BACKEND = Path(__file__).resolve().parents[1]


def test_no_profile_changes_nothing():
    env = {"GRAPH_BACKEND": "zep", "LLM_BASE_URL": "https://api.example.com/v1"}
    assert apply_profile(env) == {}
    assert env == {"GRAPH_BACKEND": "zep", "LLM_BASE_URL": "https://api.example.com/v1"}


def test_local_profile_fills_unset_modes_and_keeps_explicit_ones():
    env = {"MIROFISH_PROFILE": "Local", "REPORT_MODE": "agent", "CONTENT_MODE": " "}
    apply_profile(env)
    assert env["REPORT_MODE"] == "agent"  # explicit mode wins
    assert env["CONTENT_MODE"] == "tiered"  # blank counts as unset
    assert env["SIM_DECISION_BACKEND"] == "system_one"
    assert env["GRAPH_BACKEND"] == "local" and env["ONTOLOGY_MODE"] == "template"
    assert env["LLM_BASE_URL"] == env["SYSTEM_ONE_BASE_URL"] == LOCAL_LLM["LLM_BASE_URL"]


def test_hosted_endpoints_from_env_example_are_replaced():
    # The values .env.example ships with, plus the Reddit boost endpoint.
    env = {
        "MIROFISH_PROFILE": "local",
        "LLM_API_KEY": "your_api_key_here",
        "LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "LLM_MODEL_NAME": "qwen-plus",
        "EMBED_BASE_URL": "https://embeddings.example.com/v1",
        "LLM_BOOST_API_KEY": "k",
        "LLM_BOOST_BASE_URL": "https://boost.example.com/v1",
        "LLM_BOOST_MODEL_NAME": "m",
    }
    apply_profile(env)
    for key, value in {**LOCAL_LLM, **LOCAL_EMBED}.items():
        assert env[key] == value
    assert not any(key in env for key in DROPPED)
    assert env["SYSTEM_ONE_MODEL"] == LOCAL_LLM["LLM_MODEL_NAME"]


def test_local_endpoint_and_model_the_user_set_are_kept_and_followed():
    env = {
        "MIROFISH_PROFILE": "local",
        "LLM_BASE_URL": "http://localhost:8010/v1",
        "LLM_MODEL_NAME": "SubSir/Qwen3.5-4B-AWQ",
        "LLM_API_KEY": "x",
    }
    apply_profile(env)
    assert env["LLM_BASE_URL"] == "http://localhost:8010/v1"
    assert env["SYSTEM_ONE_BASE_URL"] == "http://localhost:8010/v1"  # System One follows
    assert env["SYSTEM_ONE_MODEL"] == "SubSir/Qwen3.5-4B-AWQ"


def test_local_llm_profile_keeps_llm_decisions_with_a_memory_budget():
    env = {"MIROFISH_PROFILE": "local-llm"}
    apply_profile(env)
    assert env["SIM_DECISION_BACKEND"] == "llm"
    assert int(env["SIM_AGENT_CONTEXT_TOKENS"]) < 8192
    assert env["GRAPH_BACKEND"] == "local"


def test_unknown_profile_is_an_error():
    with pytest.raises(ValueError):
        apply_profile({"MIROFISH_PROFILE": "cloud"})


def test_default_environ_and_config_pick_up_the_profile(monkeypatch):
    import app.config as config

    for key in ("GRAPH_BACKEND", "REPORT_MODE", "SIM_CONFIG_MODE", "LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MIROFISH_PROFILE", "local")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore any repo .env
    try:
        importlib.reload(config)
        assert config.Config.GRAPH_BACKEND == "local"
        assert config.Config.REPORT_MODE == "metrics"
        assert config.Config.SIM_CONFIG_MODE == "structured"
        assert config.Config.LLM_BASE_URL == LOCAL_LLM["LLM_BASE_URL"]
    finally:
        monkeypatch.delenv("MIROFISH_PROFILE")
        importlib.reload(config)


def test_every_profile_key_is_read_somewhere():
    sources = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for folder in ("app", "scripts")
        for p in (BACKEND / folder).rglob("*.py")
        if p.name != "profiles.py"
    )
    keys = set(LOCAL_LLM) | set(LOCAL_EMBED) | {k for profile in PROFILES.values() for k in profile}
    missing = [k for k in sorted(keys) if not re.search(rf"[\"']{k}[\"']", sources)]
    assert not missing
