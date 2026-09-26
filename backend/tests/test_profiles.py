import pytest

from app.profiles import PROFILES, apply_profile


def test_no_profile_changes_nothing():
    env = {"GRAPH_BACKEND": "zep"}
    assert apply_profile(env) == {}
    assert env == {"GRAPH_BACKEND": "zep"}


def test_local_profile_fills_only_unset_variables():
    env = {"MIROFISH_PROFILE": "Local", "LLM_MODEL_NAME": "my-model", "REPORT_MODE": " "}
    applied = apply_profile(env)
    assert env["LLM_MODEL_NAME"] == "my-model"  # explicit settings win
    assert "LLM_MODEL_NAME" not in applied
    assert env["REPORT_MODE"] == "metrics"  # blank counts as unset
    assert env["SIM_DECISION_BACKEND"] == "system_one"
    assert env["GRAPH_BACKEND"] == "local" and env["ONTOLOGY_MODE"] == "template"


def test_local_llm_profile_keeps_llm_decisions_with_a_memory_budget():
    env = {"MIROFISH_PROFILE": "local-llm"}
    apply_profile(env)
    assert env["SIM_DECISION_BACKEND"] == "llm"
    assert int(env["SIM_AGENT_CONTEXT_TOKENS"]) < 8192
    assert env["GRAPH_BACKEND"] == "local"


def test_unknown_profile_is_an_error():
    with pytest.raises(ValueError):
        apply_profile({"MIROFISH_PROFILE": "cloud"})


def test_profiles_point_at_one_local_server():
    for profile in PROFILES.values():
        assert profile["LLM_BASE_URL"] == profile["SYSTEM_ONE_BASE_URL"]
        assert "127.0.0.1" in profile["EMBED_BASE_URL"]
