"""Tests for the local compose profile and .env.example keys."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _command_text(service: dict) -> str:
    command = service.get("command", "")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def _flag_value(text: str, flag: str) -> str | None:
    parts = text.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return None


def test_local_profile_services():
    document = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = document["services"]

    local_llm = services["local-llm"]
    local_embed = services["local-embed"]
    assert local_llm["profiles"] == ["local"]
    assert local_embed["profiles"] == ["local"]

    command = _command_text(local_llm)
    assert "--enable-prefix-caching" in command
    assert int(_flag_value(command, "--max-logprobs")) >= 64
    assert int(_flag_value(command, "--max-model-len")) >= 8192

    devices = (
        local_embed.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices")
    )
    assert not devices
    assert "profiles" not in services["mirofish"]


def test_env_example_local_keys():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LLM_BASE_URL=http://localhost:8000/v1" in text
    assert "EMBED_BASE_URL=" in text
    assert "EMBED_MODEL_NAME=" in text
    assert "LOCAL_LLM_MODEL=" in text
