"""LLM clients are built in one place (#68)."""

import ast
from pathlib import Path

import pytest

from app.config import Config
from app.utils import llm_factory
from app.utils.llm_factory import CONTENT, PREP, SUMMARY, make_llm_client

APP = Path(__file__).resolve().parents[1] / "app"


def test_the_factory_resolves_config_and_policy(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", "k")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = make_llm_client(CONTENT)
    assert client.api_key == "k" and str(client.base_url).startswith("http://127.0.0.1:8000/v1")
    assert client.timeout == 30 and client.max_retries == 0
    assert make_llm_client(SUMMARY).timeout == 120
    default = make_llm_client(PREP)
    assert default.max_retries == llm_factory.OpenAI(api_key="x").max_retries  # SDK default kept


def test_a_missing_key_raises_unless_optional(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", None)
    with pytest.raises(ValueError):
        make_llm_client()
    assert make_llm_client(require_key=False).api_key == "local"


def test_no_other_module_builds_an_openai_client():
    offenders = []
    for path in APP.rglob("*.py"):
        if path.name == "llm_factory.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name in ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"):
                    offenders.append(f"{path.relative_to(APP)}:{node.lineno}")
    assert not offenders, offenders
