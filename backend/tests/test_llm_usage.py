"""
Tests for LLM usage tracking and camel wrapper.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from app.utils.llm_usage import (
    VALID_STAGES,
    record_usage,
    usage_stage,
    wrap_camel_model,
)
from app.utils.openai_chat_compat import create_chat_completion


class FakeUsage:
    def __init__(self, prompt_tokens: int = 150, completion_tokens: int = 80):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMessage:
    def __init__(self, content: str = ""):
        self.content = content


class FakeChoice:
    def __init__(self, content: str = "", finish_reason: str = "stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(
        self,
        prompt_tokens: int = 150,
        completion_tokens: int = 80,
        model: str = "gpt-4o",
        content: str = "",
    ):
        self.usage = FakeUsage(prompt_tokens, completion_tokens)
        self.model = model
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, response=None):
        self.response = response or FakeResponse()

    def create(self, **kwargs):
        return self.response


class FakeClient:
    def __init__(self, response=None):
        self.chat = MagicMock()
        self.chat.completions = FakeChatCompletions(response)


def test_no_metrics_dir_is_noop(tmp_path):
    client = FakeClient()
    # No usage_stage active, no env var
    resp = create_chat_completion(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp is not None
    # No file created in tmp_path
    assert len(list(tmp_path.iterdir())) == 0


def test_create_chat_completion_records_usage(tmp_path):
    client = FakeClient(FakeResponse(prompt_tokens=120, completion_tokens=45, model="gpt-4o"))
    metrics_dir = str(tmp_path / "metrics")

    with usage_stage("ontology", metrics_dir=metrics_dir):
        resp = create_chat_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )
        assert resp is not None

    jsonl_path = tmp_path / "metrics" / "llm_usage.jsonl"
    assert jsonl_path.is_file()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    record = lines[0]
    # Check four required fields
    assert "stage" in record
    assert "prompt_tokens" in record
    assert "completion_tokens" in record
    assert "latency_ms" in record

    assert record["stage"] == "ontology"
    assert record["prompt_tokens"] == 120
    assert record["completion_tokens"] == 45
    assert record["latency_ms"] >= 0
    assert record["model"] == "gpt-4o"


def test_env_var_overrides_metrics_dir(tmp_path, monkeypatch):
    override_dir = str(tmp_path / "env_metrics")
    monkeypatch.setenv("MIROFISH_METRICS_DIR", override_dir)

    client = FakeClient()
    with usage_stage("profile", metrics_dir=str(tmp_path / "ignored_dir")):
        create_chat_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )

    # File should be in override_dir, not ignored_dir
    assert (tmp_path / "env_metrics" / "llm_usage.jsonl").is_file()
    assert not (tmp_path / "ignored_dir" / "llm_usage.jsonl").exists()


def test_invalid_stage_raises_error():
    with pytest.raises(ValueError, match="Invalid stage"):
        with usage_stage("non_existent_stage"):
            pass


class FakeCamelBackend:
    def __init__(self, response=None):
        self.response = response or FakeResponse(prompt_tokens=200, completion_tokens=60, model="gpt-4o-mini")
        self.model_type = "gpt-4o-mini"

    def run(self, *args, **kwargs):
        return self.response

    async def arun(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_camel_wrapper_sync_and_async(tmp_path):
    metrics_dir = str(tmp_path / "sim_metrics")
    fake_backend = FakeCamelBackend()
    wrapped = wrap_camel_model(fake_backend, default_stage="simulation")

    with usage_stage("simulation", metrics_dir=metrics_dir):
        # Sync run
        wrapped.run("hello")

        # Async arun under interview stage
        with usage_stage("interview"):
            await wrapped.arun("interview prompt")

    jsonl_path = tmp_path / "sim_metrics" / "llm_usage.jsonl"
    assert jsonl_path.is_file()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 2
    # First record: simulation
    assert records[0]["stage"] == "simulation"
    assert records[0]["prompt_tokens"] == 200
    assert records[0]["completion_tokens"] == 60
    assert "latency_ms" in records[0]

    # Second record: interview
    assert records[1]["stage"] == "interview"
    assert records[1]["prompt_tokens"] == 200
    assert records[1]["completion_tokens"] == 60
    assert "latency_ms" in records[1]


def test_threadpool_contextvar_propagation(tmp_path):
    import concurrent.futures
    import contextvars

    metrics_dir = str(tmp_path / "metrics")

    def worker_func():
        return record_usage(FakeResponse(), latency_ms=15.0)

    with usage_stage("profile", metrics_dir=metrics_dir):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut = executor.submit(contextvars.copy_context().run, worker_func)
            entry = fut.result()

    assert entry["stage"] == "profile"
    metrics_file = tmp_path / "metrics" / "llm_usage.jsonl"
    assert metrics_file.is_file()
    with open(metrics_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["stage"] == "profile"


def test_oasis_profile_generator_records_profile_stage_metrics(tmp_path, monkeypatch):
    from app.services.oasis_profile_generator import OasisProfileGenerator
    from app.services.zep_entity_reader import EntityNode

    metrics_dir = str(tmp_path / "metrics")
    generator = OasisProfileGenerator(api_key="fake-key")

    fake_profile_json = json.dumps({
        "user_name": "AliceSmith",
        "bio": "Bio of Alice",
        "persona": "Persona of Alice",
        "topics": ["Tech"],
        "activity_level": 0.8,
    })
    mock_resp = FakeResponse(prompt_tokens=150, completion_tokens=80, model="gpt-4o-mini", content=fake_profile_json)

    def fake_create_chat_completion(*args, **kwargs):
        record_usage(mock_resp, latency_ms=25.0, model="gpt-4o-mini")
        return mock_resp

    monkeypatch.setattr(
        "app.services.oasis_profile_generator.create_chat_completion",
        fake_create_chat_completion,
    )

    entities = [
        EntityNode(uuid="1", name="Alice", labels=["Person"], summary="Expert", attributes={}),
        EntityNode(uuid="2", name="Bob", labels=["Person"], summary="Analyst", attributes={}),
    ]

    with usage_stage("profile", metrics_dir=metrics_dir):
        profiles = generator.generate_profiles_from_entities(
            entities=entities,
            use_llm=True,
            parallel_count=2,
        )

    assert len(profiles) == 2
    metrics_file = tmp_path / "metrics" / "llm_usage.jsonl"
    assert metrics_file.is_file()
    with open(metrics_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    for rec in records:
        assert rec["stage"] == "profile"
        assert rec["prompt_tokens"] == 150
        assert rec["completion_tokens"] == 80


def test_record_usage_error_isolation(tmp_path, monkeypatch):
    def bad_open(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr("builtins.open", bad_open)
    with usage_stage("simulation", metrics_dir=str(tmp_path)):
        # Should log warning and not raise an exception
        entry = record_usage(FakeResponse(), latency_ms=10.0)
        assert entry["stage"] == "simulation"
