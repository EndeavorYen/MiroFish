import pytest

from app.config import Config
from app.graph.store import get_graph_store
from app.graph.zep_store import ZepGraphStore


def test_local_backend_raises_not_implemented(monkeypatch):
    monkeypatch.delenv("ZEP_API_KEY", raising=False)
    calls = {"count": 0}

    def reject_client(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("get_zep_client should not be called")

    monkeypatch.setattr("app.utils.zep.get_zep_client", reject_client)

    with pytest.raises(NotImplementedError, match="GRAPH_BACKEND=local"):
        get_graph_store(backend="local")
    assert calls["count"] == 0


def test_default_backend_returns_zep_store(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
    monkeypatch.setattr("app.utils.zep.get_zep_client", lambda _key: object())

    store = get_graph_store(api_key="k")

    assert isinstance(store, ZepGraphStore)


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="neo4j"):
        get_graph_store(backend="neo4j")


def test_validate_skips_zep_checks_for_local(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "local")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    monkeypatch.setattr(Config, "LLM_API_KEY", "x")
    monkeypatch.setenv("ZEP_API_URL", "http://x")

    errors = Config.validate()

    assert not any("ZEP_API_KEY" in error or "ZEP_API_URL" in error for error in errors)


def test_validate_requires_zep_key_by_default_and_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    monkeypatch.setattr(Config, "LLM_API_KEY", "x")
    monkeypatch.delenv("ZEP_API_URL", raising=False)

    zep_errors = Config.validate()
    assert any("ZEP_API_KEY 未配置" in error for error in zep_errors)

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "neo4j")
    unknown_errors = Config.validate()
    assert any("GRAPH_BACKEND 必须是 zep 或 local" in error for error in unknown_errors)
