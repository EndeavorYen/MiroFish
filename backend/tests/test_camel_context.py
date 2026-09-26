import pytest

from app.utils.camel_context import (
    TurnBudgetMemory,
    apply_agent_graph_budget,
    apply_memory_budget,
    context_budget_from_env,
)


@pytest.fixture
def agent(monkeypatch):
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    monkeypatch.setenv("OPENAI_API_KEY", "local")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://127.0.0.1:9/v1")
    backend = ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type="qwen3.5-4b")
    assert backend.token_limit > 10_000_000  # why agent memory grew unbounded
    return ChatAgent(system_message="你是一位東海市居民。", model=backend)


def _turn(agent, i, feed_chars=400):
    """One OASIS step: feed observation, a tool call and its result."""

    from camel.messages import BaseMessage, FunctionCallingMessage
    from camel.types import OpenAIBackendRole

    agent.update_memory(
        BaseMessage.make_user_message(role_name="User", content=f"第{i}回合動態：" + "貼文" * feed_chars),
        OpenAIBackendRole.USER,
    )
    call = FunctionCallingMessage(
        role_name="assistant", role_type=BaseMessage.make_assistant_message("a", "").role_type,
        meta_dict=None, content="", func_name="like_post", args={"post_id": i}, tool_call_id=f"c{i}",
    )
    agent.update_memory(call, OpenAIBackendRole.ASSISTANT)
    result = FunctionCallingMessage(
        role_name="assistant", role_type=call.role_type, meta_dict=None, content="",
        func_name="like_post", result={"success": True}, tool_call_id=f"c{i}",
    )
    agent.update_memory(result, OpenAIBackendRole.FUNCTION)


def test_memory_keeps_whole_recent_turns_within_budget(agent):
    apply_memory_budget(agent, 3000)
    assert isinstance(agent.memory, TurnBudgetMemory)
    for i in range(30):
        _turn(agent, i)
    messages, tokens = agent.memory.get_context()
    assert tokens <= 3000
    assert messages[0]["role"] == "system"
    users = [m for m in messages if m["role"] == "user"]
    assert 1 <= len(users) < 30
    assert "第29回合" in users[-1]["content"]  # the newest turn is kept
    # Whole turns only: every tool result follows its call, the window starts at a user message.
    assert messages[1]["role"] == "user"
    call_ids = [c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    result_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert result_ids == call_ids
    # Nothing is sliced into chunks.
    assert not any("[chunk" in str(m.get("content")) for m in messages)


def test_current_turn_is_kept_even_over_budget(agent):
    apply_memory_budget(agent, 50)
    _turn(agent, 0)
    _turn(agent, 1)
    messages, _ = agent.memory.get_context()
    users = [m for m in messages if m["role"] == "user"]
    assert len(users) == 1 and "第1回合" in users[0]["content"]


def test_graph_budget_and_none(agent):
    class Graph:
        def get_agents(self):
            return [(0, agent)]

    assert apply_agent_graph_budget(Graph(), None) == 0
    assert not isinstance(agent.memory, TurnBudgetMemory)
    assert apply_agent_graph_budget(Graph(), 4096) == 1
    assert agent.memory._turn_budget == 4096


def test_budget_from_env(monkeypatch):
    from app.utils.camel_context import DEFAULT_LOCAL_BUDGET

    monkeypatch.delenv("SIM_AGENT_CONTEXT_TOKENS", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    assert context_budget_from_env() is None  # hosted model: camel's default
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    assert context_budget_from_env() == DEFAULT_LOCAL_BUDGET  # local model: safe default
    monkeypatch.setenv("SIM_AGENT_CONTEXT_TOKENS", " 4096 ")
    assert context_budget_from_env() == 4096
    for off in ("0", "off"):
        monkeypatch.setenv("SIM_AGENT_CONTEXT_TOKENS", off)
        assert context_budget_from_env() is None
    for bad in ("abc", "-5"):
        monkeypatch.setenv("SIM_AGENT_CONTEXT_TOKENS", bad)
        with pytest.raises(ValueError):
            context_budget_from_env()


def test_extra_system_records_are_not_charged(agent):
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    apply_memory_budget(agent, 2000)
    for i in range(40):  # OASIS writes one per ManualAction; camel never sends them
        agent.update_memory(BaseMessage.make_assistant_message("system", "初始貼文" * 50), OpenAIBackendRole.SYSTEM)
    for i in range(3):
        _turn(agent, i, feed_chars=50)
    messages, _ = agent.memory.get_context()
    assert len([m for m in messages if m["role"] == "user"]) == 3


def test_oasis_observations_keep_chinese_characters():
    from oasis.social_agent import agent_environment

    from app.utils.oasis_prompts import install_unicode_observations

    install_unicode_observations()
    install_unicode_observations()  # idempotent
    text = agent_environment.json.dumps([{"content": "東海市空中計程車", "likes": 1}], indent=4)
    assert text == '[{"content":"東海市空中計程車","likes":1}]'  # compact, same content
    assert "東海市空中計程車" in text and "\\u" not in text
    assert agent_environment.json.loads(text)[0]["content"] == "東海市空中計程車"
    assert agent_environment.json.dumps("東", ensure_ascii=True) == '"\\u6771"'  # explicit wins


def test_only_pretty_printed_observations_are_compacted():
    from oasis.social_agent import agent_environment

    from app.utils.oasis_prompts import install_unicode_observations

    install_unicode_observations()
    assert agent_environment.json.dumps({"a": ["群"]}) == '{"a": ["群"]}'  # group calls keep default spacing
    assert agent_environment.json.dumps([1], indent=0) == "[\n1\n]"  # falsy indent untouched
    assert agent_environment.json.dumps([1, 2], indent=2, separators=(", ", ": ")) == "[\n  1, \n  2\n]"
