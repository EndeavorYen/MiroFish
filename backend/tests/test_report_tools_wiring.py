"""The ReportAgent's tools reach the graph tools service it was given (#68).

The zep_* rename rewrote ``self.zep_tools.`` call sites to an attribute
that did not exist; every tool call then "failed" inside _execute_tool's
catch-all and reports were written without graph evidence.
"""

from types import SimpleNamespace

from app.services.report_agent import ReportAgent


class StubTools:
    def __init__(self):
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append(name)
        return SimpleNamespace(to_text=lambda: f"{name} ok")

    def get_graph_statistics(self, graph_id):
        self.calls.append("get_graph_statistics")
        return {"graph_id": graph_id, "total_nodes": 3}

    def quick_search(self, *args, **kwargs):
        return self._record("quick_search")

    def get_simulation_context(self, *args, **kwargs):
        self.calls.append("get_simulation_context")
        return {"graph_statistics": {}, "total_entities": 0, "related_facts": []}


def _agent(tools):
    return ReportAgent("g1", "sim-1", "OpenAI 漲價", llm_client=object(), zep_tools=tools)


def test_tool_calls_reach_the_given_service():
    tools = StubTools()
    agent = _agent(tools)
    assert "total_nodes" in agent._execute_tool("get_graph_statistics", {})
    assert "quick_search ok" in agent._execute_tool("quick_search", {"query": "票價"})
    assert tools.calls == ["get_graph_statistics", "quick_search"]


def test_the_outline_step_reads_the_simulation_context():
    tools = StubTools()

    class NoLLM:
        def chat_json(self, *args, **kwargs):
            return {"title": "t", "summary": "s", "sections": [{"title": "一", "description": "d"}]}

    agent = ReportAgent("g1", "sim-1", "OpenAI 漲價", llm_client=NoLLM(), zep_tools=tools)
    agent.plan_outline()
    assert "get_simulation_context" in tools.calls
