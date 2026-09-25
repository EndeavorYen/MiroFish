"""Deterministic metrics report (#12)."""

import json
import sqlite3

import pytest

from app.services.metrics_report import (
    SUMMARY_MAX_TOKENS,
    compute_metrics,
    render_markdown,
    write_metrics_report,
)
from app.simulation_policy.emotion import AgentStateStore


@pytest.fixture
def sim_dir(tmp_path):
    config = {
        "simulation_id": "sim_test",
        "agent_configs": [
            {"agent_id": 1, "entity_type": "GovernmentAgency", "stance": "supportive"},
            {"agent_id": 2, "entity_type": "Person", "stance": "opposing"},
            {"agent_id": 3, "entity_type": "Person", "stance": "neutral"},
        ],
        "event_config": {"hot_topics": ["票價", "噪音"]},
    }
    (tmp_path / "simulation_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "twitter").mkdir()
    actions = [
        {"event_type": "round_start", "round": 1},
        {"round": 1, "agent_id": 1, "action_type": "CREATE_POST"},
        {"round": 1, "agent_id": 2, "action_type": "REPOST"},
        {"round": 2, "agent_id": 3, "action_type": "LIKE_POST"},
        {"round": 2, "agent_id": 2, "action_type": "CREATE_POST"},
    ]
    (tmp_path / "twitter" / "actions.jsonl").write_text(
        "\n".join(json.dumps(a) for a in actions) + "\n", encoding="utf-8"
    )
    db = sqlite3.connect(tmp_path / "twitter_simulation.db")
    db.execute("CREATE TABLE user (user_id INTEGER, name TEXT)")
    db.execute(
        "CREATE TABLE post (post_id INTEGER, user_id INTEGER, original_post_id INTEGER, content TEXT,"
        " quote_content TEXT, created_at TEXT, num_likes INTEGER, num_shares INTEGER)"
    )
    db.executemany("INSERT INTO user VALUES (?, ?)", [(1, "交通委"), (2, "王淑芬"), (3, "陳維")])
    db.executemany(
        "INSERT INTO post VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, None, "試點下月啟動", None, "t1", 3, 2),
            (2, 2, 1, "試點下月啟動", None, "t2", 0, 0),
            (3, 3, 2, "試點下月啟動", "太貴了", "t3", 1, 0),
            (4, 2, None, "噪音太大", None, "t4", 0, 0),
        ],
    )
    db.commit()
    db.close()
    store = AgentStateStore(str(tmp_path / "agent_state.db"))
    store.save(1, "twitter", 1, {"anger": 0.1, "joy": 0.5})
    store.save(1, "twitter", 2, {"anger": 0.5, "joy": 0.1})
    store.save(2, "twitter", 2, {"anger": 0.7, "joy": 0.1})
    store.close()
    decisions = [
        {"round": 1, "agent_id": 2, "intent": {"stance": 0.2}},
        {"round": 2, "agent_id": 3, "intent": {"stance": 0.4}},
        {"round": 1, "agent_id": 1, "intent": {"stance": 0.9}},
    ]
    (tmp_path / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_metrics_are_deterministic(sim_dir, tmp_path_factory):
    first = compute_metrics(str(sim_dir))
    second = compute_metrics(str(sim_dir))
    assert first == second
    out_a = tmp_path_factory.mktemp("a")
    out_b = tmp_path_factory.mktemp("b")
    write_metrics_report(str(sim_dir), str(out_a), "req")
    write_metrics_report(str(sim_dir), str(out_b), "req")
    for name in ("report_metrics.json", "report_metrics.md"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_metric_values(sim_dir):
    metrics = compute_metrics(str(sim_dir))
    assert metrics["agents"] == 3
    assert metrics["actions"]["twitter"]["total"] == {"CREATE_POST": 2, "LIKE_POST": 1, "REPOST": 1}
    assert metrics["actions"]["twitter"]["by_round"]["2"] == {"CREATE_POST": 1, "LIKE_POST": 1}
    overall = metrics["emotion"]["twitter"]["overall"]
    assert overall["1"] == {"anger": 0.3, "joy": 0.3}
    assert metrics["emotion"]["twitter"]["by_entity_type"]["Person"]["2"] == {"anger": 0.7, "joy": 0.1}
    top = metrics["spread"]["twitter"][0]
    assert top["post_id"] == 1 and top["reposts_and_quotes"] == 2
    assert [s["by"] for s in top["chain"]] == ["王淑芬", "陳維"]
    assert [s["kind"] for s in top["chain"]] == ["repost", "quote"]
    assert metrics["stance"]["Person"]["expressed_stance_mean"] == 0.3
    assert metrics["stance"]["Person"]["configured_stances"] == {"neutral": 1, "opposing": 1}


def test_markdown_and_summary_budget(sim_dir, tmp_path):
    from app.utils.llm_usage import usage_stage  # noqa: F401 - stage is set inside

    calls = []

    def summary(prompt, max_tokens):
        calls.append((prompt, max_tokens))
        return "政府支持、民眾偏反對，擴散集中在試點消息。"

    metrics, markdown = write_metrics_report(
        str(sim_dir), str(tmp_path / "report"), "模擬需求", summary_fn=summary
    )
    assert calls and calls[0][1] == SUMMARY_MAX_TOKENS
    assert "模擬需求" in calls[0][0] and "試點下月啟動" in calls[0][0]
    assert markdown.startswith("# 模擬指標報告")
    assert "政府支持" in markdown and "擴散路徑：王淑芬 → 陳維" in markdown
    assert json.loads((tmp_path / "report" / "report_metrics.json").read_text(encoding="utf-8")) == metrics


def test_llm_runs_without_emotion_state(sim_dir):
    (sim_dir / "agent_state.db").unlink()
    (sim_dir / "decisions.jsonl").unlink()
    metrics = compute_metrics(str(sim_dir))
    assert metrics["emotion"] is None
    assert metrics["stance"]["Person"]["expressed_stance_mean"] is None
    assert "LLM 決策模式不記錄情緒" in render_markdown(metrics)


def test_summary_failure_keeps_the_report(sim_dir, tmp_path):
    def broken(prompt, max_tokens):
        raise TimeoutError("down")

    _, markdown = write_metrics_report(str(sim_dir), str(tmp_path), "req", summary_fn=broken)
    assert "摘要產生失敗" in markdown and "## 動作分布" in markdown


def test_generate_metrics_report_saves_a_completed_report(sim_dir, tmp_path, monkeypatch):
    from app.services import metrics_report
    from app.services.report_agent import ReportManager, ReportStatus
    from app.services.simulation_manager import SimulationManager

    monkeypatch.setattr(SimulationManager, "_get_simulation_dir", lambda self, sid: str(sim_dir))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports"))
    report = metrics_report.generate_metrics_report(
        "sim_test", "g", "req", "report_1", summary_fn=lambda p, m: "摘要"
    )
    assert report.status == ReportStatus.COMPLETED
    saved = ReportManager.get_report("report_1")
    assert saved.markdown_content.startswith("# 模擬指標報告")
    assert (tmp_path / "reports" / "report_1" / "report_metrics.json").exists()
