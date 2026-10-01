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
    from app.config import Config

    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports"))
    report = metrics_report.generate_metrics_report(
        "sim_test", "g", "req", "report_1", summary_fn=lambda p, m: "摘要"
    )
    assert report.status == ReportStatus.COMPLETED
    saved = ReportManager.get_report("report_1")
    assert saved.markdown_content.startswith("# 模擬指標報告")
    assert (tmp_path / "reports" / "report_1" / "report_metrics.json").exists()

    # The report page completes from agent_log.jsonl.
    log_path = tmp_path / "reports" / "report_1" / "agent_log.jsonl"
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    actions = [row["action"] for row in logs]
    assert actions[0] == "report_start" and actions[-1] == "report_complete"
    outline = next(row for row in logs if row["action"] == "planning_complete")["details"]["outline"]
    sections = [row for row in logs if row["action"] == "section_complete"]
    assert len(sections) == len(outline["sections"]) >= 3
    assert outline["sections"][0]["title"] == "摘要"
    assert sections[0]["section_index"] == 1
    assert sections[0]["details"]["content"].startswith("## 摘要")


def test_missing_emotion_table_and_bad_stance_are_ignored(sim_dir):
    import sqlite3

    from app.services.metrics_report import compute_metrics

    db = sim_dir / "agent_state.db"
    db.unlink()
    sqlite3.connect(db).close()  # created but never written
    with open(sim_dir / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"agent_id": 1, "intent": {"stance": "high"}}) + "\n")
    metrics = compute_metrics(str(sim_dir))
    assert metrics["emotion"] is None


def _scan_dir(tmp_path, stances, texts_by_round, structured=True):
    """agent_id -> configured stance 0..1; texts_by_round: [(round, agent_id, text)].

    ``structured`` writes the stance_raw the structured prep records; the LLM
    prep (the hybrid profile) writes the key as null, as the real config does."""

    config = {
        "simulation_requirement": "模擬空中計程車試點後，各方的反應。",
        "agent_configs": [
            {"agent_id": a, "entity_name": f"角色{a}", "entity_type": "Person", "sentiment_bias": s * 2 - 1,
             "stance_raw": s if structured else None}
            for a, s in stances.items()
        ],
    }
    (tmp_path / "simulation_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "twitter").mkdir()
    rows = [{"event_type": "round_start", "round": 0}] + [
        {"round": r, "agent_id": a, "agent_name": f"角色{a}", "action_type": "CREATE_POST",
         "action_args": {"content": text}}
        for r, a, text in texts_by_round
    ]
    (tmp_path / "twitter" / "actions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (tmp_path / "decisions.jsonl").write_text(json.dumps({"round": 1, "agent_id": 1, "intent": {"stance": 0.5}}), encoding="utf-8")
    return tmp_path


def test_scan_conclusions_with_their_confidence(tmp_path):
    from app.services.metrics_report import scan_conclusions

    score = {"讚": 0.9, "好": 0.75, "爛": 0.1}
    posts = [(1, 1, "讚"), (1, 2, "讚"), (2, 3, "爛"), (5, 1, "好"), (6, 3, "爛"), (6, 2, "讚")]
    sim = _scan_dir(tmp_path, {1: 0.9, 2: 0.8, 3: 0.1}, posts)
    questions = []

    def fake(text, question):
        questions.append(question)
        return score[text]

    scan = scan_conclusions(str(sim), score_fn=fake)
    assert set(questions) == {"這則貼文對「空中計程車試點」的立場是什麼？"}
    assert scan["local_path"] is True
    assert scan["main_camp"]["value"] == "support" and scan["main_camp"]["counts"] == {"oppose": 1, "neutral": 0, "support": 2}
    assert scan["main_camp"]["confidence"] == "high"
    assert scan["tendency"]["value"] == pytest.approx(sum(score[t] for _, _, t in posts) / len(posts), abs=1e-3)
    assert [row["name"] for row in scan["ranking"]["most_supportive"]][:1] == ["角色2"]
    assert scan["ranking"]["most_opposed"][0]["name"] == "角色3"
    assert scan["ranking"]["confidence"] == "low" and not scan["ranking"]["indistinct"]
    assert scan["trend"]["confidence"] == "low"
    text = render_markdown({**compute_metrics(str(sim)), "scan": scan})
    assert "## 掃描結論與可信度" in text and "local-llm" in text


def test_scan_flags_roles_it_cannot_tell_apart(tmp_path):
    from app.services.metrics_report import scan_conclusions

    sim = _scan_dir(tmp_path, {1: 0.5, 2: 0.52, 3: 0.49}, [(1, 1, "a"), (1, 2, "b"), (2, 3, "c")])
    scan = scan_conclusions(str(sim), score_fn=lambda text, question: 0.5)
    assert scan["ranking"]["indistinct"] is True
    assert scan["ranking"]["confidence"] == "none"
    assert "無法區分" in render_markdown({**compute_metrics(str(sim)), "scan": scan})


def test_scan_is_skipped_when_scoring_fails(sim_dir, tmp_path):
    def broken(text, question):
        raise ConnectionError("no model server")

    metrics, markdown = write_metrics_report(str(sim_dir), str(tmp_path), "req", score_fn=broken)
    assert metrics["scan"] is None
    assert "無法產生" in markdown  # the section says why instead of disappearing (#62)


def test_hybrid_runs_cite_their_own_evidence(tmp_path):
    from app.services.metrics_report import HYBRID_EVIDENCE, scan_conclusions

    sim = _scan_dir(tmp_path, {1: 0.5, 2: 0.5, 3: 0.5}, [(1, 1, "a"), (1, 2, "b"), (5, 3, "c")], structured=False)
    scan = scan_conclusions(str(sim), score_fn=lambda text, question: 0.5)
    assert scan["path"] == "hybrid"
    # Flat LLM-prep stances do not mean indistinct roles: the LLM agents' own
    # reading of the persona orders them (#53), so the flag is structured-only.
    assert scan["ranking"]["indistinct"] is False
    assert scan["ranking"]["confidence"] == HYBRID_EVIDENCE["ranking"][0]
    assert "混合" in scan["ranking"]["evidence"]


def test_failed_decisions_still_mark_the_local_path_and_warn(tmp_path):
    from app.services.metrics_report import decision_error_rate, scan_conclusions

    sim = _scan_dir(tmp_path, {1: 0.9, 2: 0.2, 3: 0.5}, [(1, 1, "a"), (2, 2, "b"), (9, 3, "c")])
    rows = [{"round": 1, "agent_id": 1, "action": "DO_NOTHING", "error": "ConnectError: refused"}] * 3
    rows += [{"round": 1, "agent_id": 2, "action": "CREATE_POST", "intent": {"stance": 0.4}}]
    (sim / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert decision_error_rate(str(sim)) == pytest.approx(0.75)
    scan = scan_conclusions(str(sim), score_fn=lambda text, question: 0.5)
    assert scan["path"] == "local"
    assert scan["trend"]["rounds"] == [0, 9]  # the last round with a post, not the window end
    text = render_markdown({**compute_metrics(str(sim)), "scan": scan, "decision_error_rate": 0.75})
    assert text.index("75%") < text.index("## 概況")  # the warning comes first


def test_a_scan_that_could_not_run_says_why(sim_dir, tmp_path):
    def broken(text, question):
        raise ConnectionError("refused http://127.0.0.1:8000/v1")

    posts = sim_dir / "twitter" / "actions.jsonl"
    posts.write_text(posts.read_text(encoding="utf-8") + json.dumps(
        {"round": 3, "agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "試點下月啟動"}},
        ensure_ascii=False) + "\n", encoding="utf-8")
    metrics, markdown = write_metrics_report(str(sim_dir), str(tmp_path), "req", score_fn=broken)
    assert metrics["scan"] is None
    assert "## 掃描結論與可信度" in markdown and "無法產生" in markdown and "ConnectionError" in markdown
