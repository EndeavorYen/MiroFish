"""
Tests for sim_metrics.py script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.sim_metrics import (
    compute_distinct_2,
    compute_sim_metrics,
    format_metrics_json,
    load_action_entries,
    tokenize_text,
)


@pytest.fixture
def fixture_actions_path() -> str:
    path = Path(__file__).parent / "fixtures" / "golden_scenario" / "actions.jsonl"
    assert path.is_file(), f"Fixture actions file not found: {path}"
    return str(path)


def test_tokenize_text():
    # Chinese + English
    text = "東海市 2026 年启动了！hello world"
    tokens = tokenize_text(text)
    assert "東" in tokens
    assert "海" in tokens
    assert "2026" in tokens
    assert "hello" in tokens
    assert "world" in tokens


def test_compute_distinct_2():
    # Distinct bigrams
    # tokens: ['a', 'b', 'c'] -> ('a', 'b'), ('b', 'c') -> 2 unique / 2 total = 1.0
    assert compute_distinct_2(["a b c"]) == 1.0

    # Repeated bigrams: ['a', 'b', 'a', 'b'] -> ('a', 'b'), ('b', 'a'), ('a', 'b') -> 2 unique / 3 total = 2/3
    assert compute_distinct_2(["a b a b"]) == round(2 / 3, 4)

    # Empty or single token -> 0.0
    assert compute_distinct_2([]) == 0.0
    assert compute_distinct_2(["hello"]) == 0.0


def test_load_action_entries_skips_events(fixture_actions_path):
    entries = load_action_entries(fixture_actions_path)
    # Ensure no entry has 'event_type'
    assert len(entries) > 0
    for e in entries:
        assert "event_type" not in e
        assert "action_type" in e
        assert "round" in e


def test_compute_sim_metrics_structure(fixture_actions_path):
    metrics = compute_sim_metrics(fixture_actions_path)

    assert "rounds" in metrics
    assert "summary" in metrics
    assert "total_rounds" in metrics
    assert metrics["total_rounds"] == 2

    # Round 1 checks
    r1 = metrics["rounds"][0]
    assert r1["round"] == 1
    assert r1["active_agents_count"] == 4
    assert r1["action_type_distribution"] == {
        "CREATE_POST": 2,
        "DO_NOTHING": 1,
        "LIKE_POST": 1,
    }
    assert r1["post_count"] == 2
    assert r1["distinct_2"] > 0.0

    # Round 2 checks
    r2 = metrics["rounds"][1]
    assert r2["round"] == 2
    assert r2["active_agents_count"] == 3
    assert r2["action_type_distribution"] == {
        "CREATE_POST": 2,
        "REPOST": 1,
    }
    assert r2["post_count"] == 2
    assert r2["distinct_2"] > 0.0

    summary = metrics["summary"]
    assert summary["total_actions"] == 7
    assert summary["total_posts"] == 4
    assert summary["overall_active_agents_count"] == 5
    assert summary["overall_distinct_2"] > 0.0


def test_deterministic_json_output(fixture_actions_path):
    # Running compute_sim_metrics twice produces bit-for-bit identical JSON
    out1 = format_metrics_json(compute_sim_metrics(fixture_actions_path))
    out2 = format_metrics_json(compute_sim_metrics(fixture_actions_path))
    assert out1 == out2


def test_directory_metrics_stay_split_by_platform(fixture_actions_path, tmp_path):
    sim_dir = tmp_path / "sim"
    twitter_dir = sim_dir / "twitter"
    reddit_dir = sim_dir / "reddit"
    twitter_dir.mkdir(parents=True)
    reddit_dir.mkdir()

    twitter_actions = Path(fixture_actions_path).read_text(encoding="utf-8")
    (twitter_dir / "actions.jsonl").write_text(twitter_actions, encoding="utf-8")
    reddit_line = json.dumps(
        {
            "round": 1,
            "agent_id": 1,
            "agent_name": "RedditOnly",
            "action_type": "CREATE_POST",
            "action_args": {"content": "reddit only post"},
        },
        ensure_ascii=False,
    )
    (reddit_dir / "actions.jsonl").write_text(reddit_line + "\n", encoding="utf-8")

    metrics = compute_sim_metrics(str(sim_dir))
    assert set(metrics["platforms"]) == {"reddit", "twitter"}

    twitter = metrics["platforms"]["twitter"]
    reddit = metrics["platforms"]["reddit"]
    assert twitter["rounds"][0]["active_agents_count"] == 4
    assert twitter["summary"]["total_actions"] == 7
    assert reddit["rounds"][0]["active_agents_count"] == 1
    assert reddit["rounds"][0]["action_type_distribution"] == {"CREATE_POST": 1}
    assert reddit["summary"]["total_actions"] == 1

    again = format_metrics_json(compute_sim_metrics(str(sim_dir)))
    assert format_metrics_json(metrics) == again


def test_cli_execution(fixture_actions_path, tmp_path):
    script_path = str(Path(__file__).parent.parent / "scripts" / "sim_metrics.py")
    out_file = tmp_path / "metrics_out.json"

    cmd = [sys.executable, script_path, fixture_actions_path, "-o", str(out_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"

    assert out_file.is_file()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_rounds"] == 2
    assert data["summary"]["total_actions"] == 7
