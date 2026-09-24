"""
Tests for deterministic agent selection with --seed in simulation runners.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.run_twitter_simulation import TwitterSimulationRunner
from scripts.run_reddit_simulation import RedditSimulationRunner
from scripts.run_parallel_simulation import get_active_agents_for_round


def _make_config(path, seed=None):
    agent_configs = [
        {
            "agent_id": i,
            "active_hours": list(range(24)),
            "activity_level": 0.5 + (i % 5) * 0.1,
        }
        for i in range(1, 25)
    ]
    config_data = {
        "simulation_id": "test_sim_seeded",
        "seed": seed,
        "time_config": {
            "agents_per_hour_min": 3,
            "agents_per_hour_max": 8,
            "peak_hours": [9, 10, 11, 14, 15, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "peak_activity_multiplier": 1.5,
            "off_peak_activity_multiplier": 0.3,
        },
        "agent_configs": agent_configs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)
    return str(path)


def _make_mock_env():
    mock_env = MagicMock()
    mock_env.agent_graph.get_agent = lambda aid: SimpleNamespace(id=aid, name=f"agent_{aid}")
    return mock_env


def test_twitter_simulation_runner_deterministic_with_seed(tmp_path):
    cfg_file = _make_config(tmp_path / "config.json")
    mock_env = _make_mock_env()

    runner1 = TwitterSimulationRunner(cfg_file, wait_for_commands=False, seed=42)
    runner2 = TwitterSimulationRunner(cfg_file, wait_for_commands=False, seed=42)
    runner3 = TwitterSimulationRunner(cfg_file, wait_for_commands=False, seed=999)

    seq1, seq2, seq3 = [], [], []
    for r in range(12):
        hour = (r * 2) % 24
        agents1 = runner1._get_active_agents_for_round(mock_env, current_hour=hour, round_num=r)
        agents2 = runner2._get_active_agents_for_round(mock_env, current_hour=hour, round_num=r)
        agents3 = runner3._get_active_agents_for_round(mock_env, current_hour=hour, round_num=r)
        seq1.append([aid for aid, _ in agents1])
        seq2.append([aid for aid, _ in agents2])
        seq3.append([aid for aid, _ in agents3])

    assert seq1 == seq2, "Identical seed must produce identical active agent sequences round-by-round"
    assert seq1 != seq3, "Different seeds should produce different active agent sequences"


def test_reddit_simulation_runner_deterministic_with_seed(tmp_path):
    cfg_file = _make_config(tmp_path / "config.json")
    mock_env = _make_mock_env()

    runner1 = RedditSimulationRunner(cfg_file, wait_for_commands=False, seed=42)
    runner2 = RedditSimulationRunner(cfg_file, wait_for_commands=False, seed=42)
    runner3 = RedditSimulationRunner(cfg_file, wait_for_commands=False, seed=999)

    seq1, seq2, seq3 = [], [], []
    for r in range(12):
        hour = (r * 2) % 24
        agents1 = runner1._get_active_agents_for_round(mock_env, current_hour=hour, round_num=r)
        agents2 = runner2._get_active_agents_for_round(mock_env, current_hour=hour, round_num=r)
        agents3 = runner3._get_active_agents_for_round(mock_env, current_hour=hour, round_num=r)
        seq1.append([aid for aid, _ in agents1])
        seq2.append([aid for aid, _ in agents2])
        seq3.append([aid for aid, _ in agents3])

    assert seq1 == seq2, "Identical seed must produce identical active agent sequences round-by-round"
    assert seq1 != seq3, "Different seeds should produce different active agent sequences"


def test_parallel_simulation_get_active_agents_deterministic(tmp_path):
    cfg_file = _make_config(tmp_path / "config.json")
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    mock_env = _make_mock_env()

    rng1 = random.Random(42)
    rng2 = random.Random(42)
    rng3 = random.Random(999)

    seq1, seq2, seq3 = [], [], []
    for r in range(12):
        hour = (r * 2) % 24
        agents1 = get_active_agents_for_round(mock_env, config, hour, r, rng=rng1)
        agents2 = get_active_agents_for_round(mock_env, config, hour, r, rng=rng2)
        agents3 = get_active_agents_for_round(mock_env, config, hour, r, rng=rng3)
        seq1.append([aid for aid, _ in agents1])
        seq2.append([aid for aid, _ in agents2])
        seq3.append([aid for aid, _ in agents3])

    assert seq1 == seq2, "Parallel get_active_agents_for_round must be deterministic with same rng"
    assert seq1 != seq3, "Different rng seeds must produce different agent sequences"


def test_parallel_simulation_platform_rng_independence(tmp_path):
    cfg_file = _make_config(tmp_path / "config.json")
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    mock_env = _make_mock_env()

    # Run Twitter and Reddit sequentially
    tw_rng_seq = random.Random(42)
    rd_rng_seq = random.Random(43)
    tw_seq = [
        [aid for aid, _ in get_active_agents_for_round(mock_env, config, (r * 2) % 24, r, rng=tw_rng_seq)]
        for r in range(6)
    ]
    rd_seq = [
        [aid for aid, _ in get_active_agents_for_round(mock_env, config, (r * 2) % 24, r, rng=rd_rng_seq)]
        for r in range(6)
    ]

    # Run Twitter and Reddit interleaved
    tw_rng_intl = random.Random(42)
    rd_rng_intl = random.Random(43)
    tw_intl, rd_intl = [], []
    for r in range(6):
        h = (r * 2) % 24
        tw_agents = get_active_agents_for_round(mock_env, config, h, r, rng=tw_rng_intl)
        rd_agents = get_active_agents_for_round(mock_env, config, h, r, rng=rd_rng_intl)
        tw_intl.append([aid for aid, _ in tw_agents])
        rd_intl.append([aid for aid, _ in rd_agents])

    assert tw_seq == tw_intl, "Twitter RNG must be unaffected by Reddit execution"
    assert rd_seq == rd_intl, "Reddit RNG must be unaffected by Twitter execution"


def test_runner_reads_seed_from_config(tmp_path):
    cfg_file = _make_config(tmp_path / "config.json", seed=123)
    runner = TwitterSimulationRunner(cfg_file, wait_for_commands=False)
    assert runner.seed == 123
    assert runner.rng is not random
    assert runner.wait_for_commands is False


def _write_runner_config(path, *, hours: int, initial_posts):
    payload = {
        "simulation_id": "empty_initial_posts",
        "time_config": {
            "total_simulation_hours": hours,
            "minutes_per_round": 30,
        },
        "agent_configs": [],
        "event_config": {"initial_posts": initial_posts},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _patch_runner_env(monkeypatch, module_name: str, graph_name: str):
    env = MagicMock()
    env.reset = AsyncMock()
    env.step = AsyncMock()
    env.close = AsyncMock()

    async def fake_graph(**kwargs):
        return MagicMock()

    monkeypatch.setattr(f"{module_name}.{graph_name}", fake_graph)
    monkeypatch.setattr(f"{module_name}.oasis.make", lambda **kwargs: env)
    monkeypatch.setattr(
        f"{module_name}.IPCHandler",
        MagicMock(return_value=MagicMock()),
    )
    return env


@pytest.mark.asyncio
async def test_twitter_empty_initial_posts_does_not_raise(tmp_path, monkeypatch):
    cfg = _write_runner_config(
        tmp_path / "simulation_config.json",
        hours=0,
        initial_posts=[],
    )
    (tmp_path / "twitter_profiles.csv").write_text("user_id\n", encoding="utf-8")
    runner = TwitterSimulationRunner(cfg, wait_for_commands=False, seed=1)
    runner._create_model = lambda: object()
    env = _patch_runner_env(
        monkeypatch,
        "scripts.run_twitter_simulation",
        "generate_twitter_agent_graph",
    )

    await runner.run()

    env.step.assert_not_called()
    env.close.assert_awaited()


@pytest.mark.asyncio
async def test_reddit_empty_initial_posts_does_not_raise(tmp_path, monkeypatch):
    cfg = _write_runner_config(
        tmp_path / "simulation_config.json",
        hours=0,
        initial_posts=[],
    )
    (tmp_path / "reddit_profiles.json").write_text("[]", encoding="utf-8")
    runner = RedditSimulationRunner(cfg, wait_for_commands=False, seed=1)
    runner._create_model = lambda: object()
    env = _patch_runner_env(
        monkeypatch,
        "scripts.run_reddit_simulation",
        "generate_reddit_agent_graph",
    )

    await runner.run()

    env.step.assert_not_called()
    env.close.assert_awaited()
