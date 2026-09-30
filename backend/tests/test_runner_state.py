"""Run state persistence and completion detection of the runner (#68)."""

import pytest

from app.services.simulation_runner import RunnerStatus, SimulationRunner, SimulationRunState


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    yield tmp_path
    SimulationRunner._run_states.pop("sim-state", None)


def test_run_state_survives_a_restart(runs):
    state = SimulationRunState(simulation_id="sim-state", runner_status=RunnerStatus.RUNNING,
                               current_round=7, total_rounds=24)
    state.twitter_completed = True
    state.error = "模型服務無回應"
    SimulationRunner._save_run_state(state)
    SimulationRunner._run_states.pop("sim-state")  # a new backend process

    loaded = SimulationRunner.get_run_state("sim-state")
    assert loaded.runner_status == RunnerStatus.RUNNING
    assert (loaded.current_round, loaded.total_rounds) == (7, 24)
    assert loaded.twitter_completed is True and loaded.error == "模型服務無回應"
    assert SimulationRunner.get_run_state("sim-state") is loaded  # cached after the first read


def test_a_missing_or_broken_state_file_reads_as_none(runs):
    assert SimulationRunner.get_run_state("sim-state") is None
    (runs / "sim-state").mkdir()
    (runs / "sim-state" / "run_state.json").write_text("{torn", encoding="utf-8")
    assert SimulationRunner._load_run_state("sim-state") is None


@pytest.mark.parametrize(
    "logs, twitter_done, reddit_done, complete",
    [
        ((), False, False, False),  # nothing started
        (("twitter",), False, False, False),
        (("twitter",), True, False, True),  # twitter only, done
        (("twitter", "reddit"), True, False, False),  # reddit still running
        (("twitter", "reddit"), True, True, True),
    ],
)
def test_a_run_completes_when_every_platform_it_started_is_done(runs, logs, twitter_done, reddit_done, complete):
    for platform in logs:
        (runs / "sim-state" / platform).mkdir(parents=True)
        (runs / "sim-state" / platform / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    state = SimulationRunState(simulation_id="sim-state")
    state.twitter_completed, state.reddit_completed = twitter_done, reddit_done
    assert SimulationRunner._check_all_platforms_completed(state) is complete
