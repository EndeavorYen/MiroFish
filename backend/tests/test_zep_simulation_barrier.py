from types import SimpleNamespace
import json

import pytest
from flask import Flask

from app.api import simulation as simulation_api
from app.services import simulation_runner as runner_module
from app.services.simulation_manager import SimulationStatus
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunState,
    SimulationRunner,
    SimulationStopPending,
)


def test_manual_stop_surfaces_graph_ingestion_failure(monkeypatch):
    state = SimulationRunState(
        simulation_id="sim-1",
        runner_status=RunnerStatus.RUNNING,
    )
    saved = []
    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        classmethod(lambda _cls, _simulation_id: state),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_save_run_state",
        classmethod(lambda _cls, value: saved.append(value.runner_status)),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "stop_updater",
        classmethod(
            lambda _cls, _simulation_id: (_ for _ in ()).throw(
                RuntimeError("ingestion incomplete")
            )
        ),
    )
    SimulationRunner._processes.pop("sim-1", None)
    SimulationRunner._graph_memory_enabled["sim-1"] = True

    try:
        with pytest.raises(RuntimeError, match="ingestion incomplete"):
            SimulationRunner.stop_simulation("sim-1")

        assert state.runner_status == RunnerStatus.FAILED
        assert "ingestion incomplete" in state.error
        assert saved[-1] == RunnerStatus.FAILED
    finally:
        SimulationRunner._graph_memory_enabled.pop("sim-1", None)
        SimulationRunner._manual_stop_requests.discard("sim-1")


def test_platform_completion_does_not_publish_terminal_success_before_barrier(
    monkeypatch, tmp_path
):
    simulation_id = "sim-1"
    sim_dir = tmp_path / simulation_id / "twitter"
    sim_dir.mkdir(parents=True)
    log_path = sim_dir / "actions.jsonl"
    log_path.write_text(
        '{"event_type":"simulation_end","total_rounds":1,"total_actions":0}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.RUNNING,
        twitter_running=True,
    )

    SimulationRunner._read_action_log(str(log_path), 0, state, "twitter")

    assert state.twitter_completed is True
    assert state.runner_status == RunnerStatus.RUNNING


def test_manual_stop_timeout_leaves_monitor_owned_state_stopping(monkeypatch):
    state = SimulationRunState(
        simulation_id="sim-timeout",
        runner_status=RunnerStatus.RUNNING,
    )

    class Monitor:
        def join(self, timeout):
            assert timeout >= 30

        def is_alive(self):
            return True

    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        classmethod(lambda _cls, _simulation_id: state),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_save_run_state",
        classmethod(lambda _cls, _state: None),
    )
    SimulationRunner._monitor_threads["sim-timeout"] = Monitor()
    SimulationRunner._processes.pop("sim-timeout", None)
    SimulationRunner._graph_memory_enabled.pop("sim-timeout", None)

    try:
        with pytest.raises(TimeoutError, match="仍在停止中"):
            SimulationRunner.stop_simulation("sim-timeout")
        assert state.runner_status == RunnerStatus.STOPPING
    finally:
        SimulationRunner._monitor_threads.pop("sim-timeout", None)
        SimulationRunner._manual_stop_requests.discard("sim-timeout")


def test_failed_ingestion_finalization_can_be_retried(monkeypatch):
    state = SimulationRunState(
        simulation_id="sim-retry",
        runner_status=RunnerStatus.FAILED,
        error="first drain timed out",
    )
    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        classmethod(lambda _cls, _simulation_id: state),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_save_run_state",
        classmethod(lambda _cls, _state: None),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "get_updater",
        classmethod(lambda _cls, _simulation_id: object()),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "stop_updater",
        classmethod(lambda _cls, _simulation_id: None),
    )
    SimulationRunner._graph_memory_enabled["sim-retry"] = True
    SimulationRunner._monitor_threads.pop("sim-retry", None)

    try:
        result = SimulationRunner.stop_simulation("sim-retry")
        assert result.runner_status == RunnerStatus.STOPPED
        assert result.error is None
    finally:
        SimulationRunner._graph_memory_enabled.pop("sim-retry", None)
        SimulationRunner._manual_stop_requests.discard("sim-retry")


def test_stop_api_keeps_pending_finalization_out_of_failed_state(monkeypatch):
    simulation = SimpleNamespace(status=SimulationStatus.STOPPING, error=None)
    saved = []
    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "stop_simulation",
        classmethod(
            lambda _cls, _simulation_id: (_ for _ in ()).throw(
                SimulationStopPending("still draining")
            )
        ),
    )
    monkeypatch.setattr(
        simulation_api.run,
        "SimulationManager",
        lambda: SimpleNamespace(
            get_simulation=lambda _simulation_id: simulation,
            _save_simulation_state=lambda state: saved.append(state.status),
        ),
    )

    app = Flask(__name__)
    with app.test_request_context(
        "/api/simulation/stop",
        method="POST",
        json={"simulation_id": "sim-pending"},
    ):
        response, status = simulation_api.stop_simulation()

    assert status == 202
    assert response.get_json()["pending"] is True
    assert simulation.status == SimulationStatus.STOPPING
    assert saved == []


@pytest.mark.parametrize(
    "field",
    ["force", "enable_graph_memory_update"],
)
def test_simulation_start_rejects_string_booleans(field):
    app = Flask(__name__)
    with app.test_request_context(
        "/api/simulation/start",
        method="POST",
        json={"simulation_id": "sim-1", field: "false"},
    ):
        response, status = simulation_api.start_simulation()

    assert status == 400
    assert "JSON boolean" in response.get_json()["error"]


def test_force_restart_does_not_continue_while_old_ingestion_is_pending(monkeypatch):
    simulation = SimpleNamespace(
        simulation_id="sim-1",
        project_id="proj-1",
        graph_id="graph-1",
        status=SimulationStatus.STOPPING,
    )
    cleanup_called = []
    monkeypatch.setattr(
        simulation_api.run,
        "SimulationManager",
        lambda: SimpleNamespace(
            get_simulation=lambda _simulation_id: simulation,
            _save_simulation_state=lambda _state: None,
        ),
    )
    monkeypatch.setattr(
        simulation_api.run,
        "_check_simulation_prepared",
        lambda _simulation_id: (True, {}),
    )
    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "get_run_state",
        classmethod(
            lambda _cls, _simulation_id: SimpleNamespace(
                runner_status=RunnerStatus.STOPPING
            )
        ),
    )
    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "stop_simulation",
        classmethod(
            lambda _cls, _simulation_id: (_ for _ in ()).throw(
                SimulationStopPending("still draining")
            )
        ),
    )
    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "cleanup_simulation_logs",
        classmethod(
            lambda _cls, _simulation_id: cleanup_called.append(True)
        ),
    )
    monkeypatch.setattr(
        simulation_api.GraphMemoryManager,
        "get_updater",
        classmethod(lambda _cls, _simulation_id: object()),
    )

    app = Flask(__name__)
    with app.test_request_context(
        "/api/simulation/start",
        method="POST",
        json={"simulation_id": "sim-1", "force": True},
    ):
        response, status = simulation_api.start_simulation()

    assert status == 409
    assert response.get_json()["pending"] is True
    assert cleanup_called == []


def test_monitor_start_failure_terminates_the_spawned_process(monkeypatch, tmp_path):
    simulation_id = "sim-start-failure"
    sim_dir = tmp_path / "runs" / simulation_id
    scripts_dir = tmp_path / "scripts"
    sim_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    (sim_dir / "simulation_config.json").write_text(
        json.dumps({
            "time_config": {
                "total_simulation_hours": 1,
                "minutes_per_round": 60,
            }
        }),
        encoding="utf-8",
    )
    (scripts_dir / "run_parallel_simulation.py").write_text("pass\n", encoding="utf-8")

    class Process:
        pid = 123

        def poll(self):
            return None

    class BrokenThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("monitor failed")

    terminated = []
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(SimulationRunner, "SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(runner_module.threading, "Thread", BrokenThread)
    monkeypatch.setattr(
        SimulationRunner,
        "_terminate_process",
        classmethod(lambda _cls, _process, sim_id: terminated.append(sim_id)),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _cls, *_args, **_kwargs: None),
    )

    try:
        with pytest.raises(RuntimeError, match="monitor failed"):
            SimulationRunner.start_simulation(
                simulation_id,
                platform="twitter",
                enable_graph_memory_update=False,
            )
        assert terminated == [simulation_id]
        assert simulation_id not in SimulationRunner._processes
        assert simulation_id not in SimulationRunner._action_queues
        assert simulation_id not in SimulationRunner._stdout_files
    finally:
        SimulationRunner._run_states.pop(simulation_id, None)
        SimulationRunner._processes.pop(simulation_id, None)
        SimulationRunner._action_queues.pop(simulation_id, None)
        SimulationRunner._stdout_files.pop(simulation_id, None)
        SimulationRunner._stderr_files.pop(simulation_id, None)
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)


def test_shutdown_terminates_producer_before_tail_read_and_updater_drain(
    monkeypatch,
):
    simulation_id = "sim-shutdown-order"
    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.RUNNING,
    )
    events = []

    class Process:
        pid = 123
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

    process = Process()

    class Monitor:
        alive = True

        def join(self, timeout):
            assert timeout >= 30
            events.extend(["tail-read", "updater-drain"])
            state.runner_status = RunnerStatus.STOPPED
            SimulationRunner._graph_memory_enabled.pop(simulation_id, None)
            self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        classmethod(lambda _cls, _simulation_id: state),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_save_run_state",
        classmethod(lambda _cls, _state: None),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _cls, *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_terminate_process",
        classmethod(
            lambda _cls, proc, _simulation_id, **_kwargs: (
                events.append("producer-terminate"),
                setattr(proc, "stopped", True),
            )
        ),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "get_simulation_ids",
        classmethod(lambda _cls: [simulation_id]),
    )
    updater = object()
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "get_updater",
        classmethod(lambda _cls, _simulation_id: updater),
    )

    SimulationRunner._cleanup_done = False
    SimulationRunner._processes[simulation_id] = process
    SimulationRunner._monitor_threads[simulation_id] = Monitor()
    SimulationRunner._graph_memory_enabled[simulation_id] = True
    try:
        SimulationRunner.cleanup_all_simulations()
        assert events == [
            "producer-terminate",
            "tail-read",
            "updater-drain",
        ]
        assert state.runner_status == RunnerStatus.STOPPED
    finally:
        SimulationRunner._cleanup_done = False
        SimulationRunner._processes.pop(simulation_id, None)
        SimulationRunner._monitor_threads.pop(simulation_id, None)
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)
        SimulationRunner._manual_stop_requests.discard(simulation_id)


def test_shutdown_drain_failure_remains_failed_and_retryable(monkeypatch):
    simulation_id = "sim-shutdown-failure"
    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.RUNNING,
    )
    updater = object()

    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        classmethod(lambda _cls, _simulation_id: state),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_save_run_state",
        classmethod(lambda _cls, _state: None),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _cls, *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "get_simulation_ids",
        classmethod(lambda _cls: [simulation_id]),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "get_updater",
        classmethod(lambda _cls, _simulation_id: updater),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager,
        "stop_updater",
        classmethod(
            lambda _cls, _simulation_id: (_ for _ in ()).throw(
                RuntimeError("drain incomplete")
            )
        ),
    )

    SimulationRunner._cleanup_done = False
    SimulationRunner._graph_memory_enabled[simulation_id] = True
    SimulationRunner._monitor_threads.pop(simulation_id, None)
    SimulationRunner._processes.pop(simulation_id, None)
    try:
        SimulationRunner.cleanup_all_simulations()
        assert state.runner_status == RunnerStatus.FAILED
        assert "drain incomplete" in state.error
        assert SimulationRunner._graph_memory_enabled[simulation_id] is True
        assert SimulationRunner._cleanup_done is False
    finally:
        SimulationRunner._cleanup_done = False
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)
        SimulationRunner._manual_stop_requests.discard(simulation_id)


def test_completion_is_published_after_drain_while_the_env_stays_up(monkeypatch, tmp_path):
    """#49: the script keeps its environment for interviews after both
    platforms end, so the process does not exit. COMPLETED must still be
    published, after the graph writes drain, and a later exit (close_env,
    stop) must not turn it into FAILED."""

    simulation_id = "sim-waiting"
    for platform in ("twitter", "reddit"):
        path = tmp_path / simulation_id / platform
        path.mkdir(parents=True)
        (path / "actions.jsonl").write_text(
            '{"event_type":"simulation_end","total_rounds":1,"total_actions":0}\n', encoding="utf-8"
        )
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _s: None)
    state = SimulationRunState(simulation_id=simulation_id, runner_status=RunnerStatus.RUNNING,
                               twitter_running=True, reddit_running=True)
    events = []
    published = []

    class Process:
        polls = 0
        returncode = 1  # killed by close_env later

        def poll(self):
            self.polls += 1
            if (published and self.polls > 5) or self.polls > 50:  # never hang the test
                return 1
            return None

    monkeypatch.setattr(SimulationRunner, "get_run_state", classmethod(lambda _c, _s: state))
    monkeypatch.setattr(SimulationRunner, "_save_run_state", classmethod(lambda _c, _s: None))
    monkeypatch.setattr(
        SimulationRunner, "_sync_simulation_status",
        classmethod(lambda _c, _sid, status, *_a, **_k: published.append((status, process.polls))),
    )
    monkeypatch.setattr(
        runner_module.GraphMemoryManager, "stop_updater",
        classmethod(lambda _c, _sid: events.append(("drain", len(published)))),
    )
    process = Process()
    SimulationRunner._processes[simulation_id] = process
    SimulationRunner._graph_memory_enabled[simulation_id] = True
    try:
        SimulationRunner._monitor_simulation(simulation_id)
    finally:
        SimulationRunner._processes.pop(simulation_id, None)
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)

    completed = [p for p in published if p[0] == RunnerStatus.COMPLETED]
    assert completed, published
    assert events and events[0][0] == "drain"
    # Drained before COMPLETED, and COMPLETED while the process still ran.
    assert published.index(completed[0]) >= events[0][1]
    assert completed[0][1] <= 5
    assert state.runner_status == RunnerStatus.COMPLETED
    assert RunnerStatus.FAILED not in [p[0] for p in published]


class _LiveEnv:
    """An interview environment still running after COMPLETED."""

    pid = 4242

    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0


def _completed(monkeypatch, simulation_id):
    state = SimulationRunState(simulation_id=simulation_id, runner_status=RunnerStatus.COMPLETED)
    monkeypatch.setattr(SimulationRunner, "get_run_state", classmethod(lambda _c, _s: state))
    monkeypatch.setattr(SimulationRunner, "_save_run_state", classmethod(lambda _c, _s: None))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", classmethod(lambda *_a, **_k: None))
    ended = []
    monkeypatch.setattr(
        SimulationRunner, "_terminate_process",
        classmethod(lambda _c, proc, sid, **_k: (ended.append(sid), setattr(proc, "alive", False))),
    )
    return state, ended


def test_stop_after_completion_ends_the_environment(monkeypatch):
    state, ended = _completed(monkeypatch, "sim-stop-done")
    SimulationRunner._processes["sim-stop-done"] = _LiveEnv()
    try:
        result = SimulationRunner.stop_simulation("sim-stop-done")
    finally:
        SimulationRunner._processes.pop("sim-stop-done", None)
    assert ended == ["sim-stop-done"]
    assert result.runner_status == RunnerStatus.COMPLETED


def test_shutdown_ends_a_completed_runs_environment(monkeypatch):
    state, ended = _completed(monkeypatch, "sim-shutdown-done")
    monkeypatch.setattr(runner_module.GraphMemoryManager, "get_simulation_ids", classmethod(lambda _c: []))
    monkeypatch.setattr(runner_module.GraphMemoryManager, "get_updater", classmethod(lambda _c, _s: None))
    monkeypatch.setattr(
        SimulationRunner, "stop_simulation",
        classmethod(lambda _c, _s: (_ for _ in ()).throw(AssertionError("stop must not run"))),
    )
    SimulationRunner._cleanup_done = False
    SimulationRunner._processes["sim-shutdown-done"] = _LiveEnv()
    try:
        SimulationRunner.cleanup_all_simulations()
    finally:
        SimulationRunner._cleanup_done = False
        SimulationRunner._processes.pop("sim-shutdown-done", None)
    assert ended == ["sim-shutdown-done"]


def test_restart_ends_the_previous_environment_first(monkeypatch, tmp_path):
    state, ended = _completed(monkeypatch, "sim-restart")
    (tmp_path / "sim-restart").mkdir()
    (tmp_path / "sim-restart" / "simulation_config.json").write_text(
        json.dumps({"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}), encoding="utf-8"
    )
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    SimulationRunner._processes["sim-restart"] = _LiveEnv()

    class Stop(Exception):
        pass

    # Stop right after the claim so no real process is started.
    monkeypatch.setattr(
        SimulationRunner, "_finalization_lock",
        classmethod(lambda _c, _s: (_ for _ in ()).throw(Stop())),
    )
    try:
        with pytest.raises(Stop):
            SimulationRunner.start_simulation("sim-restart", platform="parallel")
    finally:
        SimulationRunner._processes.pop("sim-restart", None)
    assert ended == ["sim-restart"]


def test_old_monitor_leaves_a_newer_run_alone(monkeypatch, tmp_path):
    simulation_id = "sim-owner"
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _s: None)
    state = SimulationRunState(simulation_id=simulation_id, runner_status=RunnerStatus.COMPLETED)
    saved = []
    monkeypatch.setattr(SimulationRunner, "get_run_state", classmethod(lambda _c, _s: state))
    monkeypatch.setattr(SimulationRunner, "_save_run_state", classmethod(lambda _c, s: saved.append(s)))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", classmethod(lambda *_a, **_k: None))
    old, new = _LiveEnv(), _LiveEnv()
    SimulationRunner._processes[simulation_id] = old

    real_poll = old.poll

    def poll():
        SimulationRunner._processes[simulation_id] = new  # a restart replaces it
        return real_poll()

    old.poll = poll
    try:
        SimulationRunner._monitor_simulation(simulation_id)
        assert SimulationRunner._processes.get(simulation_id) is new
    finally:
        SimulationRunner._processes.pop(simulation_id, None)


def _race_to_completion(monkeypatch, simulation_id, action):
    """A thread holds the finalization lock with the run STOPPING (the drain),
    then publishes COMPLETED and releases. ``action`` runs meanwhile and first
    sees STOPPING, then COMPLETED once it has the lock (PR #55 review)."""

    import threading

    state = SimulationRunState(simulation_id=simulation_id, runner_status=RunnerStatus.STOPPING)
    monkeypatch.setattr(SimulationRunner, "get_run_state", classmethod(lambda _c, _s: state))
    monkeypatch.setattr(SimulationRunner, "_save_run_state", classmethod(lambda _c, _s: None))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", classmethod(lambda *_a, **_k: None))
    ended = []
    monkeypatch.setattr(
        SimulationRunner, "_terminate_process",
        classmethod(lambda _c, proc, sid, **_k: (ended.append(sid), setattr(proc, "alive", False))),
    )
    process = _LiveEnv()
    SimulationRunner._processes[simulation_id] = process
    lock = SimulationRunner._finalization_lock(simulation_id)
    lock.acquire()
    outcome = {}

    def run():
        try:
            outcome["result"] = action()
        except Exception as error:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = error

    worker = threading.Thread(target=run)
    try:
        worker.start()
        worker.join(0.3)  # the action now waits on the lock
        state.runner_status = RunnerStatus.COMPLETED
    finally:
        lock.release()
    worker.join(10)
    SimulationRunner._processes.pop(simulation_id, None)
    return outcome, ended, process


def test_stop_racing_the_early_completion_ends_the_environment(monkeypatch):
    outcome, ended, process = _race_to_completion(
        monkeypatch, "sim-race-stop", lambda: SimulationRunner.stop_simulation("sim-race-stop")
    )
    assert "error" not in outcome, outcome
    assert outcome["result"].runner_status == RunnerStatus.COMPLETED
    assert ended == ["sim-race-stop"] and not process.alive


def test_shutdown_racing_the_early_completion_leaves_no_orphan(monkeypatch):
    monkeypatch.setattr(runner_module.GraphMemoryManager, "get_simulation_ids", classmethod(lambda _c: []))
    monkeypatch.setattr(runner_module.GraphMemoryManager, "get_updater", classmethod(lambda _c, _s: None))
    SimulationRunner._cleanup_done = False
    try:
        outcome, ended, process = _race_to_completion(
            monkeypatch, "sim-race-shutdown", SimulationRunner.cleanup_all_simulations
        )
        assert "error" not in outcome, outcome
        assert ended == ["sim-race-shutdown"] and not process.alive
        assert SimulationRunner._cleanup_done is True
    finally:
        SimulationRunner._cleanup_done = False


@pytest.mark.parametrize("platform, flag", [("twitter", "--twitter-only"), ("reddit", "--reddit-only"), ("parallel", None)])
def test_one_platform_runs_the_parallel_script_with_its_flag(monkeypatch, tmp_path, platform, flag):
    """#68: the single-platform scripts are gone; one entry point for all."""

    simulation_id = f"sim-one-{platform}"
    sim_dir = tmp_path / "runs" / simulation_id
    scripts_dir = tmp_path / "scripts"
    sim_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    (sim_dir / "simulation_config.json").write_text(
        json.dumps({"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}), encoding="utf-8"
    )
    (scripts_dir / "run_parallel_simulation.py").write_text("pass\n", encoding="utf-8")
    commands = []

    class Process:
        pid = 123

        def poll(self):
            return None

    class IdleThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(SimulationRunner, "SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda cmd, **_kw: commands.append(cmd) or Process())
    monkeypatch.setattr(runner_module.threading, "Thread", IdleThread)
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", classmethod(lambda _cls, *_a, **_k: None))
    try:
        state = SimulationRunner.start_simulation(simulation_id, platform=platform)
        cmd = commands[0]
        assert cmd[1].endswith("run_parallel_simulation.py")
        assert (flag in cmd) if flag else not {"--twitter-only", "--reddit-only"} & set(cmd)
        assert state.twitter_running == (platform != "reddit")
        assert state.reddit_running == (platform != "twitter")
    finally:
        for registry in (SimulationRunner._processes, SimulationRunner._action_queues,
                         SimulationRunner._stdout_files, SimulationRunner._run_states,
                         SimulationRunner._graph_memory_enabled):
            registry.pop(simulation_id, None)
