"""Tasks survive a backend restart (#68). No model service needed."""

import logging
import sqlite3
from datetime import datetime, timedelta

import pytest

from app.models.task import INTERRUPTED_ERROR, TaskManager, TaskStatus


def _restart(manager: TaskManager) -> None:
    """What a new backend process sees: nothing in memory, the database on disk."""

    with manager._task_lock:
        manager._tasks.clear()
        manager._paused_until = 0.0
        if manager._db is not None:
            manager._db.close()
        manager._db = None


@pytest.fixture
def manager():
    manager = TaskManager()
    _restart(manager)  # the database of this test (RUNS_DB_PATH, conftest)
    yield manager
    _restart(manager)


def test_a_task_is_still_there_after_a_restart(manager):
    task_id = manager.create_task("graph_build", metadata={"project_id": "proj_1"})
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=40, message="建構中",
                        progress_detail={"chunks": [4, 10]})
    manager.complete_task(task_id, {"graph_id": "g1", "nodes": 12})

    _restart(manager)
    task = manager.get_task(task_id)
    assert task.status == TaskStatus.COMPLETED and task.progress == 100
    assert task.result == {"graph_id": "g1", "nodes": 12}
    assert task.metadata == {"project_id": "proj_1"} and task.progress_detail == {"chunks": [4, 10]}
    assert [t["task_id"] for t in manager.list_tasks("graph_build")] == [task_id]


def test_tasks_going_at_a_restart_are_failed_and_old_ones_dropped(manager):
    going = manager.create_task("graph_build")
    manager.update_task(going, status=TaskStatus.PROCESSING, progress=30)
    waiting = manager.create_task("report_generate")
    done = manager.create_task("simulation_prepare")
    manager.complete_task(done, {})
    old = manager.create_task("simulation_prepare")
    manager.fail_task(old, "boom")
    long_going = manager.create_task("graph_build")  # started long ago, still going at the restart
    with manager._task_lock:  # a day and more ago
        for task_id in (old, long_going):
            task = manager._tasks[task_id]
            task.created_at = task.updated_at = datetime.now() - timedelta(hours=25)
            manager._save(task)

    _restart(manager)
    manager.recover_at_startup(logging.getLogger("test"))
    for task_id in (going, waiting):
        task = manager.get_task(task_id)
        assert task.status == TaskStatus.FAILED and task.error == INTERRUPTED_ERROR
    assert manager.get_task(going).progress == 30  # where it stopped
    assert manager.get_task(done).status == TaskStatus.COMPLETED
    assert manager.get_task(old) is None
    assert manager.get_task(long_going).error == INTERRUPTED_ERROR  # reported, not dropped in the same pass


def test_after_a_restart_the_api_reports_the_task_failed_not_missing(manager):
    from app import create_app

    task_id = manager.create_task("graph_build")
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=50)
    _restart(manager)

    client = create_app().test_client()  # start-up recovers the tasks
    response = client.get(f"/api/graph/task/{task_id}")
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert (data["status"], data["error"], data["progress"]) == ("failed", INTERRUPTED_ERROR, 50)
    assert client.get("/api/graph/task/no-such-task").status_code == 404


def test_start_up_does_not_create_the_database(manager, tmp_path, monkeypatch):
    path = tmp_path / "fresh" / "runs.sqlite"
    monkeypatch.setenv("RUNS_DB_PATH", str(path))
    manager.recover_at_startup(logging.getLogger("test"))
    assert not path.exists()


@pytest.mark.parametrize("error", [sqlite3.OperationalError("database is locked"),
                                   TypeError("keys must be str"), UnicodeEncodeError("utf-8", "", 0, 1, "surrogate")])
def test_a_save_error_does_not_stop_the_task_and_pauses_saving(manager, monkeypatch, error):
    task_id = manager.create_task("graph_build")
    calls = []

    def broken(task):
        calls.append(task.progress)
        raise error

    monkeypatch.setattr(manager._store(), "save", broken)
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=70)
    manager.update_task(task_id, progress=80)
    manager.create_task("report_generate")
    assert manager.get_task(task_id).progress == 80  # still tracked in memory
    assert calls == [70]  # one failed save, then a pause instead of one stall per update


def test_a_going_task_found_only_in_the_database_is_reported_interrupted(manager):
    """Even if start-up recovery could not run: a going task that this
    process does not hold in memory has no thread."""

    task_id = manager.create_task("graph_build")
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=40)
    _restart(manager)  # no recover_at_startup
    task = manager.get_task(task_id)
    assert (task.status, task.error, task.progress) == (TaskStatus.FAILED, INTERRUPTED_ERROR, 40)
    assert manager.list_tasks()[0]["status"] == "failed"


def test_start_up_recovery_leaves_this_process_own_tasks_alone(manager):
    task_id = manager.create_task("graph_build")
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=10)
    manager.recover_at_startup(logging.getLogger("test"))  # e.g. a second create_app
    manager.update_task(task_id, progress=60)
    task = manager.get_task(task_id)
    assert (task.status, task.progress) == (TaskStatus.PROCESSING, 60)
