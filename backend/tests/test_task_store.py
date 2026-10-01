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
    with manager._task_lock:  # a day and more ago
        task = manager._tasks[old]
        task.created_at = datetime.now() - timedelta(hours=25)
        manager._save(task)

    _restart(manager)
    manager.recover_at_startup(logging.getLogger("test"))
    for task_id in (going, waiting):
        task = manager.get_task(task_id)
        assert task.status == TaskStatus.FAILED and task.error == INTERRUPTED_ERROR
    assert manager.get_task(going).progress == 30  # where it stopped
    assert manager.get_task(done).status == TaskStatus.COMPLETED
    assert manager.get_task(old) is None


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


def test_a_database_error_does_not_stop_the_task(manager, monkeypatch, caplog):
    task_id = manager.create_task("graph_build")

    def locked(task):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(manager._store(), "save", locked)
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=70)
    assert manager.get_task(task_id).progress == 70  # still tracked in memory
