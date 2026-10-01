"""
任务状态管理
用于跟踪长时间运行的任务（如图谱构建）

Tasks are kept in memory and written to the same SQLite database as the runs
(#68, #63), so a status query still answers after a backend restart. At
start-up, tasks that were going lost their thread and are marked failed.
"""

import json
import os
import sqlite3
import time
import uuid
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..utils.locale import t


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待中
    PROCESSING = "processing"    # 处理中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败


@dataclass
class Task:
    """任务数据类"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    progress: int = 0              # 总进度百分比 0-100
    message: str = ""              # 状态消息
    result: Optional[Dict] = None  # 任务结果
    error: Optional[str] = None    # 错误信息
    metadata: Dict = field(default_factory=dict)  # 额外元数据
    progress_detail: Dict = field(default_factory=dict)  # 详细进度信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "message": self.message,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    result TEXT,
    error TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    progress_detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS tasks_created ON tasks(created_at);
"""
_COLUMNS = "task_id, task_type, status, created_at, updated_at, progress, message, result, error, metadata, progress_detail"
INTERRUPTED_ERROR = "backend restarted"
BUSY_TIMEOUT_MS = 2000  # a locked database must not hold every task update for long
PAUSE_AFTER_ERROR_S = 30.0
CLEANUP_EVERY_S = 3600.0
GOING = (TaskStatus.PENDING, TaskStatus.PROCESSING)


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class TaskStore:
    """The ``tasks`` table. One connection, used under the manager's lock."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # progress is written often
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.executescript(_SCHEMA)

    def save(self, task: "Task") -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO tasks ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task.task_id, task.task_type, task.status.value, task.created_at.isoformat(),
             task.updated_at.isoformat(), task.progress, task.message,
             None if task.result is None else _dumps(task.result), task.error,
             _dumps(task.metadata), _dumps(task.progress_detail)),
        )

    @staticmethod
    def _task(row) -> "Task":
        return Task(
            task_id=row[0], task_type=row[1], status=TaskStatus(row[2]),
            created_at=datetime.fromisoformat(row[3]), updated_at=datetime.fromisoformat(row[4]),
            progress=row[5], message=row[6], result=None if row[7] is None else json.loads(row[7]),
            error=row[8], metadata=json.loads(row[9]), progress_detail=json.loads(row[10]),
        )

    def get(self, task_id: str) -> Optional["Task"]:
        row = self._conn.execute(f"SELECT {_COLUMNS} FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def all(self) -> list:
        return [self._task(r) for r in self._conn.execute(f"SELECT {_COLUMNS} FROM tasks")]

    def mark_interrupted(self, message: str, keep: set) -> list:
        """Going tasks that are not ``keep`` (this process's own) lost their thread."""

        now = datetime.now().isoformat()
        ids = [r[0] for r in self._conn.execute(
            "SELECT task_id FROM tasks WHERE status IN (?, ?)",
            (TaskStatus.PENDING.value, TaskStatus.PROCESSING.value)) if r[0] not in keep]
        self._conn.executemany(
            "UPDATE tasks SET status = ?, message = ?, error = ?, updated_at = ? WHERE task_id = ?",
            [(TaskStatus.FAILED.value, message, INTERRUPTED_ERROR, now, i) for i in ids])
        return ids

    def delete_finished_before(self, cutoff: datetime) -> int:
        return self._conn.execute(
            "DELETE FROM tasks WHERE updated_at < ? AND status IN (?, ?)",
            (cutoff.isoformat(), TaskStatus.COMPLETED.value, TaskStatus.FAILED.value)).rowcount

    def close(self) -> None:
        self._conn.close()


def _db_path() -> str:
    from ..runs.store import default_db_path  # one database for runs and tasks

    return os.path.abspath(default_db_path())


class TaskManager:
    """
    任务管理器
    线程安全的任务状态管理
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, Task] = {}
                    cls._instance._task_lock = threading.Lock()
                    cls._instance._db = None
                    cls._instance._paused_until = 0.0
                    cls._instance._last_cleanup = time.monotonic()
        return cls._instance

    def _store(self) -> TaskStore:
        """Under ``_task_lock``. Reopened when the database path changes (tests)."""

        path = _db_path()
        if self._db is None or self._db.path != path:
            if self._db is not None:
                self._db.close()
            self._db = TaskStore(path)
        return self._db

    def _save(self, task: Task) -> None:
        """Under ``_task_lock``. Saving never fails a task: on any error it is
        logged, the task goes on in memory, and saving pauses for a while so a
        locked database does not slow every progress update."""

        if time.monotonic() < self._paused_until:
            return
        try:
            self._store().save(task)
        except Exception as error:
            self._paused_until = time.monotonic() + PAUSE_AFTER_ERROR_S
            from ..utils.logger import get_logger
            get_logger('mirofish.task').error(
                f"could not save task {task.task_id} (saving paused {int(PAUSE_AFTER_ERROR_S)} s): {error}")

    def _stored(self, task_id: Optional[str] = None) -> list:
        """Under ``_task_lock``: tasks from the database that are not in memory.
        A going task there is not this process's own (those are in memory),
        so its thread is gone, even if start-up recovery could not run."""

        try:
            store = self._store()
            if task_id is None:
                tasks = store.all()
            else:
                task = store.get(task_id)
                tasks = [task] if task else []
        except Exception:
            return []
        tasks = [task for task in tasks if task.task_id not in self._tasks]
        for task in tasks:
            if task.status in GOING:
                task.status, task.error = TaskStatus.FAILED, INTERRUPTED_ERROR
                task.message = t('progress.taskFailed')
        return tasks
    
    def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        创建新任务
        
        Args:
            task_type: 任务类型
            metadata: 额外元数据
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
            self._save(task)
            if time.monotonic() - self._last_cleanup > CLEANUP_EVERY_S:
                self._cleanup(datetime.now() - timedelta(hours=24))

        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务（内存里没有时读数据库：重启前的任务）"""
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task is None:
                stored = self._stored(task_id)
                task = stored[0] if stored else None
            return task
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        progress_detail: Optional[Dict] = None
    ):
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            progress: 进度
            message: 消息
            result: 结果
            error: 错误信息
            progress_detail: 详细进度信息
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.updated_at = datetime.now()
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message is not None:
                    task.message = message
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if progress_detail is not None:
                    task.progress_detail = progress_detail
                self._save(task)
    
    def complete_task(self, task_id: str, result: Dict):
        """标记任务完成"""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message=t('progress.taskComplete'),
            result=result
        )
    
    def fail_task(self, task_id: str, error: str):
        """标记任务失败"""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=t('progress.taskFailed'),
            error=error
        )
    
    def list_tasks(self, task_type: Optional[str] = None) -> list:
        """列出任务"""
        with self._task_lock:
            tasks = self._stored() + list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)]
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """清理旧任务（最后更新早于 max_age_hours 的已结束任务）"""
        with self._task_lock:
            return self._cleanup(datetime.now() - timedelta(hours=max_age_hours))

    def _cleanup(self, cutoff: datetime) -> int:
        """Under ``_task_lock``. Runs at start-up and then about once an hour."""

        self._last_cleanup = time.monotonic()
        old_ids = [
            tid for tid, task in self._tasks.items()
            if task.updated_at < cutoff and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        ]
        for tid in old_ids:
            del self._tasks[tid]
        try:
            return self._store().delete_finished_before(cutoff)
        except Exception:
            return 0

    def recover_at_startup(self, logger: Any) -> None:
        """Tasks that were going when the backend stopped lost their thread:
        mark them failed, and drop finished tasks older than a day. This
        process's own tasks (in memory, e.g. a second ``create_app``) are left
        alone."""

        if not os.path.exists(_db_path()):
            return  # never used: do not create the database at start-up
        try:
            with self._task_lock:
                ids = self._store().mark_interrupted(t('progress.taskFailed'), keep=set(self._tasks))
                removed = self._cleanup(datetime.now() - timedelta(hours=24))
        except Exception as error:  # a locked or broken database must not stop the backend
            logger.error(f"could not recover tasks from {_db_path()}: {error}")
            return
        if ids:
            logger.warning(f"tasks interrupted by the restart: {', '.join(ids)}")
        if removed:
            logger.info(f"removed {removed} finished tasks older than a day")

