"""
OASIS模拟运行器
在后台运行模拟并记录每个Agent的动作，支持实时状态监控
"""

import os
import sys
import json
import time
import threading
import subprocess
import signal
from typing import Dict, Any, Optional
from datetime import datetime
from queue import Queue

from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale
from ..utils.zep import (
    ZEP_HTTP_REQUEST_TIMEOUT_SECONDS,
    ZEP_INGESTION_WAIT_TIMEOUT_SECONDS,
)
from .graph_memory_updater import GraphMemoryManager

logger = get_logger('mirofish.simulation_runner')

# Split in #68; these names stay importable from here.
from .runner_models import (  # noqa: F401
    IS_WINDOWS,
    RunnerStatus,
    SimulationStopPending,
    AgentAction,
    RoundSummary,
    SimulationRunState,
    _CompletedWhileStopping,
)
from .runner_monitor import MonitorMixin
from .runner_actions import ActionQueryMixin
from .runner_cleanup import CleanupMixin
from .runner_interview import InterviewMixin


class SimulationRunner(MonitorMixin, ActionQueryMixin, CleanupMixin, InterviewMixin):
    """
    模拟运行器
    
    负责：
    1. 在后台进程中运行OASIS模拟
    2. 解析运行日志，记录每个Agent的动作
    3. 提供实时状态查询接口
    4. 支持暂停/停止/恢复操作
    """

    RUN_STATE_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )

    SCRIPTS_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../scripts'
    )

    _run_states: Dict[str, SimulationRunState] = {}

    _processes: Dict[str, subprocess.Popen] = {}

    _action_queues: Dict[str, Queue] = {}

    _monitor_threads: Dict[str, threading.Thread] = {}

    _stdout_files: Dict[str, Any] = {}  # 存储 stdout 文件句柄

    _stderr_files: Dict[str, Any] = {}  # 存储 stderr 文件句柄

    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> enabled

    _finalization_locks: Dict[str, threading.Lock] = {}

    _finalization_locks_guard = threading.Lock()

    _manual_stop_requests: set[str] = set()

    @classmethod
    def _finalization_lock(cls, simulation_id: str) -> threading.Lock:
        with cls._finalization_locks_guard:
            return cls._finalization_locks.setdefault(
                simulation_id, threading.Lock()
            )

    @classmethod
    def _sync_simulation_status(
        cls,
        simulation_id: str,
        runner_status: RunnerStatus,
        error: str | None = None,
    ) -> None:
        """Keep persisted simulation metadata aligned with run_state.json."""

        from .simulation_manager import SimulationManager, SimulationStatus

        status_map = {
            RunnerStatus.RUNNING: SimulationStatus.RUNNING,
            RunnerStatus.STOPPING: SimulationStatus.STOPPING,
            RunnerStatus.STOPPED: SimulationStatus.STOPPED,
            RunnerStatus.COMPLETED: SimulationStatus.COMPLETED,
            RunnerStatus.FAILED: SimulationStatus.FAILED,
        }
        status = status_map.get(runner_status)
        if status is None:
            return
        try:
            manager = SimulationManager()
            simulation = manager.get_simulation(simulation_id)
            if simulation is None:
                return
            simulation.status = status
            simulation.error = error
            manager._save_simulation_state(simulation)
        except Exception as sync_error:
            # state.json is a secondary projection. Never let a projection
            # failure skip the authoritative run-state finalization or Zep
            # ingestion drain.
            logger.error(
                "同步模拟状态失败: simulation_id=%s, status=%s, error=%s",
                simulation_id,
                runner_status.value,
                sync_error,
            )

    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """获取运行状态"""
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]
        
        # 尝试从文件加载
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state

    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """从文件加载运行状态"""
        state_file = os.path.join(cls.RUN_STATE_DIR, simulation_id, "run_state.json")
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # 各平台独立轮次和时间
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )
            
            # 加载最近动作
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(AgentAction(
                    round_num=a.get("round_num", 0),
                    timestamp=a.get("timestamp", ""),
                    platform=a.get("platform", ""),
                    agent_id=a.get("agent_id", 0),
                    agent_name=a.get("agent_name", ""),
                    action_type=a.get("action_type", ""),
                    action_args=a.get("action_args", {}),
                    result=a.get("result"),
                    success=a.get("success", True),
                ))
            
            return state
        except Exception as e:
            logger.error(f"加载运行状态失败: {str(e)}")
            return None

    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """保存运行状态到文件"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")
        
        data = state.to_detail_dict()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        cls._run_states[state.simulation_id] = state

    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: int = None,  # 最大模拟轮数（可选，用于截断过长的模拟）
        enable_graph_memory_update: bool = False,  # 是否将活动更新到Zep图谱
        graph_id: str = None,  # Zep图谱ID（启用图谱更新时必需）
        seed: Optional[int] = None  # 随机种子（可选，用于复现Agent激活序列）
    ) -> SimulationRunState:
        """
        启动模拟
        
        Args:
            simulation_id: 模拟ID
            platform: 运行平台 (twitter/reddit/parallel)
            max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
            enable_graph_memory_update: 是否将Agent活动动态更新到Zep图谱
            graph_id: Zep图谱ID（启用图谱更新时必需）
            
        Returns:
            SimulationRunState
        """
        # 加载模拟配置
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            raise ValueError(f"模拟配置不存在，请先调用 /prepare 接口")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 初始化运行状态
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)
        
        # 如果指定了最大轮数，则截断
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(f"轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )
        
        previous = cls.get_run_state(simulation_id)
        if previous is not None and previous.runner_status == RunnerStatus.COMPLETED:
            # The last run's interview environment would otherwise share the
            # sim dir and IPC with the new process (#49).
            cls.close_environment(simulation_id)

        # Atomically claim this simulation ID. The expensive updater/process
        # startup happens after releasing the lock, while the persisted
        # STARTING state makes every concurrent start fail closed.
        with cls._finalization_lock(simulation_id):
            existing = cls.get_run_state(simulation_id)
            active_statuses = {
                RunnerStatus.STARTING,
                RunnerStatus.RUNNING,
                RunnerStatus.PAUSED,
                RunnerStatus.STOPPING,
            }
            if (
                existing and existing.runner_status in active_statuses
            ) or GraphMemoryManager.get_updater(simulation_id) is not None:
                raise ValueError(f"模拟已在运行或结束处理中: {simulation_id}")
            cls._save_run_state(state)
        
        # 如果启用图谱记忆更新，创建更新器
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError("启用图谱记忆更新时必须提供 graph_id")
            
            try:
                GraphMemoryManager.create_updater(simulation_id, graph_id)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(f"已启用图谱记忆更新: simulation_id={simulation_id}, graph_id={graph_id}")
            except Exception as e:
                logger.error(f"创建图谱记忆更新器失败: {e}")
                cls._graph_memory_enabled[simulation_id] = False
                state.runner_status = RunnerStatus.FAILED
                state.error = f"Zep图谱更新器初始化失败: {e}"
                with cls._finalization_lock(simulation_id):
                    cls._save_run_state(state)
                    cls._sync_simulation_status(
                        simulation_id,
                        RunnerStatus.FAILED,
                        state.error,
                    )
                raise RuntimeError(state.error) from e
        else:
            cls._graph_memory_enabled[simulation_id] = False
        
        # One script for every platform choice (#68): a single platform is
        # run_parallel_simulation.py --twitter-only / --reddit-only.
        script_name = "run_parallel_simulation.py"
        platform_flags = {"twitter": ["--twitter-only"], "reddit": ["--reddit-only"]}.get(platform, [])
        state.twitter_running = platform != "reddit"
        state.reddit_running = platform != "twitter"
        
        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)
        
        if not os.path.exists(script_path):
            cleanup_error = None
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    GraphMemoryManager.stop_updater(simulation_id)
                    cls._graph_memory_enabled.pop(simulation_id, None)
                except Exception as error:
                    cleanup_error = error
            state.runner_status = RunnerStatus.FAILED
            state.twitter_running = False
            state.reddit_running = False
            state.error = f"脚本不存在: {script_path}"
            if cleanup_error is not None:
                state.error += f"; Zep图谱写入清理失败: {cleanup_error}"
            with cls._finalization_lock(simulation_id):
                cls._save_run_state(state)
                cls._sync_simulation_status(
                    simulation_id,
                    RunnerStatus.FAILED,
                    state.error,
                )
            raise ValueError(state.error)
        
        # 创建动作队列
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue

        process = None
        main_log_file = None

        # 启动模拟进程
        try:
            # 构建运行命令，使用完整路径
            # 新的日志结构：
            #   twitter/actions.jsonl - Twitter 动作日志
            #   reddit/actions.jsonl  - Reddit 动作日志
            #   simulation.log        - 主进程日志
            
            cmd = [
                sys.executable,  # Python解释器
                script_path,
                "--config", config_path,  # 使用完整配置文件路径
                *platform_flags,
            ]
            
            # 如果指定了最大轮数，添加到命令行参数
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])
            
            # 如果指定了随机种子，添加到命令行参数
            effective_seed = seed if seed is not None else config.get("seed")
            if effective_seed is not None:
                cmd.extend(["--seed", str(effective_seed)])
            
            # 创建主日志文件，避免 stdout/stderr 管道缓冲区满导致进程阻塞
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, 'w', encoding='utf-8')
            # A previous run's abort reason must not become this run's (#62).
            failure_path = os.path.join(sim_dir, "failure.json")
            if os.path.exists(failure_path):
                os.remove(failure_path)

            # 设置子进程环境变量，确保 Windows 上使用 UTF-8 编码
            # 这可以修复第三方库（如 OASIS）读取文件时未指定编码的问题
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'  # Python 3.7+ 支持，让所有 open() 默认使用 UTF-8
            env['PYTHONIOENCODING'] = 'utf-8'  # 确保 stdout/stderr 使用 UTF-8
            
            # 设置工作目录为模拟目录（数据库等文件会生成在此）
            # 使用 start_new_session=True 创建新的进程组，确保可以通过 os.killpg 终止所有子进程
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # stderr 也写入同一个文件
                text=True,
                encoding='utf-8',  # 显式指定编码
                bufsize=1,
                env=env,  # 传递带有 UTF-8 设置的环境变量
                start_new_session=True,  # 创建新进程组，确保服务器关闭时能终止所有相关进程
            )
            
            # Capture locale before spawning monitor thread
            current_locale = get_locale()

            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id, current_locale),
                daemon=True
            )

            # Atomically publish every resource needed by stop/finalization.
            # The monitor is registered before start; if it exits immediately,
            # it waits on the same lock until RUNNING is fully visible.
            with cls._finalization_lock(simulation_id):
                cls._stdout_files[simulation_id] = main_log_file
                cls._stderr_files[simulation_id] = None
                state.process_pid = process.pid
                state.runner_status = RunnerStatus.RUNNING
                cls._processes[simulation_id] = process
                cls._monitor_threads[simulation_id] = monitor_thread
                cls._save_run_state(state)
                cls._sync_simulation_status(
                    simulation_id,
                    RunnerStatus.RUNNING,
                )
                monitor_thread.start()
            
            logger.info(f"模拟启动成功: {simulation_id}, pid={process.pid}, platform={platform}")
            
        except Exception as e:
            cleanup_errors = []
            if process is not None and process.poll() is None:
                try:
                    cls._terminate_process(process, simulation_id)
                except Exception as error:
                    cleanup_errors.append(f"子进程终止失败: {error}")
            cls._processes.pop(simulation_id, None)
            cls._monitor_threads.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)
            cls._stdout_files.pop(simulation_id, None)
            cls._stderr_files.pop(simulation_id, None)
            if main_log_file is not None:
                try:
                    main_log_file.close()
                except Exception as error:
                    cleanup_errors.append(f"日志关闭失败: {error}")
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    GraphMemoryManager.stop_updater(simulation_id)
                    cls._graph_memory_enabled.pop(simulation_id, None)
                except Exception as error:
                    cleanup_errors.append(f"Zep图谱写入清理失败: {error}")
            state.runner_status = RunnerStatus.FAILED
            state.twitter_running = False
            state.reddit_running = False
            state.error = str(e)
            if cleanup_errors:
                state.error += "; " + "; ".join(cleanup_errors)
            with cls._finalization_lock(simulation_id):
                cls._save_run_state(state)
                cls._sync_simulation_status(
                    simulation_id,
                    RunnerStatus.FAILED,
                    state.error,
                )
            raise
        
        return state

    @classmethod
    def _monitor_simulation(cls, simulation_id: str, locale: str = 'zh'):
        """监控模拟进程，解析动作日志"""
        set_locale(locale)
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        # 新的日志结构：分平台的动作日志
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)
        
        if not process or not state:
            return
        
        twitter_position = 0
        reddit_position = 0
        
        monitor_error: Exception | None = None
        exit_code: int | None = None
        try:
            # A newer run of the same simulation replaces the registered
            # process; this monitor then stops reading and writing state.
            while process.poll() is None and cls._processes.get(simulation_id) is process:
                # 读取 Twitter 动作日志
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )
                
                # 读取 Reddit 动作日志
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )
                
                # 更新状态
                cls._save_run_state(state)
                # After both platforms end, the script keeps its environment
                # for interviews and does not exit (#49). Publish completion
                # here, behind the same ingestion drain, instead of waiting
                # for an exit that only comes with close_env or a stop.
                if (
                    state.runner_status == RunnerStatus.RUNNING
                    and simulation_id not in cls._manual_stop_requests
                    and cls._check_all_platforms_completed(state)
                ):
                    with cls._finalization_lock(simulation_id):
                        # A stop can take the lock between the check and here.
                        latest = cls.get_run_state(simulation_id) or state
                        if (
                            latest.runner_status == RunnerStatus.RUNNING
                            and simulation_id not in cls._manual_stop_requests
                        ):
                            if latest is not state:
                                latest.twitter_completed = state.twitter_completed
                                latest.reddit_completed = state.reddit_completed
                                state = latest
                            cls._publish_terminal(simulation_id, state, RunnerStatus.COMPLETED, None)
                time.sleep(2)
            
            # 进程结束后，最后读取一次日志
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(twitter_actions_log, twitter_position, state, "twitter")
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(reddit_actions_log, reddit_position, state, "reddit")
            
            exit_code = process.returncode
            
        except Exception as e:
            logger.error(f"监控线程异常: {simulation_id}, error={str(e)}")
            monitor_error = e
        
        finally:
            # Manual stop and natural completion can observe the same process
            # exit. Serialize terminal state and updater drain so only one path
            # owns the final result.
            owner = cls._processes.get(simulation_id) in (None, process)
            with cls._finalization_lock(simulation_id):
                latest_state = cls.get_run_state(simulation_id)
                if latest_state is not None:
                    state = latest_state

                if owner and state.runner_status not in {
                    RunnerStatus.STOPPED,
                    RunnerStatus.FAILED,
                    RunnerStatus.COMPLETED,  # published when both platforms ended (#49)
                }:
                    manual_stop = simulation_id in cls._manual_stop_requests
                    desired_status = (
                        RunnerStatus.STOPPED
                        if manual_stop
                        else RunnerStatus.COMPLETED
                    )
                    error_message = None
                    if not manual_stop and monitor_error is not None:
                        desired_status = RunnerStatus.FAILED
                        error_message = str(monitor_error)
                    elif not manual_stop and exit_code != 0:
                        desired_status = RunnerStatus.FAILED
                        error_message = cls._failure_reason(sim_dir, exit_code)

                    cls._publish_terminal(simulation_id, state, desired_status, error_message)
                cls._manual_stop_requests.discard(simulation_id)
            
            # 清理进程资源（a newer run's entries are not this monitor's to drop）
            if owner:
                cls._processes.pop(simulation_id, None)
                cls._action_queues.pop(simulation_id, None)
                cls._monitor_threads.pop(simulation_id, None)
            
            # 关闭日志文件句柄
            if simulation_id in cls._stdout_files:
                try:
                    cls._stdout_files[simulation_id].close()
                except Exception:
                    pass
                cls._stdout_files.pop(simulation_id, None)
            if simulation_id in cls._stderr_files and cls._stderr_files[simulation_id]:
                try:
                    cls._stderr_files[simulation_id].close()
                except Exception:
                    pass
                cls._stderr_files.pop(simulation_id, None)

    @classmethod
    def _terminate_process(cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10):
        """
        跨平台终止进程及其子进程
        
        Args:
            process: 要终止的进程
            simulation_id: 模拟ID（用于日志）
            timeout: 等待进程退出的超时时间（秒）
        """
        if IS_WINDOWS:
            # Windows: 使用 taskkill 命令终止进程树
            # /F = 强制终止, /T = 终止进程树（包括子进程）
            logger.info(f"终止进程树 (Windows): simulation={simulation_id}, pid={process.pid}")
            try:
                # 先尝试优雅终止
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T'],
                    capture_output=True,
                    timeout=5
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # 强制终止
                    logger.warning(f"进程未响应，强制终止: {simulation_id}")
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                        capture_output=True,
                        timeout=5
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"taskkill 失败，尝试 terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: 使用进程组终止
            # 由于使用了 start_new_session=True，进程组 ID 等于主进程 PID
            pgid = os.getpgid(process.pid)
            logger.info(f"终止进程组 (Unix): simulation={simulation_id}, pgid={pgid}")
            
            # 先发送 SIGTERM 给整个进程组
            os.killpg(pgid, signal.SIGTERM)
            
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # 如果超时后还没结束，强制发送 SIGKILL
                logger.warning(f"进程组未响应 SIGTERM，强制终止: {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)

    @classmethod
    def close_environment(cls, simulation_id: str, timeout: float = 30.0) -> bool:
        """End the interview environment of a COMPLETED run.

        After both platforms end, the script stays up for interviews and the
        run is already COMPLETED (#49). Restart, stop and shutdown must still
        end that process. Called without the finalization lock: the monitor's
        exit path takes it. Returns whether a live process was ended.
        """

        process = cls._processes.get(simulation_id)
        if process is None or process.poll() is not None:
            return False
        try:
            cls._terminate_process(process, simulation_id)
        except ProcessLookupError:
            pass
        except Exception as error:  # e.g. TimeoutExpired on the final wait
            logger.error(f"关闭模拟环境失败，强制结束: {simulation_id}, error={error}")
            try:
                process.kill()
            except Exception:
                pass
        monitor = cls._monitor_threads.get(simulation_id)
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout)
        return True

    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """停止模拟"""
        current = cls.get_run_state(simulation_id)
        if current is not None and current.runner_status == RunnerStatus.COMPLETED:
            # Finished: only the interview environment may still be up.
            cls.close_environment(simulation_id)
            return cls.get_run_state(simulation_id) or current
        try:
            with cls._finalization_lock(simulation_id):
                state = cls.get_run_state(simulation_id)
                if not state:
                    raise ValueError(f"模拟不存在: {simulation_id}")
                if state.runner_status == RunnerStatus.STOPPED:
                    return state
                if state.runner_status == RunnerStatus.COMPLETED:
                    # Published by the monitor while this stop waited on the
                    # lock (the drain shows STOPPING). End the environment
                    # outside the lock: its monitor takes the lock on exit.
                    raise _CompletedWhileStopping()

                pending_updater = GraphMemoryManager.get_updater(simulation_id)
                retrying_finalization = (
                    pending_updater is not None
                    and state.runner_status in {
                        RunnerStatus.STOPPING,
                        RunnerStatus.FAILED,
                    }
                )
                if (
                    state.runner_status not in [
                        RunnerStatus.STARTING,
                        RunnerStatus.RUNNING,
                        RunnerStatus.PAUSED,
                        RunnerStatus.STOPPING,
                    ]
                    and not retrying_finalization
                ):
                    raise ValueError(
                        f"模拟未在运行: {simulation_id}, status={state.runner_status}"
                    )

                state.runner_status = RunnerStatus.STOPPING
                cls._manual_stop_requests.add(simulation_id)
                cls._save_run_state(state)
                cls._sync_simulation_status(simulation_id, RunnerStatus.STOPPING)

                # 终止进程
                process = cls._processes.get(simulation_id)
                if process and process.poll() is None:
                    try:
                        cls._terminate_process(process, simulation_id)
                    except ProcessLookupError:
                        pass
                    except Exception as e:
                        logger.error(f"终止进程组失败: {simulation_id}, error={e}")
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except Exception:
                            process.kill()

            # Let the monitor consume the final action-log tail and own the single
            # updater drain. It will publish STOPPED (rather than COMPLETED) because
            # the manual-stop marker is set above.
            monitor = cls._monitor_threads.get(simulation_id)
            if (
                not retrying_finalization
                and
                monitor is not None
                and monitor is not threading.current_thread()
                and monitor.is_alive()
            ):
                wait_timeout = max(
                    30.0,
                    ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
                    + ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
                    + 5,
                )
                monitor.join(timeout=wait_timeout)
                if monitor.is_alive():
                    # The monitor still owns finalization and may be inside one
                    # bounded HTTP request. Do not block on or overwrite its lock;
                    # leave the observable state as STOPPING and let polling expose
                    # the eventual STOPPED/FAILED result.
                    raise SimulationStopPending(
                        f"模拟仍在停止中，图谱写入未在 {wait_timeout:.0f}s 内完成"
                    )
            else:
                # Restart recovery or tests may have no monitor thread. Complete
                # the same barrier synchronously in this request.
                with cls._finalization_lock(simulation_id):
                    state = cls.get_run_state(simulation_id) or state
                    if cls._graph_memory_enabled.get(simulation_id, False):
                        try:
                            GraphMemoryManager.stop_updater(simulation_id)
                            cls._graph_memory_enabled.pop(simulation_id, None)
                        except Exception as error:
                            state.runner_status = RunnerStatus.FAILED
                            state.twitter_running = False
                            state.reddit_running = False
                            state.completed_at = datetime.now().isoformat()
                            state.error = f"Zep图谱写入未完整完成: {error}"
                            cls._save_run_state(state)
                            cls._sync_simulation_status(
                                simulation_id,
                                RunnerStatus.FAILED,
                                state.error,
                            )
                            raise RuntimeError(state.error) from error
                    state.runner_status = RunnerStatus.STOPPED
                    state.twitter_running = False
                    state.reddit_running = False
                    state.completed_at = datetime.now().isoformat()
                    state.error = None
                    cls._save_run_state(state)
                    cls._sync_simulation_status(
                        simulation_id,
                        RunnerStatus.STOPPED,
                    )
                    cls._manual_stop_requests.discard(simulation_id)
        except _CompletedWhileStopping:
            cls.close_environment(simulation_id)
            return cls.get_run_state(simulation_id) or current

        state = cls.get_run_state(simulation_id) or state
        if state.runner_status == RunnerStatus.FAILED:
            raise RuntimeError(state.error or "模拟停止失败")
        if state.runner_status != RunnerStatus.STOPPED:
            raise RuntimeError(
                f"模拟停止未达到终态: {simulation_id}, status={state.runner_status}"
            )

        logger.info(f"模拟已停止: {simulation_id}")
        return state

    _cleanup_done = False
