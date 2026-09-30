"""Reading the action logs, completion and failure of a run."""

import os
import json
from datetime import datetime

from ..utils.logger import get_logger
from .zep_graph_memory_updater import ZepGraphMemoryManager

logger = get_logger('mirofish.simulation_runner')
from .runner_models import (
    RunnerStatus,
    AgentAction,
    SimulationRunState,
)


class MonitorMixin:
    """Part of SimulationRunner (#68); reads its class state through ``cls``."""

    @classmethod
    def _failure_reason(cls, sim_dir: str, exit_code: int) -> str:
        """Why a simulation process exited non-zero: its failure.json when it
        wrote one (the model service stopped answering, #62), else the tail of
        simulation.log."""

        try:
            with open(os.path.join(sim_dir, "failure.json"), encoding="utf-8") as f:
                reason = json.load(f).get("reason")
            if reason:
                return str(reason)
        except (OSError, ValueError):
            pass
        error_info = ""
        try:
            main_log_path = os.path.join(sim_dir, "simulation.log")
            if os.path.exists(main_log_path):
                with open(main_log_path, "r", encoding="utf-8") as f:
                    error_info = f.read()[-2000:]
        except OSError:
            pass
        return f"进程退出码: {exit_code}, 错误: {error_info}"

    @classmethod
    def _publish_terminal(
        cls,
        simulation_id: str,
        state: SimulationRunState,
        desired_status: RunnerStatus,
        error_message: str | None,
    ) -> None:
        """Drain the graph writes, then publish the terminal status. Callers
        hold the finalization lock."""

        state.twitter_running = False
        state.reddit_running = False

        if cls._graph_memory_enabled.get(simulation_id, False):
            # STOPPING is a non-terminal ingestion barrier. The UI
            # and report API must not observe COMPLETED until every
            # accepted episode is processed by Zep Cloud.
            state.runner_status = RunnerStatus.STOPPING
            cls._save_run_state(state)
            cls._sync_simulation_status(
                simulation_id,
                RunnerStatus.STOPPING,
            )
            try:
                ZepGraphMemoryManager.stop_updater(simulation_id)
                cls._graph_memory_enabled.pop(simulation_id, None)
                logger.info(
                    "已停止图谱记忆更新: simulation_id=%s",
                    simulation_id,
                )
            except Exception as error:
                logger.error(f"停止图谱记忆更新器失败: {error}")
                desired_status = RunnerStatus.FAILED
                error_message = f"Zep图谱写入未完整完成: {error}"

        state.runner_status = desired_status
        state.error = error_message
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)
        cls._sync_simulation_status(
            simulation_id,
            desired_status,
            error_message,
        )
        if desired_status == RunnerStatus.COMPLETED:
            logger.info(f"模拟完成: {simulation_id}")
        else:
            logger.error(f"模拟失败: {simulation_id}, error={state.error}")

    @classmethod
    def _read_action_log(
        cls, 
        log_path: str, 
        position: int, 
        state: SimulationRunState,
        platform: str
    ) -> int:
        """
        读取动作日志文件
        
        Args:
            log_path: 日志文件路径
            position: 上次读取位置
            state: 运行状态对象
            platform: 平台名称 (twitter/reddit)
            
        Returns:
            新的读取位置
        """
        # 检查是否启用了图谱记忆更新
        graph_memory_enabled = cls._graph_memory_enabled.get(state.simulation_id, False)
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = ZepGraphMemoryManager.get_updater(state.simulation_id)
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)
                            
                            # 处理事件类型的条目
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")
                                
                                # 检测 simulation_end 事件，标记平台已完成
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                        logger.info(f"Twitter 模拟已完成: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                        logger.info(f"Reddit 模拟已完成: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    
                                    # 检查是否所有启用的平台都已完成
                                    # 如果只运行了一个平台，只检查那个平台
                                    # 如果运行了两个平台，需要两个都完成
                                    all_completed = cls._check_all_platforms_completed(state)
                                    if all_completed:
                                        # Platform completion is only an input
                                        # signal. The monitor publishes the
                                        # terminal status after the process has
                                        # exited and Zep ingestion has drained.
                                        logger.info(
                                            f"所有平台已结束，等待进程与图谱写入完成: "
                                            f"{state.simulation_id}"
                                        )
                                
                                # 更新轮次信息（从 round_end 事件）
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get("simulated_hours", 0)
                                    
                                    # 更新各平台独立的轮次和时间
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours
                                    
                                    # 总体轮次取两个平台的最大值
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # 总体时间取两个平台的最大值
                                    state.simulated_hours = max(state.twitter_simulated_hours, state.reddit_simulated_hours)
                                
                                continue
                            
                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)
                            
                            # 更新轮次
                            if action.round_num and action.round_num > state.current_round:
                                state.current_round = action.round_num
                            
                            # 如果启用了图谱记忆更新，将活动发送到Zep
                            if graph_updater:
                                graph_updater.add_activity_from_dict(action_data, platform)
                            
                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"读取动作日志失败: {log_path}, error={e}")
            return position

    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        检查所有启用的平台是否都已完成模拟
        
        通过检查对应的 actions.jsonl 文件是否存在来判断平台是否被启用
        
        Returns:
            True 如果所有启用的平台都已完成
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        # 检查哪些平台被启用（通过文件是否存在判断）
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)
        
        # 如果平台被启用但未完成，则返回 False
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False
        
        # 至少有一个平台被启用且已完成
        return twitter_enabled or reddit_enabled
