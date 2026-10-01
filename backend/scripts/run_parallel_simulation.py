"""
OASIS 双平台并行模拟预设脚本
同时运行Twitter和Reddit模拟，读取相同的配置文件

功能特性:
- 双平台（Twitter + Reddit）并行模拟
- 完成模拟后不立即关闭环境，进入等待命令模式
- 支持通过IPC接收Interview命令
- 支持单个Agent采访和批量采访
- 支持远程关闭环境命令

使用方式:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait  # 完成后立即关闭
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

日志结构:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter 平台动作日志
    ├── reddit/
    │   └── actions.jsonl    # Reddit 平台动作日志
    ├── simulation.log       # 主模拟进程日志
    └── run_state.json       # 运行状态（API 查询用）
"""

# ============================================================
# 解决 Windows 编码问题：在所有 import 之前设置 UTF-8 编码
# 这是为了修复 OASIS 第三方库读取文件时未指定编码的问题
# ============================================================
import sys
import os

if sys.platform == 'win32':
    # 设置 Python 默认 I/O 编码为 UTF-8
    # 这会影响所有未指定编码的 open() 调用
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    
    # 重新配置标准输出流为 UTF-8（解决控制台中文乱码）
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    # 强制设置默认编码（影响 open() 函数的默认编码）
    # 注意：这需要在 Python 启动时就设置，运行时设置可能不生效
    # 所以我们还需要 monkey-patch 内置的 open 函数
    import builtins
    _original_open = builtins.open
    
    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, 
                   newline=None, closefd=True, opener=None):
        """
        包装 open() 函数，对于文本模式默认使用 UTF-8 编码
        这可以修复第三方库（如 OASIS）读取文件时未指定编码的问题
        """
        # 只对文本模式（非二进制）且未指定编码的情况设置默认编码
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors, 
                              newline, closefd, opener)
    
    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import random
import signal
from datetime import datetime
from typing import Dict, Any, Optional


# 全局变量：用于信号处理
_shutdown_event = None
_cleanup_done = False

# 添加 backend 目录到路径
# 脚本固定位于 backend/scripts/ 目录
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# 加载项目根目录的 .env 文件（包含 LLM_API_KEY 等配置）
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"已加载环境配置: {_env_file}")
else:
    # 尝试加载 backend/.env
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"已加载环境配置: {_backend_env}")

from app.profiles import apply_profile

apply_profile()  # MIROFISH_PROFILE after this script's own .env load (#36)
from app.simulation_policy.lazy_imports import defer_neo4j
from app.simulation_policy.light_recsys import install as install_light_recsys

# Before anything imports oasis: its recommender loads torch and twhin-bert,
# its agent graph the neo4j driver MiroFish does not use (#65).
install_light_recsys(os.environ)
defer_neo4j()
from app.utils.llm_usage import usage_stage
from app.simulation_policy.oasis_bridge import (
    ModelServiceUnavailable,
    build_policy,
    decision_backend,
    record_failure,
    system_one_actions,
)


class MaxTokensWarningFilter(logging.Filter):
    """过滤掉 camel-ai 关于 max_tokens 的警告（我们故意不设置 max_tokens，让模型自行决定）"""
    
    def filter(self, record):
        # 过滤掉包含 max_tokens 警告的日志
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# 在模块加载时立即添加过滤器，确保在 camel 代码执行前生效
logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging():
    """
    禁用 OASIS 库的详细日志输出
    OASIS 的日志太冗余（记录每个 agent 的观察和动作），我们使用自己的 action_logger
    """
    # 禁用 OASIS 的所有日志器
    oasis_loggers = [
        "social.agent",
        "social.twitter", 
        "social.rec",
        "oasis.env",
        "table",
    ]
    
    for logger_name in oasis_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # 只记录严重错误
        logger.handlers.clear()
        logger.propagate = False


def init_logging_for_simulation(simulation_dir: str):
    """
    初始化模拟的日志配置
    
    Args:
        simulation_dir: 模拟目录路径
    """
    # 禁用 OASIS 的详细日志
    disable_oasis_logging()
    
    # 清理旧的 log 目录（如果存在）
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)


from action_logger import SimulationLogManager, PlatformActionLogger

try:
    from camel.models import ModelFactory  # noqa: F401  (dependency check)
    from camel.types import ModelPlatformType  # noqa: F401  (dependency check)
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请先安装: pip install oasis-ai camel-ai")
    sys.exit(1)

# Chinese feed text as characters, not \uXXXX escapes (#28).
from app.utils.oasis_prompts import install_unicode_observations  # noqa: E402

install_unicode_observations()


# Twitter可用动作（不包含INTERVIEW，INTERVIEW只能通过ManualAction手动触发）
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]

# Reddit可用动作（不包含INTERVIEW，INTERVIEW只能通过ManualAction手动触发）
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
]


# Split in #68; these names stay importable from this script.
from sim_ipc import (  # noqa: E402,F401
    IPC_COMMANDS_DIR,
    IPC_RESPONSES_DIR,
    ENV_STATUS_FILE,
    CommandType,
    ParallelIPCHandler,
)
from sim_actions import (  # noqa: E402,F401
    FILTERED_ACTIONS,
    ACTION_TYPE_MAP,
    get_agent_names_from_config,
    fetch_new_actions_from_db,
    _enrich_action_context,
    _get_post_info,
    _get_user_name,
    _get_comment_info,
)
from sim_agents import (  # noqa: E402,F401
    _apply_memory_budget,
    create_model,
    get_active_agents_for_round,
    PlatformSimulation,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def run_twitter_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
    rng: Optional[random.Random] = None,
    policy: Any = None,
) -> PlatformSimulation:
    """运行Twitter模拟
    
    Args:
        config: 模拟配置
        simulation_dir: 模拟目录
        action_logger: 动作日志记录器
        main_logger: 主日志管理器
        max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
        
    Returns:
        PlatformSimulation: 包含env和agent_graph的结果对象
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Twitter] {msg}")
        print(f"[Twitter] {msg}")
    
    log_info("初始化...")
    
    # Twitter 使用通用 LLM 配置
    model = create_model(config, use_boost=False)
    
    # OASIS Twitter使用CSV格式
    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        log_info(f"错误: Profile文件不存在: {profile_path}")
        return result
    
    result.agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )
    _apply_memory_budget(result.agent_graph, "Twitter", log_info)
    
    # 从配置文件获取 Agent 真实名称映射（使用 entity_name 而非默认的 Agent_X）
    agent_names = get_agent_names_from_config(config)
    # 如果配置中没有某个 agent，则使用 OASIS 的默认名称
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        semaphore=30,  # 限制最大并发 LLM 请求数，防止 API 过载
    )
    
    await result.env.reset()
    log_info("环境已启动")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # 跟踪数据库中最后处理的行号（使用 rowid 避免 created_at 格式差异）
    
    # 执行初始事件
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    # 记录 round 0 开始（初始事件阶段）
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"已发布 {len(initial_actions)} 条初始帖子")
    
    # 记录 round 0 结束
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    # 主模拟循环
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # 如果指定了最大轮数，则截断
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        # 检查是否收到退出信号
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"收到退出信号，在第 {round_num + 1} 轮停止模拟")
            break
        
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num, rng=rng
        )
        
        # 无论是否有活跃agent，都记录round开始
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        
        if not active_agents:
            # 没有活跃agent时也记录round结束（actions_count=0）
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue
        
        metrics_dir = os.path.join(simulation_dir, "metrics")
        with usage_stage("simulation", metrics_dir=metrics_dir):
            if policy is not None:
                actions = await system_one_actions(
                    result.env,
                    active_agents,
                    policy,
                    "twitter",
                    round_num + 1,  # same numbering as actions.jsonl
                    topics=config.get("event_config", {}).get("hot_topics", []),
                )
            else:
                actions = {agent: LLMAction() for _, agent in active_agents}
            await result.env.step(actions)
        
        # 从数据库获取实际执行的动作并记录
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
                total_actions += 1
                round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    # 注意：不关闭环境，保留给Interview使用
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"模拟循环完成! 耗时: {elapsed:.1f}秒, 总动作: {total_actions}")
    
    return result


async def run_reddit_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
    rng: Optional[random.Random] = None,
    policy: Any = None,
) -> PlatformSimulation:
    """运行Reddit模拟
    
    Args:
        config: 模拟配置
        simulation_dir: 模拟目录
        action_logger: 动作日志记录器
        main_logger: 主日志管理器
        max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
        
    Returns:
        PlatformSimulation: 包含env和agent_graph的结果对象
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")
    
    log_info("初始化...")
    
    # Reddit 使用加速 LLM 配置（如果有的话，否则回退到通用配置）
    model = create_model(config, use_boost=True)
    
    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"错误: Profile文件不存在: {profile_path}")
        return result
    
    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )
    _apply_memory_budget(result.agent_graph, "Reddit", log_info)
    
    # 从配置文件获取 Agent 真实名称映射（使用 entity_name 而非默认的 Agent_X）
    agent_names = get_agent_names_from_config(config)
    # 如果配置中没有某个 agent，则使用 OASIS 的默认名称
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=30,  # 限制最大并发 LLM 请求数，防止 API 过载
    )
    
    await result.env.reset()
    log_info("环境已启动")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # 跟踪数据库中最后处理的行号（使用 rowid 避免 created_at 格式差异）
    
    # 执行初始事件
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    # 记录 round 0 开始（初始事件阶段）
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                if agent in initial_actions:
                    if not isinstance(initial_actions[agent], list):
                        initial_actions[agent] = [initial_actions[agent]]
                    initial_actions[agent].append(ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    ))
                else:
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"已发布 {len(initial_actions)} 条初始帖子")
    
    # 记录 round 0 结束
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    # 主模拟循环
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # 如果指定了最大轮数，则截断
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        # 检查是否收到退出信号
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"收到退出信号，在第 {round_num + 1} 轮停止模拟")
            break
        
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num, rng=rng
        )
        
        # 无论是否有活跃agent，都记录round开始
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        
        if not active_agents:
            # 没有活跃agent时也记录round结束（actions_count=0）
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue
        
        metrics_dir = os.path.join(simulation_dir, "metrics")
        with usage_stage("simulation", metrics_dir=metrics_dir):
            if policy is not None:
                actions = await system_one_actions(
                    result.env,
                    active_agents,
                    policy,
                    "reddit",
                    round_num + 1,  # same numbering as actions.jsonl
                    topics=config.get("event_config", {}).get("hot_topics", []),
                )
            else:
                actions = {agent: LLMAction() for _, agent in active_agents}
            await result.env.step(actions)
        
        # 从数据库获取实际执行的动作并记录
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
                total_actions += 1
                round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    # 注意：不关闭环境，保留给Interview使用
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"模拟循环完成! 耗时: {elapsed:.1f}秒, 总动作: {total_actions}")
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='OASIS双平台并行模拟')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='配置文件路径 (simulation_config.json)'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='只运行Twitter模拟'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='只运行Reddit模拟'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='最大模拟轮数（可选，用于截断过长的模拟）'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='模拟完成后立即关闭环境，不进入等待命令模式'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子，用于复现Agent激活序列'
    )
    parser.add_argument(
        '--decision-backend',
        choices=['llm', 'system_one'],
        default=None,
        help='Agent decision backend (default: SIM_DECISION_BACKEND or llm)'
    )
    
    args = parser.parse_args()
    
    # 在 main 函数开始时创建 shutdown 事件，确保整个程序都能响应退出信号
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait
    
    # 初始化日志配置（禁用 OASIS 日志，清理旧文件）
    init_logging_for_simulation(simulation_dir)
    
    # 创建日志管理器
    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()
    
    log_manager.info("=" * 60)
    log_manager.info("OASIS 双平台并行模拟")
    log_manager.info(f"配置文件: {args.config}")
    log_manager.info(f"模拟ID: {config.get('simulation_id', 'unknown')}")
    log_manager.info(f"等待命令模式: {'启用' if wait_for_commands else '禁用'}")
    log_manager.info("=" * 60)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    
    log_manager.info(f"模拟参数:")
    log_manager.info(f"  - 总模拟时长: {total_hours}小时")
    log_manager.info(f"  - 每轮时间: {minutes_per_round}分钟")
    log_manager.info(f"  - 配置总轮数: {config_total_rounds}")
    if args.max_rounds:
        log_manager.info(f"  - 最大轮数限制: {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(f"  - 实际执行轮数: {args.max_rounds} (已截断)")
    log_manager.info(f"  - Agent数量: {len(config.get('agent_configs', []))}")
    
    log_manager.info("日志结构:")
    log_manager.info(f"  - 主日志: simulation.log")
    log_manager.info(f"  - Twitter动作: twitter/actions.jsonl")
    log_manager.info(f"  - Reddit动作: reddit/actions.jsonl")
    log_manager.info("=" * 60)
    
    start_time = datetime.now()
    
    # 存储两个平台的模拟结果
    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    
    raw_seed = args.seed if args.seed is not None else config.get("seed")
    try:
        seed = int(raw_seed) if raw_seed is not None else None
    except (ValueError, TypeError):
        seed = None
    rng_twitter = random.Random(seed) if seed is not None else None
    rng_reddit = random.Random(seed + 1) if seed is not None else None

    policy_twitter = policy_reddit = None
    # Interviews read the resolved backend from the environment.
    os.environ["SIM_DECISION_BACKEND"] = decision_backend(args.decision_backend)
    if os.environ["SIM_DECISION_BACKEND"] == "system_one":
        if seed is not None:
            # OASIS refresh/recsys draw from the global random module.
            random.seed(seed)
        policy_twitter = build_policy("twitter", simulation_dir, seed=seed)
        policy_reddit = build_policy("reddit", simulation_dir, seed=seed + 1 if seed is not None else None)
        log_manager.info("决策后端: system_one（不经 LLM decode）")
    elif os.environ.get("SIM_OPINION_DYNAMICS", "off").strip().lower() not in ("", "0", "off", "false", "no"):
        # Opinion dynamics moves the System One decisions' stance (#59).
        log_manager.info("SIM_OPINION_DYNAMICS 只作用於 SIM_DECISION_BACKEND=system_one；LLM 決策不使用")
    
    try:
        if args.twitter_only:
            twitter_result = await run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds, rng=rng_twitter, policy=policy_twitter)
        elif args.reddit_only:
            reddit_result = await run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds, rng=rng_reddit, policy=policy_reddit)
        else:
            # 并行运行（每个平台使用独立的日志记录器）
            results = await asyncio.gather(
                run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds, rng=rng_twitter, policy=policy_twitter),
                run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds, rng=rng_reddit, policy=policy_reddit),
            )
            twitter_result, reddit_result = results
    except ModelServiceUnavailable as error:
        # The model service stopped answering: stop instead of finishing on
        # failed decisions, and leave the reason for the runner (#62).
        global _EXIT_CODE
        log_manager.info(f"模拟中止: {error}")
        record_failure(simulation_dir, error)
        _EXIT_CODE = 3
        return
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"模拟循环完成! 总耗时: {total_elapsed:.1f}秒")
    
    # 是否进入等待命令模式
    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info("进入等待命令模式 - 环境保持运行")
        log_manager.info("支持的命令: interview, batch_interview, close_env")
        log_manager.info("=" * 60)
        
        # 创建IPC处理器
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None
        )
        ipc_handler.update_status("alive")
        
        # 等待命令循环（使用全局 _shutdown_event）
        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break
                # 使用 wait_for 替代 sleep，这样可以响应 shutdown_event
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break  # 收到退出信号
                except asyncio.TimeoutError:
                    pass  # 超时继续循环
        except KeyboardInterrupt:
            print("\n收到中断信号")
        except asyncio.CancelledError:
            print("\n任务被取消")
        except Exception as e:
            print(f"\n命令处理出错: {e}")
        
        log_manager.info("\n关闭环境...")
        ipc_handler.update_status("stopped")
    
    # 关闭环境
    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] 环境已关闭")
    
    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] 环境已关闭")
    
    log_manager.info("=" * 60)
    log_manager.info(f"全部完成!")
    log_manager.info(f"日志文件:")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)


def setup_signal_handlers(loop=None):
    """
    设置信号处理器，确保收到 SIGTERM/SIGINT 时能够正确退出
    
    持久化模拟场景：模拟完成后不退出，等待 interview 命令
    当收到终止信号时，需要：
    1. 通知 asyncio 循环退出等待
    2. 让程序有机会正常清理资源（关闭数据库、环境等）
    3. 然后才退出
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n收到 {sig_name} 信号，正在退出...")
        
        if not _cleanup_done:
            _cleanup_done = True
            # 设置事件通知 asyncio 循环退出（让循环有机会清理资源）
            if _shutdown_event:
                _shutdown_event.set()
        
        # 不要直接 sys.exit()，让 asyncio 循环正常退出并清理资源
        # 如果是重复收到信号，才强制退出
        else:
            print("强制退出...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


# Non-zero when the run stopped on a failed model service (#62).
_EXIT_CODE = 0

if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被中断")
    except SystemExit:
        pass
    finally:
        # 清理 multiprocessing 资源跟踪器（防止退出时的警告）
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("模拟进程已退出")
    if _EXIT_CODE:
        sys.exit(_EXIT_CODE)
