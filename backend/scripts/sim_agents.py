"""Models, memory budget and per-round agent activation for the simulation.

Split out of run_parallel_simulation.py (#68). Import it only after that
script's bootstrap (sys.path, .env, profile, light recommender) has run.
"""

import os
import random
from typing import Any, Dict, List, Optional

from camel.models import ModelFactory
from camel.types import ModelPlatformType

from app.utils.llm_usage import wrap_camel_model
from app.utils.camel_context import apply_agent_graph_budget, context_budget_from_env


def _apply_memory_budget(agent_graph, platform: str, log) -> None:
    """Keep agent chat memory within SIM_AGENT_CONTEXT_TOKENS (#28): camel
    gives unknown local model names a ~1e9 token limit, so memory would grow
    past the server's per-slot context and agent turns would be lost."""

    budget = context_budget_from_env()
    count = apply_agent_graph_budget(agent_graph, budget)
    if count:
        log(f"[{platform}] agent memory budget: {budget} tokens x {count} agents")


def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    创建LLM模型
    
    支持双 LLM 配置，用于并行模拟时提速：
    - 通用配置：LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
    - 加速配置（可选）：LLM_BOOST_API_KEY, LLM_BOOST_BASE_URL, LLM_BOOST_MODEL_NAME
    
    如果配置了加速 LLM，并行模拟时可以让不同平台使用不同的 API 服务商，提高并发能力。
    
    Args:
        config: 模拟配置字典
        use_boost: 是否使用加速 LLM 配置（如果可用）
    """
    # 检查是否有加速配置
    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)
    
    # 根据参数和配置情况选择使用哪个 LLM
    if use_boost and has_boost_config:
        # 使用加速配置
        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[加速LLM]"
    else:
        # 使用通用配置
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[通用LLM]"
    
    # 如果 .env 中没有模型名，则使用 config 作为备用
    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    
    # 设置 camel-ai 所需的环境变量
    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("缺少 API Key 配置，请在项目根目录 .env 文件中设置 LLM_API_KEY")
    
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
    
    print(f"{config_label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else '默认'}...")
    
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
    )
    return wrap_camel_model(model, default_stage="simulation")


def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int,
    rng: Optional[random.Random] = None
) -> List:
    """根据时间和配置决定本轮激活哪些Agent"""
    r = rng if rng is not None else random
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])
    
    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    
    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
    
    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0
    
    target_count = int(r.uniform(base_min, base_max) * multiplier)
    
    candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)
        
        if current_hour not in active_hours:
            continue
        
        if r.random() < activity_level:
            candidates.append(agent_id)
    
    selected_ids = r.sample(
        candidates, 
        min(target_count, len(candidates))
    ) if candidates else []
    
    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass
    
    return active_agents


class PlatformSimulation:
    """平台模拟结果容器"""
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0
