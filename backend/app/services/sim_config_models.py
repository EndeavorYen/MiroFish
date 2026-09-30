"""Simulation parameter dataclasses: agents, time, events, platforms."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class AgentActivityConfig:
    """单个Agent的活动配置"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # 活跃度配置 (0.0-1.0)
    activity_level: float = 0.5  # 整体活跃度
    
    # 发言频率（每小时预期发言次数）
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # 活跃时间段（24小时制，0-23）
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # 响应速度（对热点事件的反应延迟，单位：模拟分钟）
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # 情感倾向 (-1.0到1.0，负面到正面)
    sentiment_bias: float = 0.0
    
    # 立场（对特定话题的态度）
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # 影响力权重（决定其发言被其他Agent看到的概率）
    influence_weight: float = 1.0

    # 结构化准备的原始立场与利害角色，供校准拟合用 (#46)；LLM 准备为 None
    stance_raw: Optional[float] = None
    stakeholder_role: Optional[str] = None


@dataclass  
class TimeSimulationConfig:
    """时间模拟配置（基于中国人作息习惯）"""
    # 模拟总时长（模拟小时数）
    total_simulation_hours: int = 72  # 默认模拟72小时（3天）
    
    # 每轮代表的时间（模拟分钟）- 默认60分钟（1小时），加快时间流速
    minutes_per_round: int = 60
    
    # 每小时激活的Agent数量范围
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # 高峰时段（晚间19-22点，中国人最活跃的时间）
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # 低谷时段（凌晨0-5点，几乎无人活动）
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # 凌晨活跃度极低
    
    # 早间时段
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # 工作时段
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """事件配置"""
    # 初始事件（模拟开始时的触发事件）
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # 定时事件（在特定时间触发的事件）
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 热点话题关键词
    hot_topics: List[str] = field(default_factory=list)
    
    # 舆论引导方向
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """平台特定配置"""
    platform: str  # twitter or reddit
    
    # 推荐算法权重
    recency_weight: float = 0.4  # 时间新鲜度
    popularity_weight: float = 0.3  # 热度
    relevance_weight: float = 0.3  # 相关性
    
    # 病毒传播阈值（达到多少互动后触发扩散）
    viral_threshold: int = 10
    
    # 回声室效应强度（相似观点聚集程度）
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """完整的模拟参数配置"""
    # 基础信息
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # 时间配置
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agent配置列表
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # 事件配置
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # 平台配置
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLM配置
    llm_model: str = ""
    llm_base_url: str = ""
    
    # 生成元数据
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM的推理说明
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
