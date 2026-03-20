"""
本地图谱记忆更新服务
将模拟中的Agent活动直接转换为图谱边 + SQLite文本记录
无需LLM调用，零成本
"""

import threading
from typing import Dict, Any, Optional
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger('mirofish.graph_memory_updater')

# Maps action_type -> (edge_type, needs_post_node, needs_text_index)
# needs_post_node: whether to create/resolve a post node as target
# needs_text_index: whether to store content text in the text store
ACTION_MAP: Dict[str, tuple] = {
    "CREATE_POST":      ("POSTED",          True,  True),
    "LIKE_POST":        ("LIKED",           True,  False),
    "DISLIKE_POST":     ("DISLIKED",        True,  False),
    "REPOST":           ("REPOSTED",        True,  False),
    "QUOTE_POST":       ("QUOTED",          True,  True),
    "FOLLOW":           ("FOLLOWS",         False, False),
    "CREATE_COMMENT":   ("COMMENTED_ON",    True,  True),
    "LIKE_COMMENT":     ("LIKED_COMMENT",   True,  False),
    "DISLIKE_COMMENT":  ("DISLIKED_COMMENT",True,  False),
    "SEARCH_POSTS":     ("SEARCHED",        False, False),
    "SEARCH_USER":      ("SEARCHED_USER",   False, False),
    "MUTE":             ("MUTED",           False, False),
}

SKIP_ACTIONS = {"DO_NOTHING", "REFRESH", "TREND"}


class GraphMemoryUpdater:
    """
    本地图谱记忆更新器

    同步地将OASIS结构化动作记录为图谱边和SQLite文本记录
    无LLM调用，零成本
    """

    # Class-level map from action_type to fact-builder method name
    # Avoids rebuilding a dict on every _build_fact call
    _FACT_BUILDERS: Dict[str, str] = {
        "CREATE_POST": "_fact_create_post",
        "LIKE_POST": "_fact_like_post",
        "DISLIKE_POST": "_fact_dislike_post",
        "REPOST": "_fact_repost",
        "QUOTE_POST": "_fact_quote_post",
        "FOLLOW": "_fact_follow",
        "CREATE_COMMENT": "_fact_create_comment",
        "LIKE_COMMENT": "_fact_like_comment",
        "DISLIKE_COMMENT": "_fact_dislike_comment",
        "SEARCH_POSTS": "_fact_search",
        "SEARCH_USER": "_fact_search_user",
        "MUTE": "_fact_mute",
    }

    def __init__(self, graph_id: str, store):
        """
        初始化更新器

        Args:
            graph_id: 图谱ID
            store: NetworkXGraphStore实例（同时实现GraphStore + TextStore）
        """
        self.graph_id = graph_id
        self.store = store

        # 统计
        self._total_processed = 0
        self._total_skipped = 0
        self._total_failed = 0

        logger.info(f"GraphMemoryUpdater initialized: graph_id={graph_id}")

    def add_activity(self, data: Dict[str, Any]):
        """
        处理一条agent活动，直接写入图谱和文本存储

        Args:
            data: 活动字典，包含以下字段：
                - platform: 平台名称 (twitter/reddit)
                - agent_id: Agent的标识名 (如 "agent_1")
                - agent_name: Agent的显示名称 (如 "User1")
                - action_type: 动作类型
                - action_args: 动作参数
                - round_num: 轮次编号
                - timestamp: 时间戳字符串
        """
        action_type = data.get("action_type", "")

        # 跳过无意义动作
        if action_type in SKIP_ACTIONS:
            self._total_skipped += 1
            logger.debug(f"Skipping action: {action_type}")
            return

        if action_type not in ACTION_MAP:
            self._total_skipped += 1
            logger.debug(f"Unknown action type, skipping: {action_type}")
            return

        try:
            self._process_activity(data, action_type)
            self._total_processed += 1
        except Exception as e:
            self._total_failed += 1
            logger.error(f"Activity processing failed: action_type={action_type}, error={e}")

    def _process_activity(self, data: Dict[str, Any], action_type: str):
        """内部处理逻辑"""
        agent_id = str(data.get("agent_id", ""))
        agent_name = data.get("agent_name", agent_id)
        action_args = data.get("action_args", {})
        round_num = data.get("round_num", 0)
        timestamp_str = data.get("timestamp", datetime.now().isoformat())

        edge_type, needs_post_node, needs_text_index = ACTION_MAP[action_type]

        # 1. 找到/创建agent节点
        agent_uuid = self._ensure_agent(agent_id, agent_name)

        # 2. 确定目标节点
        target_uuid = self._resolve_target(
            action_type, action_args, needs_post_node, agent_id
        )

        # 3. 生成中文描述作为fact
        fact = self._build_fact(action_type, agent_name, action_args)

        # 4. 添加关系边（agent -> target）
        if target_uuid is not None:
            self.store.add_relation(
                self.graph_id,
                source_uuid=agent_uuid,
                target_uuid=target_uuid,
                relation_type=edge_type,
                fact=fact,
                attributes={
                    "action_type": action_type,
                    "round_num": round_num,
                    "timestamp": timestamp_str,
                },
            )

        # 5. 可选：将文本内容存入文本索引
        if needs_text_index:
            content = action_args.get("content", "")
            if content:
                try:
                    ts = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    ts = datetime.now()
                self.store.add_text(
                    graph_id=self.graph_id,
                    agent_id=agent_id,
                    content=content,
                    action_type=action_type,
                    metadata={"agent_name": agent_name, "round_num": round_num},
                    tick=round_num,
                    timestamp=ts,
                )

    def _ensure_agent(self, agent_id: str, agent_name: str) -> str:
        """
        找到已有agent节点或创建新节点

        Uses add_entity's built-in dedup (name_normalized + entity_type)
        to avoid scanning all entities.

        Returns:
            agent节点的UUID
        """
        return self.store.add_entity(
            self.graph_id,
            name=agent_id,
            entity_type="agent",
            summary=agent_name,
            attributes={"display_name": agent_name},
        )

    def _resolve_target(
        self,
        action_type: str,
        action_args: Dict[str, Any],
        needs_post_node: bool,
        agent_id: str,
    ) -> Optional[str]:
        """
        根据动作类型确定目标节点UUID

        Returns:
            目标节点UUID，或None（若无目标）
        """
        if action_type in ("FOLLOW", "MUTE"):
            # 目标是另一个agent
            target_name = action_args.get("target_user_name", "")
            if not target_name:
                return None
            return self.store.add_entity(
                self.graph_id,
                name=target_name,
                entity_type="agent",
                summary=target_name,
                attributes={},
            )

        if action_type in ("SEARCH_POSTS", "SEARCH_USER"):
            # 搜索动作：目标是搜索关键词节点
            query = (
                action_args.get("query")
                or action_args.get("keyword")
                or action_args.get("username")
                or ""
            )
            if not query:
                return None
            return self.store.add_entity(
                self.graph_id,
                name=query,
                entity_type="keyword",
                summary=query,
                attributes={},
            )

        if needs_post_node:
            # 目标是帖子/评论节点
            post_id = action_args.get("post_id") or action_args.get("comment_id", "")
            if post_id:
                # add_entity deduplicates by name+type, returns existing UUID if found
                content = action_args.get("content", "")
                return self.store.add_entity(
                    self.graph_id,
                    name=str(post_id),
                    entity_type="post",
                    summary=content[:200] if content else "",
                    attributes={"post_id": post_id},
                )
            # 无post_id时创建匿名帖子节点
            content = action_args.get("content", "")
            author = action_args.get("author", action_args.get("post_author_name", ""))
            node_name = f"post_by_{author}" if author else "anonymous_post"
            return self.store.add_entity(
                self.graph_id,
                name=node_name,
                entity_type="post",
                summary=content[:200] if content else "",
                attributes={},
            )

        return None

    def _build_fact(
        self, action_type: str, agent_name: str, action_args: Dict[str, Any]
    ) -> str:
        """
        生成中文描述作为关系的fact字段
        格式与原zep_graph_memory_updater保持一致
        """
        method_name = self._FACT_BUILDERS.get(action_type)
        if method_name:
            description = getattr(self, method_name)(action_args)
        else:
            description = f"performed {action_type} action"
        return f"{agent_name}: {description}"

    def _fact_create_post(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "")
        return f"posted: \"{content}\"" if content else "posted"

    def _fact_like_post(self, args: Dict[str, Any]) -> str:
        content = args.get("post_content", "") or args.get("content", "")
        author = args.get("post_author_name", "") or args.get("author", "")
        if content and author:
            return f"liked {author}'s post: \"{content}\""
        elif content:
            return f"liked a post: \"{content}\""
        elif author:
            return f"liked a post by {author}"
        return "liked a post"

    def _fact_dislike_post(self, args: Dict[str, Any]) -> str:
        content = args.get("post_content", "") or args.get("content", "")
        author = args.get("post_author_name", "") or args.get("author", "")
        if content and author:
            return f"disliked {author}'s post: \"{content}\""
        elif content:
            return f"disliked a post: \"{content}\""
        elif author:
            return f"disliked a post by {author}"
        return "disliked a post"

    def _fact_repost(self, args: Dict[str, Any]) -> str:
        content = args.get("original_content", "") or args.get("content", "")
        author = args.get("original_author_name", "") or args.get("author", "")
        if content and author:
            return f"reposted {author}'s post: \"{content}\""
        elif content:
            return f"reposted a post: \"{content}\""
        elif author:
            return f"reposted a post by {author}"
        return "reposted a post"

    def _fact_quote_post(self, args: Dict[str, Any]) -> str:
        original = args.get("original_content", "")
        author = args.get("original_author_name", "") or args.get("author", "")
        quote = args.get("quote_content", "") or args.get("content", "")
        if original and author:
            base = f"quoted {author}'s post \"{original}\""
        elif original:
            base = f"quoted a post \"{original}\""
        elif author:
            base = f"quoted a post by {author}"
        else:
            base = "quoted a post"
        if quote:
            base += f", commenting: \"{quote}\""
        return base

    def _fact_follow(self, args: Dict[str, Any]) -> str:
        target = args.get("target_user_name", "")
        return f"followed user \"{target}\"" if target else "followed a user"

    def _fact_create_comment(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "")
        post_content = args.get("post_content", "")
        post_author = args.get("post_author_name", "") or args.get("author", "")
        if content:
            if post_content and post_author:
                return f"commented on {post_author}'s post \"{post_content}\": \"{content}\""
            elif post_content:
                return f"commented on post \"{post_content}\": \"{content}\""
            elif post_author:
                return f"commented on {post_author}'s post: \"{content}\""
            return f"commented: \"{content}\""
        return "posted a comment"

    def _fact_like_comment(self, args: Dict[str, Any]) -> str:
        content = args.get("comment_content", "")
        author = args.get("comment_author_name", "") or args.get("author", "")
        if content and author:
            return f"liked {author}'s comment: \"{content}\""
        elif content:
            return f"liked a comment: \"{content}\""
        elif author:
            return f"liked a comment by {author}"
        return "liked a comment"

    def _fact_dislike_comment(self, args: Dict[str, Any]) -> str:
        content = args.get("comment_content", "")
        author = args.get("comment_author_name", "") or args.get("author", "")
        if content and author:
            return f"disliked {author}'s comment: \"{content}\""
        elif content:
            return f"disliked a comment: \"{content}\""
        elif author:
            return f"disliked a comment by {author}"
        return "disliked a comment"

    def _fact_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "") or args.get("keyword", "")
        return f"searched for \"{query}\"" if query else "performed a search"

    def _fact_search_user(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "") or args.get("username", "")
        return f"searched for user \"{query}\"" if query else "searched for a user"

    def _fact_mute(self, args: Dict[str, Any]) -> str:
        target = args.get("target_user_name", "")
        return f"muted user \"{target}\"" if target else "muted a user"

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "graph_id": self.graph_id,
            "total_processed": self._total_processed,
            "total_skipped": self._total_skipped,
            "total_failed": self._total_failed,
        }


class GraphMemoryManager:
    """
    管理多个模拟的本地图谱记忆更新器

    每个模拟可以有自己的更新器实例
    """

    _updaters: Dict[str, GraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(
        cls, simulation_id: str, graph_id: str, store
    ) -> GraphMemoryUpdater:
        """
        为模拟创建图谱记忆更新器

        Args:
            simulation_id: 模拟ID
            graph_id: 图谱ID
            store: NetworkXGraphStore实例

        Returns:
            GraphMemoryUpdater实例
        """
        with cls._lock:
            # 如果已存在同名更新器则直接替换
            updater = GraphMemoryUpdater(graph_id=graph_id, store=store)
            cls._updaters[simulation_id] = updater
            logger.info(
                f"Created graph memory updater: simulation_id={simulation_id}, graph_id={graph_id}"
            )
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[GraphMemoryUpdater]:
        """获取模拟的更新器"""
        with cls._lock:
            return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        """移除模拟的更新器"""
        with cls._lock:
            if simulation_id in cls._updaters:
                del cls._updaters[simulation_id]
                logger.info(f"Stopped graph memory updater: simulation_id={simulation_id}")

    @classmethod
    def stop_all(cls):
        """移除所有更新器"""
        with cls._lock:
            cls._updaters.clear()
            logger.info("Stopped all graph memory updaters")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有更新器的统计信息"""
        with cls._lock:
            return {
                sim_id: updater.get_stats()
                for sim_id, updater in cls._updaters.items()
            }
