"""
图谱构建服务
使用 LLM 实体抽取 + 本地 GraphStore 构建知识图谱
"""

import uuid
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor


EXTRACTION_SYSTEM_PROMPT = """你是一个知识图谱实体关系抽取引擎。
从给定文本中抽取实体和关系。

输出JSON格式：
{
  "entities": [{"name": "实体名", "type": "实体类型", "description": "简短描述"}],
  "relations": [{"source": "源实体名", "target": "目标实体名", "type": "关系类型", "fact": "关系描述"}]
}

实体类型可以是：Person, Organization, Location, Event, Product, Policy, 或其他合适的类型。
关系类型应该简短且描述性强。"""


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    图谱构建服务
    使用 LLM 抽取实体关系并存入本地 GraphStore
    """

    def __init__(self, store, llm_client=None):
        self.store = store
        self.llm_client = llm_client
        self.task_manager = TaskManager()

    def build_graph_async(
        self,
        text: str,
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> str:
        """
        异步构建图谱

        Args:
            text: 输入文本
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小

        Returns:
            任务ID
        """
        if not self.llm_client:
            raise ValueError("llm_client is required for graph building")
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, graph_name, chunk_size, chunk_overlap)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="开始构建图谱..."
            )

            # 1. 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"图谱已创建: {graph_id}"
            )

            # 2. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=f"文本已分割为 {total_chunks} 个块"
            )

            # 3. 逐块抽取实体关系
            for idx, chunk in enumerate(chunks):
                self.add_text_and_extract(graph_id, chunk)
                progress = 15 + int((idx + 1) / total_chunks * 75)  # 15-90%
                self.task_manager.update_task(
                    task_id,
                    progress=progress,
                    message=f"已处理 {idx + 1}/{total_chunks} 个文本块"
                )

            # 4. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message="获取图谱信息..."
            )

            graph_info = self._get_graph_info(graph_id)

            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """创建图谱"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.store.create_graph(graph_id, name, "MiroFish Social Simulation Graph")
        return graph_id

    def add_text_and_extract(self, graph_id: str, text: str):
        """Extract entities/relations from text via LLM and store in graph."""
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        result = self.llm_client.chat_json(messages, temperature=0.1)

        # Store entities, track name->uuid mapping for relation resolution
        entity_uuids = {}
        for entity in result.get("entities", []):
            entity_uuid = self.store.add_entity(
                graph_id, entity["name"], entity["type"],
                entity.get("description", ""), {}
            )
            entity_uuids[entity["name"]] = entity_uuid

        # Store relations (resolve cross-chunk entities from store if needed)
        for rel in result.get("relations", []):
            source_uuid = entity_uuids.get(rel["source"])
            target_uuid = entity_uuids.get(rel["target"])
            # Look up from store if not in current chunk's extractions
            if not source_uuid:
                source_uuid = self._find_entity_uuid(graph_id, rel["source"])
            if not target_uuid:
                target_uuid = self._find_entity_uuid(graph_id, rel["target"])
            if source_uuid and target_uuid:
                self.store.add_relation(
                    graph_id, source_uuid, target_uuid,
                    rel["type"], rel.get("fact", "")
                )

    def _find_entity_uuid(self, graph_id: str, name: str) -> Optional[str]:
        """Look up an entity UUID by name from the store."""
        if hasattr(self.store, 'find_entity_by_name'):
            return self.store.find_entity_by_name(graph_id, name)
        # Fallback for store implementations without find_entity_by_name
        norm_name = name.strip().lower()
        for entity in self.store.list_entities(graph_id, limit=10000):
            if entity.name.strip().lower() == norm_name:
                return entity.uuid
        return None

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息"""
        entities = self.store.list_entities(graph_id, limit=10000)
        relations = self.store.list_relations(graph_id, limit=10000)
        entity_types = set()
        for entity in entities:
            etype = entity.get_entity_type()
            if etype != "Unknown":
                entity_types.add(etype)
        return GraphInfo(
            graph_id=graph_id,
            node_count=len(entities),
            edge_count=len(relations),
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """获取完整图谱数据（包含详细信息）"""
        entities = self.store.list_entities(graph_id, limit=10000)
        relations = self.store.list_relations(graph_id, limit=10000)

        node_map = {e.uuid: e.name for e in entities}

        nodes_data = [
            {
                "uuid": e.uuid, "name": e.name, "labels": e.labels,
                "summary": e.summary, "attributes": e.attributes,
                "created_at": None,
            }
            for e in entities
        ]

        edges_data = []
        for r in relations:
            d = r.to_dict()
            d["fact_type"] = r.name
            d["source_node_name"] = node_map.get(r.source_node_uuid, "")
            d["target_node_name"] = node_map.get(r.target_node_uuid, "")
            d["episodes"] = []
            edges_data.append(d)

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱"""
        self.store.delete_graph(graph_id)
