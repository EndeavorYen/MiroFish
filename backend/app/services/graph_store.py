"""
Data models and protocol definitions for graph storage.

Replaces Zep Cloud SDK types with local dataclasses.
GraphStore and TextStore protocols define the interface for storage backends.
"""

from typing import Protocol, Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid as uuid_lib


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid_lib.uuid4())


def normalize_name(name: str) -> str:
    """Normalize entity name for deduplication (case-insensitive, trimmed)."""
    return name.strip().lower()


@dataclass
class EntityNode:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: dict
    related_edges: List[dict] = field(default_factory=list)
    related_nodes: List[dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_entity_type(self) -> str:
        """Return the most specific label (skip generic 'Entity'/'Node')."""
        generic = {"Entity", "Node"}
        for label in self.labels:
            if label not in generic:
                return label
        return self.labels[0] if self.labels else "Unknown"


@dataclass
class RelationEdge:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict
    created_at: datetime
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("created_at", "valid_at", "invalid_at", "expired_at"):
            if d[key] is not None:
                d[key] = d[key].isoformat()
        return d


@dataclass
class SearchResult:
    nodes: List[EntityNode]
    edges: List[RelationEdge]
    facts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "facts": self.facts,
            "total_count": len(self.facts),
        }

    def to_text(self, query: str = "") -> str:
        """Format for LLM consumption."""
        lines = []
        if query:
            lines.append(f"Query: {query}")
        lines.append(f"Found {len(self.facts)} related facts, {len(self.nodes)} entities, {len(self.edges)} relations")
        if self.facts:
            lines.append("\n### Facts:")
            for i, fact in enumerate(self.facts, 1):
                lines.append(f"{i}. {fact}")
        if self.nodes:
            lines.append("\n### Entities:")
            for node in self.nodes:
                labels_str = ", ".join(node.labels)
                lines.append(f"- **{node.name}** ({labels_str}): {node.summary}")
        if self.edges:
            lines.append("\n### Relations:")
            for edge in self.edges:
                lines.append(f"- {edge.name}: {edge.fact}")
        return "\n".join(lines)


@dataclass
class Community:
    id: int
    members: List[str]  # node UUIDs
    summary: Optional[str] = None


class GraphStore(Protocol):
    """Abstraction over graph storage and operations."""

    def create_graph(self, graph_id: str, name: str, description: str) -> str: ...
    def delete_graph(self, graph_id: str) -> None: ...

    def add_entity(self, graph_id: str, name: str, entity_type: str,
                   summary: str, attributes: dict) -> str: ...
    def get_entity(self, graph_id: str, uuid: str) -> EntityNode: ...
    def find_entity_by_name(self, graph_id: str, name: str) -> Optional[str]: ...
    def get_entity_edges(self, graph_id: str, uuid: str) -> List[RelationEdge]: ...
    def list_entities(self, graph_id: str, limit: int = 100,
                      cursor: Optional[str] = None) -> List[EntityNode]: ...

    def add_relation(self, graph_id: str, source_uuid: str, target_uuid: str,
                     relation_type: str, fact: str,
                     attributes: Optional[dict] = None,
                     temporal: Optional[dict] = None) -> str: ...
    def list_relations(self, graph_id: str, limit: int = 100,
                       cursor: Optional[str] = None) -> List[RelationEdge]: ...

    def search(self, graph_id: str, query: str, scope: str = "all",
               limit: int = 10) -> SearchResult: ...

    def detect_communities(self, graph_id: str,
                           algorithm: str = "louvain") -> List[Community]: ...

    def snapshot(self, graph_id: str) -> bytes: ...
    def restore(self, data: bytes) -> str: ...


class TextStore(Protocol):
    """Full-text searchable storage for agent-generated content."""

    def add_text(self, graph_id: str, agent_id: str, content: str,
                 action_type: str, metadata: dict,
                 tick: int, timestamp: datetime) -> str: ...
    def search_text(self, graph_id: str, query: str,
                    limit: int = 20) -> List[dict]: ...
    def get_by_agent(self, graph_id: str, agent_id: str,
                     limit: int = 50) -> List[dict]: ...
    def get_by_time_range(self, graph_id: str, start: datetime,
                          end: datetime) -> List[dict]: ...
    def get_active_facts(self, graph_id: str) -> List[dict]: ...
    def get_historical_facts(self, graph_id: str) -> List[dict]: ...
