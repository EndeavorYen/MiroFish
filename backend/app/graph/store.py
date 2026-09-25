"""Domain graph-store interface. Zep types stay out of this module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


class GraphNotFoundError(LookupError):
    """Raised when a graph or node does not exist."""


@dataclass(frozen=True)
class TextEpisode:
    text: str
    created_at: str | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GraphNode:
    uuid: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    created_at: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict[str, Any]
    created_at: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None
    episodes: list[str] = field(default_factory=list)
    fact_type: str | None = None


@dataclass(frozen=True)
class SearchResult:
    edges: list[GraphEdge]
    nodes: list[GraphNode]


class IngestionHandle(Protocol):
    @property
    def episode_ids(self) -> list[str]: ...


@dataclass(frozen=True)
class EpisodeHandle:
    """Backend-neutral handle for episodes whose ids are already known."""

    episode_ids: list[str]


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class FactNode:
    """A node addressed by a stable caller key (``post:twitter:17``).

    ``name`` is also used to attach to an existing entity node with the same
    normalized name (a simulated agent onto its source entity).
    """

    key: str
    name: str
    label: str
    summary: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredFact:
    """One edge written without text extraction.

    ``key`` is the idempotency key: writing the same key again changes
    nothing. ``mentions`` are entity names found in the target's text; each
    existing entity node with that name gets a ``MENTIONS`` edge.
    """

    key: str
    source: FactNode
    relation: str
    target: FactNode
    fact: str
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    extra_nodes: tuple[FactNode, ...] = ()
    extra_edges: tuple[tuple[str, str, str], ...] = ()  # (source key, relation, target key)
    mentions: tuple[str, ...] = ()


@runtime_checkable
class StructuredFactStore(Protocol):
    """Optional capability: write structured facts with no LLM (#8)."""

    def add_structured_facts(self, graph_id: str, facts: list[StructuredFact]) -> list[str]: ...


class GraphStore(Protocol):
    def create_graph(
        self,
        name: str,
        *,
        graph_id: str | None = None,
        graph_id_callback: Callable[[str], None] | None = None,
    ) -> str: ...

    def delete_graph(self, graph_id: str) -> None: ...

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None: ...

    def add_text_episodes(
        self,
        graph_id: str,
        episodes: list[TextEpisode],
        *,
        durable: bool,
        batch_size: int = 350,
        on_submitted: Callable[[str | None, str], None] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionHandle: ...

    def wait_until_processed(
        self,
        handle: IngestionHandle,
        *,
        deadline: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[str]: ...

    def list_nodes(self, graph_id: str) -> list[GraphNode]: ...

    def list_edges(self, graph_id: str) -> list[GraphEdge]: ...

    def get_node(self, node_uuid: str) -> GraphNode: ...

    def get_node_edges(self, node_uuid: str) -> list[GraphEdge]: ...

    def search(
        self,
        graph_id: str,
        query: str,
        scope: Literal["edges", "nodes"],
        limit: int,
        *,
        ranking: Literal["relevance", "fusion"] = "relevance",
    ) -> SearchResult: ...


def get_graph_store(*, backend: str | None = None, api_key: str | None = None) -> GraphStore:
    """Return the configured graph store.

    ``zep_store`` and the Zep client are imported only for the zep backend;
    the local backend never needs ``ZEP_API_KEY``.
    """

    from app.config import Config

    selected = Config.GRAPH_BACKEND if backend is None else backend.strip().lower()
    if selected == "local":
        from app.graph.local_store import build_local_graph_store

        return build_local_graph_store()
    if selected != "zep":
        raise ValueError(f"未知的 GRAPH_BACKEND: {selected}")

    from app.graph.zep_store import ZepGraphStore
    from app.utils.zep import get_zep_client

    key = api_key if api_key is not None else Config.ZEP_API_KEY
    if not key:
        raise ValueError("ZEP_API_KEY 未配置")
    return ZepGraphStore(get_zep_client(key))
