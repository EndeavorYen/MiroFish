"""In-memory Zep client for GraphStore contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from zep_cloud import NotFoundError


class _RawPage:
    def __init__(self, fetch, item_name: str) -> None:
        self._fetch = fetch
        self._item_name = item_name

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> Any:
        page = self._fetch(graph_id, **kwargs)
        return SimpleNamespace(data=list(getattr(page, self._item_name)), headers={})


class _NodeApi:
    def __init__(self, client: "FakeZepClient") -> None:
        self._client = client
        self.with_raw_response = _RawPage(self.get_by_graph_id, "nodes")

    def get(self, uuid_: str) -> Any:
        node = self._client.nodes_by_uuid.get(uuid_)
        if node is None:
            raise NotFoundError("node not found")
        return node

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> Any:
        self._client.node_list_calls.append({"graph_id": graph_id, **kwargs})
        nodes = [
            node for node in self._client.nodes
            if getattr(node, "graph_id", None) == graph_id
        ]
        return SimpleNamespace(nodes=nodes, next_cursor=None, total_count=len(nodes))

    def get_edges(self, node_uuid: str) -> Any:
        if node_uuid not in self._client.nodes_by_uuid:
            raise NotFoundError("node not found")
        edges = [
            edge for edge in self._client.edges
            if edge.source_node_uuid == node_uuid
        ]
        return SimpleNamespace(edges=edges, next_cursor=None)


class _EdgeApi:
    def __init__(self, client: "FakeZepClient") -> None:
        self._client = client
        self.with_raw_response = _RawPage(self.get_by_graph_id, "edges")

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> Any:
        self._client.edge_list_calls.append({"graph_id": graph_id, **kwargs})
        edges = [
            edge for edge in self._client.edges
            if getattr(edge, "graph_id", None) == graph_id
        ]
        return SimpleNamespace(edges=edges, next_cursor=None, total_count=len(edges))


class _EpisodeApi:
    def __init__(self, client: "FakeZepClient") -> None:
        self._client = client

    def get(self, uuid_: str) -> Any:
        episode = self._client.episodes.get(uuid_)
        if episode is None:
            raise NotFoundError("episode not found")
        episode.processed = True
        return episode

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> Any:
        episodes = [
            episode for episode in self._client.episodes.values()
            if episode.graph_id == graph_id
        ]
        return SimpleNamespace(episodes=episodes, next_cursor=None)


class FakeGraph:
    def __init__(self, client: "FakeZepClient") -> None:
        self._client = client
        self.search_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.node = _NodeApi(client)
        self.edge = _EdgeApi(client)
        self.episode = _EpisodeApi(client)

    def create(self, **kwargs: Any) -> Any:
        graph_id = kwargs["graph_id"]
        self._client.graphs[graph_id] = kwargs
        return SimpleNamespace(graph_id=graph_id)

    def get(self, graph_id: str) -> Any:
        if graph_id not in self._client.graphs:
            raise NotFoundError("graph not found")
        return SimpleNamespace(graph_id=graph_id, **self._client.graphs[graph_id])

    def delete(self, graph_id: str) -> None:
        if graph_id not in self._client.graphs:
            raise NotFoundError("graph not found")
        del self._client.graphs[graph_id]
        self._client.nodes = [
            node for node in self._client.nodes
            if getattr(node, "graph_id", None) != graph_id
        ]
        self._client.nodes_by_uuid = {
            node.uuid_: node for node in self._client.nodes
        }
        self._client.edges = [
            edge for edge in self._client.edges
            if getattr(edge, "graph_id", None) != graph_id
        ]

    def add(self, **kwargs: Any) -> Any:
        self.add_calls.append(kwargs)
        episode_uuid = f"ep-{len(self._client.episodes) + 1}"
        episode = SimpleNamespace(
            uuid_=episode_uuid,
            graph_id=kwargs.get("graph_id"),
            processed=True,
            content=kwargs.get("data"),
        )
        self._client.episodes[episode_uuid] = episode
        return episode

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        edge = SimpleNamespace(
            uuid_="e1",
            name="knows",
            fact="Alice met Bob",
            source_node_uuid="n-alice",
            target_node_uuid="n-bob",
            attributes={},
            created_at=None,
            valid_at=None,
            invalid_at=None,
            expired_at=None,
            episodes=[],
        )
        return SimpleNamespace(edges=[edge], nodes=[])

    def set_ontology(self, **kwargs: Any) -> None:
        self._client.ontology_calls.append(kwargs)


class FakeBatch:
    def __init__(self, client: "FakeZepClient") -> None:
        self._client = client
        self.create_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.process_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        batch_id = f"batch-{len(self.create_calls)}"
        batch = SimpleNamespace(
            batch_id=batch_id,
            status="pending",
            metadata=kwargs.get("metadata") or {},
            item_count=0,
            items=[],
            progress=None,
        )
        self._client.batches[batch_id] = batch
        return batch

    def add(self, **kwargs: Any) -> list[Any]:
        self.add_calls.append(kwargs)
        batch = self._client.batches[kwargs["batch_id"]]
        items = []
        for offset, item in enumerate(kwargs.get("items") or []):
            episode_uuid = f"ep-batch-{len(self._client.episodes) + 1}"
            episode = SimpleNamespace(
                uuid_=episode_uuid,
                graph_id=getattr(item, "graph_id", None),
                processed=True,
            )
            self._client.episodes[episode_uuid] = episode
            sequence = batch.item_count + offset
            recorded = SimpleNamespace(
                sequence_index=sequence,
                uuid_=episode_uuid,
                episode_uuid=episode_uuid,
                source_uuid=episode_uuid,
                status="succeeded",
            )
            batch.items.append(recorded)
            items.append(recorded)
        batch.item_count += len(items)
        return items

    def process(self, **kwargs: Any) -> Any:
        self.process_calls.append(kwargs)
        batch = self._client.batches[kwargs["batch_id"]]
        batch.status = "succeeded"
        return batch

    def get(self, batch_id: str) -> Any:
        batch = self._client.batches.get(batch_id)
        if batch is None:
            raise NotFoundError("batch not found")
        batch.status = "succeeded"
        batch.progress = SimpleNamespace(
            percent_complete=100,
            succeeded_items=batch.item_count,
        )
        return batch

    def list(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            batches=list(self._client.batches.values()),
            next_cursor=None,
        )

    def list_items(self, batch_id: str, **kwargs: Any) -> Any:
        batch = self._client.batches.get(batch_id)
        items = [] if batch is None else list(batch.items)
        return SimpleNamespace(items=items, next_cursor=None)


class FakeZepClient:
    def __init__(self) -> None:
        self.graphs: dict[str, dict[str, Any]] = {}
        self.nodes: list[Any] = []
        self.nodes_by_uuid: dict[str, Any] = {}
        self.edges: list[Any] = []
        self.episodes: dict[str, Any] = {}
        self.batches: dict[str, Any] = {}
        self.node_list_calls: list[dict[str, Any]] = []
        self.edge_list_calls: list[dict[str, Any]] = []
        self.ontology_calls: list[dict[str, Any]] = []
        self.graph = FakeGraph(self)
        self.batch = FakeBatch(self)
