"""Zep Cloud GraphStore. Moved call patterns stay on this side of the seam."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from zep_cloud import BatchAddItem, EntityEdgeSourceTarget, NotFoundError

from ..utils.locale import t
from ..utils.ontology import (
    MAX_ONTOLOGY_TYPES,
    RESERVED_ONTOLOGY_ATTRIBUTE_NAMES,
    normalize_ontology_attributes,
    normalize_ontology_source_targets,
)
from ..utils.zep import (
    ZEP_INGESTION_WAIT_TIMEOUT_SECONDS,
    call_zep_read_with_retry,
    is_retryable_zep_error,
    normalize_zep_search_limit,
    normalize_zep_search_query,
)
from ..utils.zep_paging import fetch_all_edges, fetch_all_nodes
from .store import (
    GraphEdge,
    GraphNode,
    GraphNotFoundError,
    ProgressCallback,
    SearchResult,
    TextEpisode,
)

_RANKING_TO_RERANKER = {
    "relevance": "cross_encoder",
    "fusion": "rrf",
}


@dataclass(frozen=True)
class BatchSubmission:
    """Durable identity for one Zep ingestion operation."""

    batch_id: str | None
    operation_id: str | None
    episode_uuids: list[str]
    item_count: int

    @property
    def episode_ids(self) -> list[str]:
        return list(self.episode_uuids)


def _episode_list(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def node_from_zep(node: Any) -> GraphNode:
    return GraphNode(
        uuid=getattr(node, "uuid_", None) or getattr(node, "uuid", None) or "",
        name=getattr(node, "name", None),
        labels=getattr(node, "labels", None),
        summary=getattr(node, "summary", None),
        attributes=getattr(node, "attributes", None),
        created_at=getattr(node, "created_at", None),
    )


def edge_from_zep(edge: Any) -> GraphEdge:
    return GraphEdge(
        uuid=getattr(edge, "uuid_", None) or getattr(edge, "uuid", None) or "",
        name=getattr(edge, "name", None) or "",
        fact=getattr(edge, "fact", None) or "",
        source_node_uuid=getattr(edge, "source_node_uuid", None) or "",
        target_node_uuid=getattr(edge, "target_node_uuid", None) or "",
        attributes=getattr(edge, "attributes", None),
        created_at=getattr(edge, "created_at", None),
        valid_at=getattr(edge, "valid_at", None),
        invalid_at=getattr(edge, "invalid_at", None),
        expired_at=getattr(edge, "expired_at", None),
        episodes=_episode_list(
            getattr(edge, "episodes", None) or getattr(edge, "episode_ids", None)
        ),
        fact_type=getattr(edge, "fact_type", None),
    )


class ZepGraphStore:
    def __init__(self, client: Any) -> None:
        self.client = client

    def create_graph(
        self,
        name: str,
        *,
        graph_id: str | None = None,
        graph_id_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Create a graph with a caller-durable ID and reconcile lost replies."""

        graph_id = graph_id or f"mirofish_{uuid.uuid4().hex[:16]}"
        if graph_id_callback:
            graph_id_callback(graph_id)

        try:
            self.client.graph.create(
                graph_id=graph_id,
                name=name,
                description="MiroFish Social Simulation Graph",
            )
        except Exception as error:
            if not is_retryable_zep_error(error):
                raise
            reconciliation_error = None
            for attempt in range(3):
                try:
                    call_zep_read_with_retry(
                        lambda: self.client.graph.get(graph_id),
                        operation_name=f"reconcile graph create {graph_id}",
                    )
                    reconciliation_error = None
                    break
                except NotFoundError as not_found:
                    reconciliation_error = not_found
                    if attempt < 2:
                        time.sleep(attempt + 1)
                except Exception as read_error:
                    reconciliation_error = read_error
                    break
            if reconciliation_error is not None:
                raise error from reconciliation_error

        return graph_id

    @staticmethod
    def build_operation_id(graph_id: str, chunks: list[str]) -> str:
        payload_hash = hashlib.sha256("\0".join(chunks).encode("utf-8")).hexdigest()
        return hashlib.sha256(
            f"{graph_id}:{payload_hash}".encode("utf-8")
        ).hexdigest()

    def _find_batch_by_operation_id(
        self,
        graph_id: str,
        operation_id: str,
        *,
        max_attempts: int = 3,
    ) -> Any | None:
        """Find one server-created batch after an ambiguous create reply."""

        for attempt in range(1, max_attempts + 1):
            matches: list[Any] = []
            cursor: int | None = None
            seen_cursors: set[int] = set()
            while True:
                page = call_zep_read_with_retry(
                    lambda: self.client.batch.list(limit=100, cursor=cursor),
                    operation_name=f"reconcile batch create {operation_id}",
                )
                for batch in getattr(page, "batches", None) or []:
                    metadata = getattr(batch, "metadata", None) or {}
                    if (
                        metadata.get("mirofish_operation_id") == operation_id
                        and metadata.get("graph_id") == graph_id
                    ):
                        matches.append(batch)
                next_cursor = getattr(page, "next_cursor", None)
                if next_cursor is None:
                    break
                if next_cursor == cursor or next_cursor in seen_cursors:
                    raise RuntimeError("Zep batch list cursor did not advance")
                seen_cursors.add(next_cursor)
                cursor = next_cursor

            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple Zep batches match operation {operation_id}; refusing ambiguity"
                )
            if matches:
                return matches[0]
            if attempt < max_attempts:
                time.sleep(attempt)
        return None

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        """设置图谱本体（公开方法）"""
        import warnings
        from typing import Optional
        from pydantic import Field
        from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel

        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in RESERVED_ONTOLOGY_ATTRIBUTE_NAMES:
                return f"entity_{attr_name}"
            return attr_name

        entity_types = {}
        for entity_def in ontology.get("entity_types", [])[:MAX_ONTOLOGY_TYPES]:
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")

            attrs = {"__doc__": description}
            annotations = {}

            for normalized in normalize_ontology_attributes(
                entity_def.get("attributes", [])
            ):
                attr_name = safe_attr_name(normalized["name"])
                attr_desc = normalized["description"]
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[EntityText]

            attrs["__annotations__"] = annotations

            entity_class = type(name, (EntityModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        edge_definitions = {}
        for edge_def in ontology.get("edge_types", [])[:MAX_ONTOLOGY_TYPES]:
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")

            attrs = {"__doc__": description}
            annotations = {}

            for normalized in normalize_ontology_attributes(
                edge_def.get("attributes", [])
            ):
                attr_name = safe_attr_name(normalized["name"])
                attr_desc = normalized["description"]
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[str]

            attrs["__annotations__"] = annotations

            class_name = "".join(word.capitalize() for word in name.split("_"))
            edge_class = type(class_name, (EdgeModel,), attrs)
            edge_class.__doc__ = description

            source_targets = []
            for st in normalize_ontology_source_targets(
                edge_def.get("source_targets", [])
            ):
                source_targets.append(
                    EntityEdgeSourceTarget(
                        source=st.get("source", "Entity"),
                        target=st.get("target", "Entity"),
                    )
                )

            if source_targets:
                edge_definitions[name] = (edge_class, source_targets)

        if entity_types or edge_definitions:
            self.client.graph.set_ontology(
                graph_ids=[graph_id],
                entities=entity_types,
                edges=edge_definitions if edge_definitions else None,
            )

    def add_text_batches(
        self,
        graph_id: str,
        chunks: list[str],
        batch_size: int = 350,
        progress_callback: Callable | None = None,
        batch_created_callback: Callable[[str | None, str], None] | None = None,
    ) -> BatchSubmission:
        """Submit document chunks through Zep's current Batch API."""

        if not graph_id:
            raise ValueError("graph_id is required")
        self.validate_batch_chunks(chunks, batch_size=batch_size)

        total_chunks = len(chunks)
        operation_id = self.build_operation_id(graph_id, chunks)
        if batch_created_callback:
            batch_created_callback(None, operation_id)

        try:
            batch = self.client.batch.create(
                metadata={
                    "mirofish_operation_id": operation_id,
                    "graph_id": graph_id,
                    "chunk_count": total_chunks,
                }
            )
        except Exception as error:
            if not is_retryable_zep_error(error):
                raise
            batch = self._find_batch_by_operation_id(graph_id, operation_id)
            if batch is None:
                raise RuntimeError(
                    "Zep batch creation is unconfirmed and no matching operation was found"
                ) from error
        batch_id = getattr(batch, "batch_id", None)
        if not batch_id:
            raise RuntimeError("Zep Batch API returned no batch_id")
        if batch_created_callback:
            batch_created_callback(batch_id, operation_id)

        episode_uuids: list[str] = []
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    t(
                        "progress.sendingBatch",
                        current=batch_num,
                        total=total_batches,
                        chunks=len(batch_chunks),
                    ),
                    progress,
                )

            items = [
                BatchAddItem(
                    type="graph_episode",
                    graph_id=graph_id,
                    data=chunk,
                    data_type="text",
                    source_description="MiroFish source document chunk",
                    metadata={
                        "mirofish_operation_id": operation_id,
                        "chunk_index": i + offset,
                        "chunk_sha256": hashlib.sha256(
                            chunk.encode("utf-8")
                        ).hexdigest(),
                    },
                )
                for offset, chunk in enumerate(batch_chunks)
            ]

            expected_item_count = i + len(items)
            try:
                item_details = self.client.batch.add(
                    batch_id=batch_id,
                    items=items,
                )
            except Exception as e:
                if progress_callback:
                    progress_callback(
                        t("progress.batchFailed", batch=batch_num, error=str(e)),
                        0,
                    )
                if is_retryable_zep_error(e):
                    recovered_items = self._reconcile_batch_item_count(
                        batch_id,
                        expected_item_count,
                    )
                    recovered_indexes = {
                        getattr(item, "sequence_index", None)
                        for item in recovered_items
                    }
                    if (
                        len(recovered_items) == expected_item_count
                        and recovered_indexes == set(range(expected_item_count))
                    ):
                        item_details = recovered_items[i:expected_item_count]
                    else:
                        raise RuntimeError(
                            f"Zep batch {batch_id} item submission is unconfirmed; "
                            "the draft was not processed or replayed"
                        ) from e
                else:
                    raise RuntimeError(
                        f"Zep batch {batch_id} item submission failed"
                    ) from e

            if len(item_details or []) != len(items):
                recovered_items = self._reconcile_batch_item_count(
                    batch_id,
                    expected_item_count,
                )
                recovered_indexes = {
                    getattr(item, "sequence_index", None)
                    for item in recovered_items
                }
                if (
                    len(recovered_items) == expected_item_count
                    and recovered_indexes == set(range(expected_item_count))
                ):
                    item_details = recovered_items[i:expected_item_count]
                else:
                    raise RuntimeError(
                        f"Zep batch {batch_id} acknowledged {len(item_details or [])} "
                        f"of {len(items)} items"
                    )
            for item in item_details:
                episode_uuid = getattr(item, "episode_uuid", None)
                if episode_uuid:
                    episode_uuids.append(episode_uuid)

        try:
            self.client.batch.process(batch_id=batch_id)
        except Exception as error:
            summary = call_zep_read_with_retry(
                lambda: self.client.batch.get(batch_id=batch_id),
                operation_name=f"reconcile batch {batch_id}",
            )
            if getattr(summary, "status", None) in {None, "draft"}:
                raise RuntimeError(
                    f"Zep batch {batch_id} processing is unconfirmed"
                ) from error

        return BatchSubmission(
            batch_id=batch_id,
            operation_id=operation_id,
            episode_uuids=episode_uuids,
            item_count=total_chunks,
        )

    @staticmethod
    def validate_batch_chunks(chunks: list[str], *, batch_size: int = 350) -> None:
        """Validate every Batch API limit before the first Cloud mutation."""

        if not chunks:
            raise ValueError("At least one text chunk is required")
        if not 1 <= batch_size <= 350:
            raise ValueError("batch_size must be between 1 and 350")
        if len(chunks) > 50_000:
            raise ValueError("A Zep batch cannot contain more than 50,000 items")
        oversized = [index for index, chunk in enumerate(chunks) if len(chunk) > 10_000]
        if oversized:
            raise ValueError(
                f"Zep batch item exceeds 10,000 characters at chunk {oversized[0]}"
            )

    def _list_batch_items(self, batch_id: str) -> list[Any]:
        items: list[Any] = []
        cursor: int | None = None
        seen_cursors: set[int] = set()
        while True:
            page = call_zep_read_with_retry(
                lambda: self.client.batch.list_items(
                    batch_id=batch_id,
                    limit=100,
                    cursor=cursor,
                ),
                operation_name=f"list batch items {batch_id}",
            )
            items.extend(getattr(page, "items", None) or [])
            next_cursor = getattr(page, "next_cursor", None)
            if next_cursor is None:
                break
            if next_cursor == cursor or next_cursor in seen_cursors:
                raise RuntimeError(f"Zep batch {batch_id} item cursor did not advance")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return items

    def _reconcile_batch_item_count(
        self,
        batch_id: str,
        expected_item_count: int,
        *,
        max_attempts: int = 3,
    ) -> list[Any]:
        """Allow a short propagation window after an ambiguous add reply."""

        items: list[Any] = []
        for attempt in range(1, max_attempts + 1):
            items = self._list_batch_items(batch_id)
            if len(items) >= expected_item_count:
                return items
            if attempt < max_attempts:
                time.sleep(attempt)
        return items

    def get_batch_summary(self, batch_id: str) -> Any:
        """Read a persisted batch identity for restart reconciliation."""

        return call_zep_read_with_retry(
            lambda: self.client.batch.get(batch_id=batch_id),
            operation_name=f"get batch {batch_id}",
        )

    def _wait_for_batch(
        self,
        submission: BatchSubmission,
        progress_callback: Callable | None = None,
        timeout: int | None = None,
    ) -> list[str]:
        """Wait for a Batch API terminal state and validate every item."""

        timeout = timeout or ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
        start_time = time.time()
        terminal_states = {"succeeded", "partial", "failed", "invalid", "canceled"}

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Zep batch {submission.batch_id} did not finish within {timeout}s"
                )

            summary = call_zep_read_with_retry(
                lambda: self.client.batch.get(batch_id=submission.batch_id),
                operation_name=f"poll batch {submission.batch_id}",
            )
            status = getattr(summary, "status", None)
            progress = getattr(summary, "progress", None)
            percent = float(getattr(progress, "percent_complete", 0) or 0) / 100
            if progress_callback:
                completed = int(getattr(progress, "succeeded_items", 0) or 0)
                progress_callback(
                    t(
                        "progress.zepProcessing",
                        completed=completed,
                        total=submission.item_count,
                        pending=max(submission.item_count - completed, 0),
                        elapsed=int(time.time() - start_time),
                    ),
                    min(max(percent, 0.0), 1.0),
                )

            if status in terminal_states:
                break
            time.sleep(3)

        items = self._list_batch_items(submission.batch_id)
        if status != "succeeded":
            failed_items = [
                item for item in items
                if getattr(item, "status", None) not in {"succeeded", "skipped"}
            ]
            first_error = getattr(failed_items[0], "error", None) if failed_items else None
            raise RuntimeError(
                f"Zep batch {submission.batch_id} ended as {status}; "
                f"failed_items={len(failed_items)}; first_error={first_error}"
            )
        if len(items) != submission.item_count:
            raise RuntimeError(
                f"Zep batch {submission.batch_id} contains {len(items)} items, "
                f"expected {submission.item_count}"
            )

        ordered_items = sorted(
            items,
            key=lambda item: getattr(item, "sequence_index", 0) or 0,
        )
        episode_uuids: list[str] = []
        for item in ordered_items:
            item_status = getattr(item, "status", None)
            episode_uuid = getattr(item, "episode_uuid", None)
            source_uuid = getattr(item, "source_uuid", None)
            if item_status != "succeeded" or not episode_uuid:
                raise RuntimeError(
                    f"Zep batch {submission.batch_id} returned an incomplete item"
                )
            if source_uuid and source_uuid != episode_uuid:
                raise RuntimeError(
                    f"Zep batch {submission.batch_id} returned mismatched episode UUIDs"
                )
            episode_uuids.append(episode_uuid)

        if progress_callback:
            progress_callback(
                t(
                    "progress.processingComplete",
                    completed=len(episode_uuids),
                    total=submission.item_count,
                ),
                1.0,
            )
        return episode_uuids

    def _wait_for_episodes(
        self,
        episode_uuids: list[str],
        progress_callback: Callable | None = None,
        timeout: int = ZEP_INGESTION_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        """等待所有 episode 处理完成（通过查询每个 episode 的 processed 状态）"""
        if not episode_uuids:
            if progress_callback:
                progress_callback(t("progress.noEpisodesWait"), 1.0)
            return

        start_time = time.time()
        pending_episodes = set(episode_uuids)
        completed_count = 0
        total_episodes = len(episode_uuids)

        if progress_callback:
            progress_callback(t("progress.waitingEpisodes", count=total_episodes), 0)

        while pending_episodes:
            if time.time() - start_time > timeout:
                if progress_callback:
                    progress_callback(
                        t(
                            "progress.episodesTimeout",
                            completed=completed_count,
                            total=total_episodes,
                        ),
                        completed_count / total_episodes,
                    )
                raise TimeoutError(
                    f"Zep episode processing timed out with "
                    f"{len(pending_episodes)} episode(s) still pending"
                )

            for ep_uuid in list(pending_episodes):
                episode = call_zep_read_with_retry(
                    lambda: self.client.graph.episode.get(uuid_=ep_uuid),
                    operation_name=f"poll episode {ep_uuid}",
                )
                is_processed = getattr(episode, "processed", False)

                if is_processed:
                    pending_episodes.remove(ep_uuid)
                    completed_count += 1

            elapsed = int(time.time() - start_time)
            if progress_callback:
                progress_callback(
                    t(
                        "progress.zepProcessing",
                        completed=completed_count,
                        total=total_episodes,
                        pending=len(pending_episodes),
                        elapsed=elapsed,
                    ),
                    completed_count / total_episodes if total_episodes > 0 else 0,
                )

            if pending_episodes:
                time.sleep(3)

        if progress_callback:
            progress_callback(
                t(
                    "progress.processingComplete",
                    completed=completed_count,
                    total=total_episodes,
                ),
                1.0,
            )

    def add_text_episodes(
        self,
        graph_id: str,
        episodes: list[TextEpisode],
        *,
        durable: bool,
        batch_size: int = 350,
        on_submitted: Callable[[str | None, str], None] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BatchSubmission:
        if durable:
            return self.add_text_batches(
                graph_id,
                [episode.text for episode in episodes],
                batch_size=batch_size,
                progress_callback=on_progress,
                batch_created_callback=on_submitted,
            )

        episode_uuids: list[str] = []
        for episode in episodes:
            created = self.client.graph.add(
                graph_id=graph_id,
                type="text",
                data=episode.text,
                created_at=episode.created_at,
                source_description=episode.source,
                metadata=episode.metadata,
            )
            episode_uuid = getattr(created, "uuid_", None) or getattr(created, "uuid", None)
            if not episode_uuid:
                raise RuntimeError("Zep graph.add returned no episode UUID")
            episode_uuids.append(str(episode_uuid))
        return BatchSubmission(
            batch_id=None,
            operation_id=None,
            episode_uuids=episode_uuids,
            item_count=len(episode_uuids),
        )

    def wait_until_processed(
        self,
        handle: BatchSubmission,
        *,
        deadline: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[str]:
        if handle.batch_id:
            remaining = None
            if deadline is not None:
                remaining = max(0, int(deadline - time.time()))
            self._wait_for_batch(handle, on_progress, remaining)
            return list(handle.episode_ids)
        if deadline is not None:
            self._wait_for_episodes_until(
                list(handle.episode_ids),
                deadline,
                on_progress,
            )
            return list(handle.episode_ids)
        self._wait_for_episodes(list(handle.episode_ids), on_progress)
        return list(handle.episode_ids)

    def _wait_for_episodes_until(
        self,
        episode_uuids: list[str],
        deadline: float,
        progress_callback: Callable | None = None,
    ) -> None:
        """Poll episodes until ``deadline`` (absolute ``time.time()``)."""

        pending_episodes = set(episode_uuids)
        while pending_episodes:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Zep simulation ingestion timed out with {len(pending_episodes)} "
                    "episode(s) pending"
                )
            for ep_uuid in list(pending_episodes):
                episode = call_zep_read_with_retry(
                    lambda: self.client.graph.episode.get(uuid_=ep_uuid),
                    operation_name=f"poll simulation episode {ep_uuid}",
                )
                if getattr(episode, "processed", False):
                    pending_episodes.remove(ep_uuid)
            if pending_episodes:
                time.sleep(3)

    def list_nodes(self, graph_id: str) -> list[GraphNode]:
        return [node_from_zep(node) for node in fetch_all_nodes(self.client, graph_id)]

    def list_edges(self, graph_id: str) -> list[GraphEdge]:
        return [edge_from_zep(edge) for edge in fetch_all_edges(self.client, graph_id)]

    def get_node(self, node_uuid: str) -> GraphNode:
        try:
            node = call_zep_read_with_retry(
                lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"get node {node_uuid}",
            )
        except NotFoundError as exc:
            raise GraphNotFoundError(str(exc)) from exc
        return node_from_zep(node)

    def get_node_edges(self, node_uuid: str) -> list[GraphEdge]:
        try:
            raw = call_zep_read_with_retry(
                lambda: self.client.graph.node.get_edges(node_uuid=node_uuid),
                operation_name=f"get node edges {node_uuid}",
            )
        except NotFoundError as exc:
            raise GraphNotFoundError(str(exc)) from exc
        raw_edges = getattr(raw, "edges", None)
        if raw_edges is None:
            raw_edges = list(raw or [])
        return [edge_from_zep(edge) for edge in raw_edges]

    def search(
        self,
        graph_id: str,
        query: str,
        scope: Literal["edges", "nodes"],
        limit: int,
        *,
        ranking: Literal["relevance", "fusion"] = "relevance",
    ) -> SearchResult:
        reranker = _RANKING_TO_RERANKER[ranking]
        raw = call_zep_read_with_retry(
            lambda: self.client.graph.search(
                graph_id=graph_id,
                query=normalize_zep_search_query(query),
                limit=normalize_zep_search_limit(limit),
                scope=scope,
                reranker=reranker,
            ),
            operation_name=f"search graph {graph_id}",
        )
        return SearchResult(
            edges=[edge_from_zep(edge) for edge in (getattr(raw, "edges", None) or [])],
            nodes=[node_from_zep(node) for node in (getattr(raw, "nodes", None) or [])],
        )

    def delete_graph(self, graph_id: str) -> None:
        try:
            self.client.graph.delete(graph_id=graph_id)
        except NotFoundError as exc:
            raise GraphNotFoundError(str(exc)) from exc
