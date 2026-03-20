"""
Search tools replacing ZepTools (InsightForge, PanoramaSearch, QuickSearch).

Uses NetworkXGraphStore for graph operations and SQLite FTS5 for text search.
"""

from typing import Optional, List, Dict, Any

from .graph_store import EntityNode, RelationEdge, SearchResult, GraphStore, TextStore


class SearchTools:
    """Search engine providing InsightForge, PanoramaSearch, QuickSearch."""

    def __init__(self, graph_id: str, store,
                 llm_client=None):
        self.graph_id = graph_id
        self.store = store
        self.llm_client = llm_client

    def quick_search(self, query: str, limit: int = 20) -> SearchResult:
        """Fast keyword search over text content and graph entities."""
        # store.search(scope="all") combines graph keyword + FTS5 text search
        graph_result = self.store.search(self.graph_id, query, scope="all", limit=limit)

        # Deduplicate facts
        seen = set()
        unique_facts = []
        for f in graph_result.facts:
            if f and f not in seen:
                seen.add(f)
                unique_facts.append(f)

        return SearchResult(
            nodes=graph_result.nodes,
            edges=graph_result.edges,
            facts=unique_facts[:limit],
        )

    def panorama_search(self, query: str, limit: int = 100) -> SearchResult:
        """Full graph panorama with temporal classification."""
        all_entities = self.store.list_entities(self.graph_id, limit=limit)
        all_relations = self.store.list_relations(self.graph_id, limit=limit)

        # Classify by temporal status
        active_edges = [e for e in all_relations if e.expired_at is None and e.invalid_at is None]
        historical_edges = [e for e in all_relations if e.expired_at is not None or e.invalid_at is not None]

        facts = []
        for e in active_edges:
            if e.fact:
                facts.append(f"[Active] {e.fact}")
        for e in historical_edges:
            if e.fact:
                facts.append(f"[Historical] {e.fact}")

        # Add entity summaries
        for entity in all_entities:
            if entity.summary:
                facts.append(f"{entity.name}: {entity.summary}")

        return SearchResult(
            nodes=all_entities[:limit],
            edges=active_edges + historical_edges,
            facts=facts[:limit],
        )

    def insight_forge(self, query: str, limit: int = 30) -> SearchResult:
        """Deep multi-query analysis with LLM sub-question decomposition."""
        # Step 1: Decompose query into sub-questions via LLM
        sub_questions = self._decompose_query(query)

        # Step 2: Search each sub-question + original query
        all_facts = []
        all_nodes = []
        all_edges = []
        queries = [query] + sub_questions

        for q in queries:
            result = self.quick_search(q, limit=10)
            all_facts.extend(result.facts)
            all_nodes.extend(result.nodes)
            all_edges.extend(result.edges)

        # Step 3: Build relationship chains from discovered entities
        seen_edge_uuids = {e.uuid for e in all_edges}
        for node in all_nodes:
            edges = self.store.get_entity_edges(self.graph_id, node.uuid)
            for edge in edges:
                if edge.fact and edge.fact not in all_facts:
                    all_facts.append(edge.fact)
                if edge.uuid not in seen_edge_uuids:
                    seen_edge_uuids.add(edge.uuid)
                    all_edges.append(edge)

        # Deduplicate
        seen_facts = set()
        unique_facts = []
        for f in all_facts:
            if f and f not in seen_facts:
                seen_facts.add(f)
                unique_facts.append(f)

        seen_nodes = set()
        unique_nodes = []
        for n in all_nodes:
            if n.uuid not in seen_nodes:
                seen_nodes.add(n.uuid)
                unique_nodes.append(n)

        seen_edge_uuids_final = set()
        unique_edges = []
        for e in all_edges:
            if e.uuid not in seen_edge_uuids_final:
                seen_edge_uuids_final.add(e.uuid)
                unique_edges.append(e)

        return SearchResult(
            nodes=unique_nodes[:limit],
            edges=unique_edges[:limit],
            facts=unique_facts[:limit],
        )

    def _decompose_query(self, query: str) -> List[str]:
        """Use LLM to break a complex query into sub-questions."""
        if not self.llm_client:
            # Fallback: split by common Chinese punctuation
            return [query]

        messages = [
            {"role": "system", "content": (
                "You are a search query analyzer. Decompose the user's complex question into 3-5 simpler, more specific sub-questions. "
                "Return JSON format: {\"sub_questions\": [\"question1\", \"question2\", ...]}"
            )},
            {"role": "user", "content": query},
        ]
        try:
            result = self.llm_client.chat_json(messages, temperature=0.3)
            return result.get("sub_questions", [query])
        except Exception:
            return [query]
