# MiroFish De-Zep Migration: NetworkX + SQLite Hybrid Architecture

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Replace all Zep Cloud dependencies with self-hosted NetworkX + SQLite architecture

---

## 1. Problem Statement

MiroFish currently depends on Zep Cloud (v3.13.0) for three critical roles across 16 source files:

1. **Ingestion Pipeline** (Step 1): Upload documents → auto-extract entities/relationships → build knowledge graph
2. **Runtime Memory** (Step 3): During simulation, convert agent actions to narratives → Zep auto-builds graph
3. **Retrieval Engine** (Step 4-5): Semantic/keyword search for report generation and agent interviews

This external dependency creates:
- **Data privacy risk**: Sensitive simulation scenarios sent to third-party cloud
- **Latency bottleneck**: Every agent action requires HTTP round-trip to Zep API
- **Cost scaling**: API calls multiply with Monte Carlo parallel simulations
- **Vendor lock-in**: 11 files directly call Zep SDK with no abstraction layer
- **Missing capability**: Zep has no community detection (critical for wargaming faction analysis)

## 2. Design Goals

1. **Zero external service dependencies** — Everything runs locally (Python stdlib + NetworkX)
2. **Complete Zep role replacement** — All three roles (ingestion, runtime, retrieval) fully covered
3. **New capability: Community detection** — Louvain/Leiden algorithm for faction identification
4. **Monte Carlo ready** — Architecture supports isolated parallel simulation branches
5. **Gradual migration** — Abstraction layer enables file-by-file migration without breaking existing functionality
6. **LLM cost minimization** — Eliminate unnecessary LLM calls (especially in runtime memory)

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Abstraction Layer (Protocol/ABC)            │
│     GraphStore / SearchEngine / MemoryStore protocols        │
│     All 16 Zep-dependent files migrate to call these         │
├────────────────────────┬────────────────────────────────────┤
│   NetworkX Graph Engine │   SQLite Persistence + Search      │
│                        │                                    │
│  • Nodes = entities    │  • Agent posts/actions (text)      │
│  • Edges = relations   │  • FTS5 full-text index            │
│  • Louvain/Leiden      │  • Temporal columns                │
│    community detection │    (valid_at, expired_at, tick)    │
│  • Path/centrality     │  • Per-branch DB files             │
│  • deepcopy branching  │  • Query interface                 │
├────────────────────────┴────────────────────────────────────┤
│                    LLM Layer (minimal usage)                 │
│  • Step 1 only: document entity extraction (JSON output)    │
│  • Step 4 only: community summaries + query decomposition   │
│  • Step 3: ZERO LLM calls (structured events → direct edges)│
└─────────────────────────────────────────────────────────────┘
```

## 4. Detailed Design

### 4.1 Abstraction Layer (Layer 5)

Define Python protocols that match the operations currently performed against Zep. All business logic code will be refactored to call these protocols instead of Zep SDK directly.

```python
# backend/app/services/graph_store.py

from typing import Protocol, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EntityNode:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: dict
    related_edges: List[dict]
    related_nodes: List[dict]

@dataclass
class RelationEdge:
    uuid: str
    name: str
    fact: str  # human-readable description
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict
    created_at: datetime
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None

@dataclass
class SearchResult:
    nodes: List[EntityNode]
    edges: List[RelationEdge]
    facts: List[str]

@dataclass
class Community:
    id: int
    members: List[str]  # node UUIDs
    summary: Optional[str] = None

class GraphStore(Protocol):
    """Abstraction over graph storage and operations."""

    # Graph lifecycle
    def create_graph(self, graph_id: str, name: str, description: str) -> str: ...
    def delete_graph(self, graph_id: str) -> None: ...

    # Entity operations
    def add_entity(self, graph_id: str, name: str, entity_type: str,
                   summary: str, attributes: dict) -> str: ...
    def get_entity(self, uuid: str) -> EntityNode: ...
    def get_entity_edges(self, uuid: str) -> List[RelationEdge]: ...
    def list_entities(self, graph_id: str, limit: int,
                      cursor: Optional[str]) -> List[EntityNode]: ...

    # Relation operations
    def add_relation(self, graph_id: str, source_uuid: str, target_uuid: str,
                     relation_type: str, fact: str,
                     temporal: Optional[dict] = None) -> str: ...
    def list_relations(self, graph_id: str, limit: int,
                       cursor: Optional[str]) -> List[RelationEdge]: ...

    # Search
    def search(self, graph_id: str, query: str, scope: str = "all",
               limit: int = 10) -> SearchResult: ...

    # Graph algorithms (NEW - not available in Zep)
    def detect_communities(self, graph_id: str,
                           algorithm: str = "louvain") -> List[Community]: ...

    # Serialization (for Monte Carlo branching)
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
```

### 4.2 Document Ingestion Engine (Layer 1)

**Replaces:** `graph_builder.py` → `graph.create()`, `graph.set_ontology()`, `graph.add_batch()`, `graph.episode.get()`

**Current flow:**
1. Text chunked (500 chars, 50 overlap)
2. Chunks sent to Zep as "episodes" in batches of 3
3. Zep auto-extracts entities and relationships (black box)
4. Poll for completion (3s intervals, 10min timeout)

**New flow:**
1. Text chunked (same parameters)
2. Each chunk → LLM call with structured JSON output:
   ```json
   {
     "entities": [{"name": "...", "type": "...", "description": "..."}],
     "relations": [{"source": "...", "target": "...", "type": "...", "fact": "..."}]
   }
   ```
3. Entities deduplicated by name (merge descriptions)
4. Stored directly in NetworkX graph + SQLite

**LLM prompt strategy:**
- Include ontology (entity types, relation types) in the system prompt
- Use JSON mode (`response_format={"type": "json_object"}`) — already supported by `llm_client.py`
- One LLM call per chunk (vs. Zep's opaque multi-call pipeline)
- Retry with exponential backoff on malformed output

**Deduplication:**
- Entity key = normalized(name) + type
- On collision: merge descriptions, keep richer attributes
- Relations deduplicated by (source, target, type) triple

### 4.3 Simulation Runtime Memory (Layer 2)

**Replaces:** `zep_graph_memory_updater.py` → `graph.add(graph_id, type="text", data=narrative)`

**Critical insight:** The current system converts structured OASIS actions into natural language, then sends that text to Zep for re-extraction back into graph form. This is an unnecessary round-trip that introduces latency, cost, and hallucination risk.

**Current flow (wasteful):**
```
OASIS action (structured) → narrative text → Zep API → Zep NLP → graph edge
```

**New flow (direct):**
```
OASIS action (structured) → graph edge + SQLite text record
```

**Action-to-edge mapping** (preserving all 12 action types from zep_graph_memory_updater.py):

| OASIS Action | Graph Edge | SQLite Record |
|-------------|-----------|---------------|
| CREATE_POST | agent --[POSTED]--> post_node | Full post text, FTS indexed |
| LIKE_POST | agent --[LIKED]--> post_node (weight +1) | Like event log |
| DISLIKE_POST | agent --[DISLIKED]--> post_node (weight -1) | Dislike event log |
| REPOST | agent --[REPOSTED]--> post_node | Repost event log |
| QUOTE_POST | agent --[QUOTED]--> post_node + new post_node | Quote text + original ref |
| FOLLOW | agent --[FOLLOWS]--> target_agent | Follow event log |
| CREATE_COMMENT | agent --[COMMENTED]--> post_node | Comment text, FTS indexed |
| LIKE_COMMENT | agent --[LIKED]--> comment_node | Like event log |
| DISLIKE_COMMENT | agent --[DISLIKED]--> comment_node (weight -1) | Dislike event log |
| SEARCH_POSTS | agent --[SEARCHED]--> query_node | Search query log |
| SEARCH_USER | agent --[SEARCHED_USER]--> query_node | User search query log |
| MUTE | agent --[MUTED]--> target_agent | Mute event log |

**Edge attributes:**
```python
{
    "action_type": "CREATE_POST",
    "tick": 5,
    "simulated_time": "2026-01-15T14:00:00",
    "created_at": datetime.now(),
    "weight": 1.0,  # incremented for repeated interactions
    "platform": "twitter"  # or "reddit"
}
```

**Performance characteristics:**
- Zero LLM calls (vs. current: 1 Zep API call per 5 activities)
- Zero network latency (vs. current: HTTP round-trip to Zep Cloud)
- Batching not needed (in-memory operations are microseconds)

### 4.4 Retrieval & Analysis Engine (Layer 4)

**Replaces:** `zep_tools.py` → InsightForge, PanoramaSearch, QuickSearch, Interview

#### 4.4.1 InsightForge Replacement

**Current:** LLM decomposes query → 6 Zep searches → extract facts → entity details → relationship chains

**New implementation:**
1. LLM decomposes query into 5 sub-questions (keep existing logic)
2. Each sub-question → SQLite FTS5 `MATCH` query
3. For each result: traverse NetworkX neighbors (1-2 hops) to build relationship chains
4. Collect unique facts from edge attributes
5. Format and return `SearchResult`

**Why FTS5 is sufficient:** The LLM sub-question decomposition already converts semantic queries into keyword-friendly phrases. The existing codebase has a keyword fallback path that works when Zep search fails — this validates the approach.

#### 4.4.2 PanoramaSearch Replacement

**Current:** Fetch all nodes/edges from Zep, classify as active vs. historical based on temporal fields

**New implementation:**
1. NetworkX: iterate all nodes and edges with attributes
2. Filter by temporal metadata:
   - Active: `expired_at IS NULL AND invalid_at IS NULL`
   - Historical: `expired_at IS NOT NULL OR invalid_at IS NOT NULL`
3. Sort by relevance (edge weight + recency)
4. Return classified results

**Performance:** For 200-500 nodes, full graph iteration is sub-millisecond.

#### 4.4.3 QuickSearch Replacement

**Current:** Single Zep `graph.search()` call with keyword fallback

**New implementation:**
1. SQLite FTS5 `MATCH` query
2. Augment with NetworkX node/edge attribute matching
3. Return combined results

#### 4.4.4 Interview (No Change Needed)

The Interview tool in `zep_tools.py` calls `/api/simulation/interview/batch` — it does NOT query Zep directly. No modification required.

#### 4.4.5 Community Detection (NEW Capability)

**Not available in Zep.** This is a net-new capability that enables the wargaming use case.

```python
def detect_communities(self, graph_id: str, algorithm: str = "louvain") -> List[Community]:
    G = self._get_graph(graph_id)
    # Filter to agent-to-agent interaction subgraph
    agent_graph = nx.subgraph_view(G, filter_node=lambda n: G.nodes[n].get("type") == "agent")

    if algorithm == "louvain":
        communities = nx.community.louvain_communities(agent_graph, weight="weight")
    elif algorithm == "leiden":
        import graspologic
        # Convert to edge list for graspologic-native Leiden
        ...

    return [Community(id=i, members=list(c)) for i, c in enumerate(communities)]
```

**Usage in wargaming:**
- Run after each simulated hour to detect emerging factions
- Compare community structure across Monte Carlo branches
- Feed community summaries to ReportAgent for analysis

### 4.5 SQLite Schema

```sql
-- Per-simulation database: simulations/{sim_id}/memory.db

CREATE TABLE entities (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    summary TEXT,
    attributes TEXT,  -- JSON
    created_at TEXT NOT NULL,
    UNIQUE(graph_id, name, entity_type)
);

CREATE TABLE relations (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    source_uuid TEXT NOT NULL REFERENCES entities(uuid),
    target_uuid TEXT NOT NULL REFERENCES entities(uuid),
    relation_type TEXT NOT NULL,
    fact TEXT,
    attributes TEXT,  -- JSON
    weight REAL DEFAULT 1.0,
    tick INTEGER,
    created_at TEXT NOT NULL,
    valid_at TEXT,
    invalid_at TEXT,
    expired_at TEXT
);

CREATE TABLE texts (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    agent_id TEXT,
    content TEXT NOT NULL,
    action_type TEXT NOT NULL,
    metadata TEXT,  -- JSON
    tick INTEGER,
    created_at TEXT NOT NULL
);

-- Full-text search index
CREATE VIRTUAL TABLE texts_fts USING fts5(
    content,
    action_type,
    agent_id,
    content='texts',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER texts_ai AFTER INSERT ON texts BEGIN
    INSERT INTO texts_fts(rowid, content, action_type, agent_id)
    VALUES (new.rowid, new.content, new.action_type, new.agent_id);
END;

-- Indexes for common queries
CREATE INDEX idx_relations_graph ON relations(graph_id);
CREATE INDEX idx_relations_temporal ON relations(graph_id, expired_at, invalid_at);
CREATE INDEX idx_relations_tick ON relations(graph_id, tick);
CREATE INDEX idx_texts_agent ON texts(graph_id, agent_id);
CREATE INDEX idx_texts_tick ON texts(graph_id, tick);
```

### 4.6 Monte Carlo Branching Support

For parallel wargaming simulations, each branch needs isolated state:

**Graph branching:**
```python
import copy, json

# Fork a simulation branch
branch_graph = copy.deepcopy(main_graph)  # ~KB for 200 nodes

# Serialize for persistence (JSON-based, safe serialization)
data = nx.node_link_data(branch_graph)
snapshot = json.dumps(data)

# Restore
restored = nx.node_link_graph(json.loads(snapshot))
```

**SQLite branching:**
- Each branch gets its own `memory_{branch_id}.db` file
- SQLite file copy is atomic and fast for small databases
- Completed branches: keep DB for analysis; pruned branches: delete file

**Memory budget:** 200-node NetworkX graph ~ 5-50 KB. 1,000 branches ~ 5-50 MB. Trivial.

## 5. Migration Plan (File-by-File)

The abstraction layer enables gradual migration. Each file can be migrated independently:

| Priority | File | Zep Dependency | Migration Complexity |
|----------|------|---------------|---------------------|
| 1 | `graph_store.py` (NEW) | — | Create abstraction layer + NetworkX/SQLite implementation |
| 2 | `graph_builder.py` | `graph.create`, `set_ontology`, `add_batch`, `episode.get` | Replace with LLM extraction + GraphStore calls |
| 3 | `zep_entity_reader.py` | `node.get_by_graph_id`, `edge.get_by_graph_id`, filtering | Replace with GraphStore.list_entities/relations |
| 4 | `zep_paging.py` (utils) | `fetch_all_nodes`, `fetch_all_edges` Zep pagination helpers | Delete — no longer needed with local graph store |
| 5 | `zep_graph_memory_updater.py` | `graph.add(type="text")` | Replace with direct GraphStore.add_relation + TextStore.add_text |
| 6 | `simulation_runner.py` | Imports `ZepGraphMemoryManager`, wires updater into sim loop | Update to use new MemoryUpdater backed by GraphStore |
| 7 | `oasis_profile_generator.py` | Creates Zep client + ZepEntityReader for secondary enrichment; `graph.search(scope="edges/nodes")` | Replace all Zep client usage with GraphStore.search |
| 8 | `simulation_manager.py` | Zep client initialization, entity reader calls | Update to use GraphStore |
| 9 | `zep_tools.py` | InsightForge, PanoramaSearch, QuickSearch | Rewrite search tools against GraphStore + TextStore |
| 10 | `report_agent.py` | Uses zep_tools indirectly | Update tool imports (minimal change) |
| 11 | `simulation_config_generator.py` | Uses entity data from reader | Already abstracted through entity reader (no direct Zep) |
| 12 | `ontology_generator.py` | References `zep_cloud.external_clients.ontology`; Zep-specific entity type limit (max 10) | Remove Zep import and constraint |
| 13 | `services/__init__.py` | Re-exports `ZepEntityReader`, `ZepGraphMemoryUpdater`, `ZepGraphMemoryManager` | Update exports to new class names |
| 14 | `api/graph.py` | Zep graph building endpoints | Update to call new graph_builder |
| 15 | `api/simulation.py` | Zep client passing | Update to pass GraphStore instance |
| 16 | `api/report.py` | Zep client passing | Update to pass GraphStore instance |
| 17 | `config.py` | `ZEP_API_KEY` definition and validation | Remove Zep config entries |

## 6. New Dependencies

```
networkx>=3.0           # Graph engine + native Louvain (already available)
graspologic-native      # Optional: Leiden community detection (pip install)
# sqlite3               # Python standard library — zero install
```

**Removed dependency:**
```
zep-cloud==3.13.0       # Fully replaced
```

## 7. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| FTS5 keyword search less capable than Zep semantic search | Medium | InsightForge's LLM sub-question decomposition converts semantic → keyword-friendly; existing keyword fallback validates approach; V2 can add sentence-transformers |
| 9B local model entity extraction quality (Step 1) | Medium | Strict JSON Schema + retry; ontology in prompt constrains output; Step 1 runs once per project (low volume) |
| NetworkX memory for very large simulations (1000+ agents) | Low | 1000 nodes ~ 100KB; well within memory; can shard if needed |
| Migration breaks existing functionality | Medium | Abstraction layer + file-by-file migration; each step can be tested independently; keep Zep as fallback during transition |

## 8. Success Criteria

1. All 15+ Zep-dependent files migrated to use abstraction layer
2. `zep-cloud` removed from `requirements.txt`
3. `ZEP_API_KEY` removed from `.env` configuration
4. All existing simulation workflows (Step 1-5) function without Zep
5. Community detection available as new capability
6. Report generation produces comparable quality output
7. Simulation runtime has lower latency than Zep-based version
8. Monte Carlo branching demonstrated with ≥10 parallel simulations

## 9. Out of Scope (Future Work)

- Wargame Matrix (systematic variable perturbation) — separate spec
- Behavior Trees / LLM responsibility downgrade — separate spec
- Automated pruning of invalid simulation branches — separate spec
- Result clustering and convergence analysis — separate spec
- Hybrid Neuro-Symbolic architecture — separate spec
- Application-specific adaptations (finance, PR crisis) — separate spec
