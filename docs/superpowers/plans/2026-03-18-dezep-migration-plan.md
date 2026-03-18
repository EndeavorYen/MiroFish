# De-Zep Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all Zep Cloud dependencies with self-hosted NetworkX + SQLite hybrid architecture, enabling local graph operations, full-text search, and community detection.

**Architecture:** Abstraction layer (Python Protocols) sits between business logic and storage backends. NetworkX handles in-memory graph algorithms. SQLite + FTS5 handles persistence and text search. LLM entity extraction replaces Zep's auto-extraction pipeline.

**Tech Stack:** Python 3.10+, NetworkX, SQLite3 (stdlib), pytest

**Spec:** `docs/superpowers/specs/2026-03-18-dezep-migration-design.md`

---

## File Structure

### New Files to Create

| File | Responsibility |
|------|---------------|
| `backend/app/services/graph_store.py` | Data models (EntityNode, RelationEdge, SearchResult, Community) + Protocol definitions (GraphStore, TextStore) |
| `backend/app/services/networkx_graph_store.py` | NetworkX + SQLite implementation of GraphStore and TextStore protocols |
| `backend/app/services/entity_reader.py` | Local entity reader replacing ZepEntityReader (uses GraphStore) |
| `backend/app/services/graph_memory_updater.py` | Local memory updater replacing ZepGraphMemoryUpdater (uses GraphStore + TextStore) |
| `backend/app/services/search_tools.py` | InsightForge, PanoramaSearch, QuickSearch replacements (uses GraphStore + TextStore) |
| `backend/tests/conftest.py` | Shared pytest fixtures |
| `backend/tests/test_graph_store.py` | Tests for data models |
| `backend/tests/test_networkx_graph_store.py` | Tests for NetworkX+SQLite implementation |
| `backend/tests/test_entity_reader.py` | Tests for local entity reader |
| `backend/tests/test_graph_memory_updater.py` | Tests for memory updater |
| `backend/tests/test_search_tools.py` | Tests for search tool replacements |
| `backend/tests/test_graph_builder.py` | Tests for migrated graph builder |

### Files to Modify

| File | Change Summary |
|------|---------------|
| `backend/app/services/graph_builder.py` | Replace Zep API calls with GraphStore + LLM entity extraction |
| `backend/app/services/oasis_profile_generator.py` | Replace Zep client + ZepEntityReader with GraphStore.search |
| `backend/app/services/simulation_runner.py` | Replace ZepGraphMemoryManager import with new GraphMemoryManager |
| `backend/app/services/simulation_manager.py` | Replace ZepEntityReader with EntityReader, remove Zep client init |
| `backend/app/services/report_agent.py` | Update tool imports from zep_tools → search_tools |
| `backend/app/services/ontology_generator.py` | Remove max-10 entity type constraint, remove Zep imports |
| `backend/app/services/__init__.py` | Update exports to new class names |
| `backend/app/api/graph.py` | Remove ZEP_API_KEY checks, pass GraphStore |
| `backend/app/api/simulation.py` | Remove ZEP_API_KEY checks, use new EntityReader |
| `backend/app/api/report.py` | Remove ZEP_API_KEY checks |
| `backend/app/config.py` | Remove ZEP_API_KEY, ZEP_REQUEST_TIMEOUT_SECONDS |
| `backend/requirements.txt` | Remove zep-cloud, add networkx |

### Files to Delete

| File | Reason |
|------|--------|
| `backend/app/utils/zep_paging.py` | Pagination helpers for Zep API — not needed with local store |
| `backend/app/services/zep_entity_reader.py` | Replaced by `entity_reader.py` |
| `backend/app/services/zep_graph_memory_updater.py` | Replaced by `graph_memory_updater.py` |
| `backend/app/services/zep_tools.py` | Replaced by `search_tools.py` |

---

## Task 1: Data Models and Protocol Definitions

**Files:**
- Create: `backend/app/services/graph_store.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_graph_store.py`

- [ ] **Step 1: Create test directory and conftest**

```bash
mkdir -p backend/tests
touch backend/tests/__init__.py
```

`backend/tests/conftest.py`:
```python
import pytest
import tempfile
import os


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary directory for SQLite databases."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def graph_id():
    """Standard test graph ID."""
    return "test-graph-001"
```

- [ ] **Step 2: Write failing tests for data models**

`backend/tests/test_graph_store.py`:
```python
from datetime import datetime


def test_entity_node_creation():
    from backend.app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Person"],
        summary="A researcher", attributes={"age": 30},
        related_edges=[], related_nodes=[]
    )
    assert node.name == "Alice"
    assert node.labels == ["Person"]


def test_entity_node_to_dict():
    from backend.app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Person"],
        summary="A researcher", attributes={},
        related_edges=[], related_nodes=[]
    )
    d = node.to_dict()
    assert d["uuid"] == "n1"
    assert d["name"] == "Alice"


def test_entity_node_get_entity_type():
    from backend.app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Entity", "Person"],
        summary="", attributes={}, related_edges=[], related_nodes=[]
    )
    assert node.get_entity_type() == "Person"


def test_entity_node_get_entity_type_generic_only():
    from backend.app.services.graph_store import EntityNode
    node = EntityNode(
        uuid="n1", name="Alice", labels=["Entity"],
        summary="", attributes={}, related_edges=[], related_nodes=[]
    )
    assert node.get_entity_type() == "Entity"


def test_relation_edge_creation():
    from backend.app.services.graph_store import RelationEdge
    edge = RelationEdge(
        uuid="e1", name="KNOWS", fact="Alice knows Bob",
        source_node_uuid="n1", target_node_uuid="n2",
        attributes={}, created_at=datetime.now()
    )
    assert edge.fact == "Alice knows Bob"
    assert edge.valid_at is None


def test_search_result_creation():
    from backend.app.services.graph_store import SearchResult
    result = SearchResult(nodes=[], edges=[], facts=["fact1"])
    assert len(result.facts) == 1


def test_community_creation():
    from backend.app.services.graph_store import Community
    c = Community(id=0, members=["n1", "n2"])
    assert c.summary is None
    assert len(c.members) == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_graph_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app.services.graph_store'`

- [ ] **Step 4: Implement data models and protocols**

`backend/app/services/graph_store.py`:
```python
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
        """Format for LLM consumption, matching ZepTools output format."""
        lines = []
        if query:
            lines.append(f"搜索查询: {query}")
        lines.append(f"找到 {len(self.facts)} 条相关信息")
        if self.facts:
            lines.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                lines.append(f"{i}. {fact}")
        if self.nodes:
            lines.append("\n### 相关实体:")
            for node in self.nodes:
                labels_str = ", ".join(node.labels)
                lines.append(f"- **{node.name}** [{labels_str}]: {node.summary}")
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_graph_store.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/graph_store.py backend/tests/
git commit -m "feat: add graph store data models and protocol definitions"
```

---

## Task 2: NetworkX + SQLite GraphStore Implementation

**Files:**
- Create: `backend/app/services/networkx_graph_store.py`
- Create: `backend/tests/test_networkx_graph_store.py`

- [ ] **Step 1: Write failing tests for NetworkXGraphStore**

`backend/tests/test_networkx_graph_store.py`:
```python
import pytest
from datetime import datetime


@pytest.fixture
def store(tmp_db_path):
    from backend.app.services.networkx_graph_store import NetworkXGraphStore
    return NetworkXGraphStore(db_path=tmp_db_path)


@pytest.fixture
def populated_store(store, graph_id):
    """Store with a graph containing entities and relations."""
    store.create_graph(graph_id, "Test Graph", "For testing")
    store.add_entity(graph_id, "Alice", "Person", "A researcher", {"age": 30})
    store.add_entity(graph_id, "Bob", "Person", "An engineer", {"age": 25})
    store.add_entity(graph_id, "ACME", "Organization", "A company", {})
    return store


class TestGraphLifecycle:
    def test_create_graph(self, store, graph_id):
        result = store.create_graph(graph_id, "My Graph", "A test graph")
        assert result == graph_id

    def test_create_duplicate_graph_raises(self, store, graph_id):
        store.create_graph(graph_id, "G1", "")
        with pytest.raises(ValueError, match="already exists"):
            store.create_graph(graph_id, "G2", "")

    def test_delete_graph(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        store.add_entity(graph_id, "Alice", "Person", "", {})
        store.delete_graph(graph_id)
        assert store.list_entities(graph_id) == []


class TestEntityOperations:
    def test_add_and_get_entity(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        uuid = store.add_entity(graph_id, "Alice", "Person", "A researcher", {"age": 30})
        entity = store.get_entity(graph_id, uuid)
        assert entity.name == "Alice"
        assert entity.labels == ["Person"]
        assert entity.attributes["age"] == 30

    def test_add_duplicate_entity_deduplicates(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        uuid1 = store.add_entity(graph_id, "Alice", "Person", "Short bio", {})
        uuid2 = store.add_entity(graph_id, "Alice", "Person", "Longer, richer bio", {"age": 30})
        assert uuid1 == uuid2  # same entity, deduplicated
        entity = store.get_entity(graph_id, uuid1)
        assert entity.summary == "Longer, richer bio"  # merged to richer
        assert entity.attributes["age"] == 30

    def test_list_entities(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        assert len(entities) == 3
        names = {e.name for e in entities}
        assert names == {"Alice", "Bob", "ACME"}

    def test_list_entities_with_limit(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id, limit=2)
        assert len(entities) == 2

    def test_get_entity_edges(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        alice = next(e for e in entities if e.name == "Alice")
        bob = next(e for e in entities if e.name == "Bob")
        populated_store.add_relation(
            graph_id, alice.uuid, bob.uuid,
            "KNOWS", "Alice knows Bob"
        )
        edges = populated_store.get_entity_edges(graph_id, alice.uuid)
        assert len(edges) == 1
        assert edges[0].fact == "Alice knows Bob"


class TestRelationOperations:
    def test_add_and_list_relations(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        alice = next(e for e in entities if e.name == "Alice")
        acme = next(e for e in entities if e.name == "ACME")
        populated_store.add_relation(
            graph_id, alice.uuid, acme.uuid,
            "WORKS_AT", "Alice works at ACME",
            attributes={"role": "lead"},
            temporal={"valid_at": datetime(2025, 1, 1).isoformat()}
        )
        relations = populated_store.list_relations(graph_id)
        assert len(relations) == 1
        assert relations[0].name == "WORKS_AT"
        assert relations[0].valid_at is not None


class TestSearch:
    def test_search_nodes_by_name(self, populated_store, graph_id):
        result = populated_store.search(graph_id, "Alice", scope="nodes")
        assert len(result.nodes) >= 1
        assert result.nodes[0].name == "Alice"

    def test_search_edges_by_fact(self, populated_store, graph_id):
        entities = populated_store.list_entities(graph_id)
        alice = next(e for e in entities if e.name == "Alice")
        bob = next(e for e in entities if e.name == "Bob")
        populated_store.add_relation(
            graph_id, alice.uuid, bob.uuid,
            "KNOWS", "Alice knows Bob from university"
        )
        result = populated_store.search(graph_id, "university", scope="edges")
        assert len(result.edges) >= 1

    def test_search_all(self, populated_store, graph_id):
        result = populated_store.search(graph_id, "researcher", scope="all")
        assert len(result.facts) >= 1


class TestCommunityDetection:
    def test_louvain_with_interactions(self, store, graph_id):
        store.create_graph(graph_id, "G", "")
        # Create two clusters: {A,B,C} and {D,E,F}
        agents = {}
        for name in ["A", "B", "C", "D", "E", "F"]:
            agents[name] = store.add_entity(graph_id, name, "agent", "", {})
        # Dense connections within clusters
        for src, tgt in [("A","B"),("B","C"),("A","C")]:
            store.add_relation(graph_id, agents[src], agents[tgt], "INTERACTS", "",
                               attributes={"weight": 5.0})
        for src, tgt in [("D","E"),("E","F"),("D","F")]:
            store.add_relation(graph_id, agents[src], agents[tgt], "INTERACTS", "",
                               attributes={"weight": 5.0})
        # Weak link between clusters
        store.add_relation(graph_id, agents["C"], agents["D"], "INTERACTS", "",
                           attributes={"weight": 1.0})
        communities = store.detect_communities(graph_id)
        assert len(communities) >= 2


class TestSerialization:
    def test_snapshot_and_restore(self, populated_store, graph_id):
        data = populated_store.snapshot(graph_id)
        assert isinstance(data, bytes)
        restored_id = populated_store.restore(data)
        entities = populated_store.list_entities(restored_id)
        assert len(entities) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_networkx_graph_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement NetworkXGraphStore**

`backend/app/services/networkx_graph_store.py`:

This is the core implementation. Key design decisions:
- One NetworkX DiGraph per graph_id (stored in `self._graphs` dict)
- One SQLite connection per instance (shared across graphs)
- Entity deduplication by normalized(name) + type
- Relations stored as NetworkX edges with full attributes
- FTS5 for text search
- Louvain community detection via `networkx.community`

```python
"""
NetworkX + SQLite implementation of GraphStore and TextStore protocols.

NetworkX handles: in-memory graph, algorithms (community detection, path finding)
SQLite handles: persistence, full-text search (FTS5), temporal queries
"""

import json
import copy
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

import networkx as nx

from .graph_store import (
    EntityNode, RelationEdge, SearchResult, Community,
    generate_uuid,
)


class NetworkXGraphStore:
    """Combined GraphStore + TextStore backed by NetworkX and SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._lock = threading.Lock()
        self._db_path = db_path or ":memory:"
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._db_lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Create SQLite tables and FTS5 index."""
        with self._db_lock:
            cur = self._conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    uuid TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    attributes TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(graph_id, name, entity_type)
                );

                CREATE TABLE IF NOT EXISTS relations (
                    uuid TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    source_uuid TEXT NOT NULL,
                    target_uuid TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    fact TEXT DEFAULT '',
                    attributes TEXT DEFAULT '{}',
                    weight REAL DEFAULT 1.0,
                    tick INTEGER,
                    created_at TEXT NOT NULL,
                    valid_at TEXT,
                    invalid_at TEXT,
                    expired_at TEXT
                );

                CREATE TABLE IF NOT EXISTS texts (
                    uuid TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    agent_id TEXT,
                    content TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    tick INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_entities_graph
                    ON entities(graph_id);
                CREATE INDEX IF NOT EXISTS idx_relations_graph
                    ON relations(graph_id);
                CREATE INDEX IF NOT EXISTS idx_relations_temporal
                    ON relations(graph_id, expired_at, invalid_at);
                CREATE INDEX IF NOT EXISTS idx_relations_tick
                    ON relations(graph_id, tick);
                CREATE INDEX IF NOT EXISTS idx_texts_agent
                    ON texts(graph_id, agent_id);
                CREATE INDEX IF NOT EXISTS idx_texts_tick
                    ON texts(graph_id, tick);
            """)
            # FTS5 virtual table (created separately to handle IF NOT EXISTS)
            try:
                cur.execute("""
                    CREATE VIRTUAL TABLE texts_fts USING fts5(
                        content, action_type, agent_id,
                        content='texts', content_rowid='rowid'
                    )
                """)
                cur.execute("""
                    CREATE TRIGGER texts_ai AFTER INSERT ON texts BEGIN
                        INSERT INTO texts_fts(rowid, content, action_type, agent_id)
                        VALUES (new.rowid, new.content, new.action_type, new.agent_id);
                    END
                """)
            except sqlite3.OperationalError:
                pass  # Already exists
            self._conn.commit()

    def _get_graph(self, graph_id: str) -> nx.DiGraph:
        if graph_id not in self._graphs:
            self._graphs[graph_id] = nx.DiGraph()
        return self._graphs[graph_id]

    # ── Graph Lifecycle ──

    def create_graph(self, graph_id: str, name: str, description: str) -> str:
        with self._lock:
            if graph_id in self._graphs:
                raise ValueError(f"Graph '{graph_id}' already exists")
            G = nx.DiGraph()
            G.graph["name"] = name
            G.graph["description"] = description
            G.graph["created_at"] = datetime.now().isoformat()
            self._graphs[graph_id] = G
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        with self._lock:
            self._graphs.pop(graph_id, None)
        with self._db_lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM entities WHERE graph_id = ?", (graph_id,))
            cur.execute("DELETE FROM relations WHERE graph_id = ?", (graph_id,))
            cur.execute("DELETE FROM texts WHERE graph_id = ?", (graph_id,))
            self._conn.commit()

    # ── Entity Operations ──

    def add_entity(self, graph_id: str, name: str, entity_type: str,
                   summary: str, attributes: dict) -> str:
        G = self._get_graph(graph_id)
        # Deduplication: check existing by normalized name + type
        norm_key = (name.strip().lower(), entity_type.strip().lower())
        for node_id, data in G.nodes(data=True):
            existing_key = (data.get("name", "").strip().lower(),
                            data.get("entity_type", "").strip().lower())
            if existing_key == norm_key:
                # Merge: keep richer summary and update attributes
                if len(summary) > len(data.get("summary", "")):
                    G.nodes[node_id]["summary"] = summary
                G.nodes[node_id]["attributes"].update(attributes)
                self._upsert_entity_db(node_id, graph_id, name, entity_type,
                                        G.nodes[node_id]["summary"],
                                        G.nodes[node_id]["attributes"])
                return node_id

        # New entity
        entity_uuid = generate_uuid()
        G.add_node(entity_uuid, name=name, entity_type=entity_type,
                   labels=[entity_type], summary=summary,
                   attributes=attributes, graph_id=graph_id,
                   created_at=datetime.now().isoformat())
        self._upsert_entity_db(entity_uuid, graph_id, name, entity_type,
                                summary, attributes)
        return entity_uuid

    def _upsert_entity_db(self, uuid: str, graph_id: str, name: str,
                           entity_type: str, summary: str, attributes: dict):
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO entities (uuid, graph_id, name, entity_type, summary,
                   attributes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(graph_id, name, entity_type)
                   DO UPDATE SET summary=excluded.summary,
                   attributes=excluded.attributes""",
                (uuid, graph_id, name, entity_type, summary,
                 json.dumps(attributes, ensure_ascii=False),
                 datetime.now().isoformat())
            )
            self._conn.commit()

    def get_entity(self, graph_id: str, uuid: str) -> EntityNode:
        G = self._get_graph(graph_id)
        if uuid not in G.nodes:
            raise KeyError(f"Entity '{uuid}' not found in graph '{graph_id}'")
        data = G.nodes[uuid]
        related_edges = self._get_related_edges_data(G, uuid)
        related_nodes = self._get_related_nodes_data(G, uuid)
        return EntityNode(
            uuid=uuid, name=data["name"],
            labels=data.get("labels", [data.get("entity_type", "Entity")]),
            summary=data.get("summary", ""),
            attributes=data.get("attributes", {}),
            related_edges=related_edges,
            related_nodes=related_nodes,
        )

    def _get_related_edges_data(self, G: nx.DiGraph, uuid: str) -> List[dict]:
        edges = []
        for _, tgt, data in G.out_edges(uuid, data=True):
            edges.append({
                "direction": "outgoing", "edge_name": data.get("name", ""),
                "fact": data.get("fact", ""),
                "target_node_uuid": tgt,
            })
        for src, _, data in G.in_edges(uuid, data=True):
            edges.append({
                "direction": "incoming", "edge_name": data.get("name", ""),
                "fact": data.get("fact", ""),
                "source_node_uuid": src,
            })
        return edges

    def _get_related_nodes_data(self, G: nx.DiGraph, uuid: str) -> List[dict]:
        neighbors = set(G.successors(uuid)) | set(G.predecessors(uuid))
        nodes = []
        for nid in neighbors:
            nd = G.nodes[nid]
            nodes.append({
                "uuid": nid, "name": nd.get("name", ""),
                "labels": nd.get("labels", []),
                "summary": nd.get("summary", ""),
            })
        return nodes

    def get_entity_edges(self, graph_id: str, uuid: str) -> List[RelationEdge]:
        G = self._get_graph(graph_id)
        edges = []
        for src, tgt, data in G.out_edges(uuid, data=True):
            edges.append(self._edge_data_to_relation(data, src, tgt))
        for src, tgt, data in G.in_edges(uuid, data=True):
            edges.append(self._edge_data_to_relation(data, src, tgt))
        return edges

    def list_entities(self, graph_id: str, limit: int = 100,
                      cursor: Optional[str] = None) -> List[EntityNode]:
        G = self._get_graph(graph_id)
        all_nodes = [
            (nid, data) for nid, data in G.nodes(data=True)
            if data.get("graph_id") == graph_id or graph_id in self._graphs
        ]
        # Cursor-based pagination
        start = 0
        if cursor:
            for i, (nid, _) in enumerate(all_nodes):
                if nid == cursor:
                    start = i + 1
                    break
        sliced = all_nodes[start:start + limit]
        return [
            EntityNode(
                uuid=nid, name=data.get("name", ""),
                labels=data.get("labels", [data.get("entity_type", "Entity")]),
                summary=data.get("summary", ""),
                attributes=data.get("attributes", {}),
                related_edges=self._get_related_edges_data(G, nid),
                related_nodes=self._get_related_nodes_data(G, nid),
            )
            for nid, data in sliced
        ]

    # ── Relation Operations ──

    def add_relation(self, graph_id: str, source_uuid: str, target_uuid: str,
                     relation_type: str, fact: str,
                     attributes: Optional[dict] = None,
                     temporal: Optional[dict] = None) -> str:
        G = self._get_graph(graph_id)
        edge_uuid = generate_uuid()
        now = datetime.now().isoformat()
        attrs = attributes or {}
        temp = temporal or {}

        G.add_edge(source_uuid, target_uuid,
                   uuid=edge_uuid, name=relation_type, fact=fact,
                   attributes=attrs,
                   weight=attrs.get("weight", 1.0),
                   created_at=now,
                   valid_at=temp.get("valid_at"),
                   invalid_at=temp.get("invalid_at"),
                   expired_at=temp.get("expired_at"),
                   tick=attrs.get("tick"))

        with self._db_lock:
            self._conn.execute(
                """INSERT INTO relations (uuid, graph_id, source_uuid, target_uuid,
                   relation_type, fact, attributes, weight, tick,
                   created_at, valid_at, invalid_at, expired_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (edge_uuid, graph_id, source_uuid, target_uuid,
                 relation_type, fact, json.dumps(attrs, ensure_ascii=False),
                 attrs.get("weight", 1.0), attrs.get("tick"),
                 now, temp.get("valid_at"), temp.get("invalid_at"),
                 temp.get("expired_at"))
            )
            self._conn.commit()
        return edge_uuid

    def list_relations(self, graph_id: str, limit: int = 100,
                       cursor: Optional[str] = None) -> List[RelationEdge]:
        G = self._get_graph(graph_id)
        all_edges = [(src, tgt, data) for src, tgt, data in G.edges(data=True)]
        start = 0
        if cursor:
            for i, (_, _, data) in enumerate(all_edges):
                if data.get("uuid") == cursor:
                    start = i + 1
                    break
        sliced = all_edges[start:start + limit]
        return [self._edge_data_to_relation(data, src, tgt) for src, tgt, data in sliced]

    def _edge_data_to_relation(self, data: dict, src: str, tgt: str) -> RelationEdge:
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return RelationEdge(
            uuid=data.get("uuid", ""),
            name=data.get("name", ""),
            fact=data.get("fact", ""),
            source_node_uuid=src,
            target_node_uuid=tgt,
            attributes=data.get("attributes", {}),
            created_at=parse_dt(data.get("created_at")) or datetime.now(),
            valid_at=parse_dt(data.get("valid_at")),
            invalid_at=parse_dt(data.get("invalid_at")),
            expired_at=parse_dt(data.get("expired_at")),
        )

    # ── Search ──

    def search(self, graph_id: str, query: str, scope: str = "all",
               limit: int = 10) -> SearchResult:
        G = self._get_graph(graph_id)
        query_lower = query.lower()
        nodes, edges, facts = [], [], []

        if scope in ("all", "nodes"):
            for nid, data in G.nodes(data=True):
                text = f"{data.get('name', '')} {data.get('summary', '')}".lower()
                if query_lower in text:
                    nodes.append(EntityNode(
                        uuid=nid, name=data.get("name", ""),
                        labels=data.get("labels", []),
                        summary=data.get("summary", ""),
                        attributes=data.get("attributes", {}),
                        related_edges=[], related_nodes=[],
                    ))
                    facts.append(f"{data.get('name', '')}: {data.get('summary', '')}")

        if scope in ("all", "edges"):
            for src, tgt, data in G.edges(data=True):
                text = f"{data.get('name', '')} {data.get('fact', '')}".lower()
                if query_lower in text:
                    edges.append(self._edge_data_to_relation(data, src, tgt))
                    facts.append(data.get("fact", ""))

        # Also search SQLite FTS if we have text content
        if scope in ("all", "text"):
            fts_results = self.search_text(graph_id, query, limit=limit)
            for row in fts_results:
                facts.append(row.get("content", ""))

        return SearchResult(
            nodes=nodes[:limit], edges=edges[:limit],
            facts=list(dict.fromkeys(facts))[:limit],  # deduplicate preserving order
        )

    # ── TextStore Operations ──

    def add_text(self, graph_id: str, agent_id: str, content: str,
                 action_type: str, metadata: dict,
                 tick: int, timestamp: datetime) -> str:
        text_uuid = generate_uuid()
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO texts (uuid, graph_id, agent_id, content,
                   action_type, metadata, tick, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (text_uuid, graph_id, agent_id, content, action_type,
                 json.dumps(metadata, ensure_ascii=False), tick,
                 timestamp.isoformat())
            )
            self._conn.commit()
        return text_uuid

    def search_text(self, graph_id: str, query: str,
                    limit: int = 20) -> List[dict]:
        with self._db_lock:
            try:
                rows = self._conn.execute(
                    """SELECT t.* FROM texts t
                       JOIN texts_fts fts ON t.rowid = fts.rowid
                       WHERE texts_fts MATCH ? AND t.graph_id = ?
                       LIMIT ?""",
                    (query, graph_id, limit)
                ).fetchall()
            except sqlite3.OperationalError:
                # FTS match failed (e.g. special chars), fallback to LIKE
                rows = self._conn.execute(
                    """SELECT * FROM texts
                       WHERE graph_id = ? AND content LIKE ?
                       LIMIT ?""",
                    (graph_id, f"%{query}%", limit)
                ).fetchall()
        return [dict(row) for row in rows]

    def get_by_agent(self, graph_id: str, agent_id: str,
                     limit: int = 50) -> List[dict]:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM texts WHERE graph_id = ? AND agent_id = ? ORDER BY tick LIMIT ?",
                (graph_id, agent_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_time_range(self, graph_id: str, start: datetime,
                          end: datetime) -> List[dict]:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM texts WHERE graph_id = ? AND created_at BETWEEN ? AND ?",
                (graph_id, start.isoformat(), end.isoformat())
            ).fetchall()
        return [dict(row) for row in rows]

    def get_active_facts(self, graph_id: str) -> List[dict]:
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT * FROM relations
                   WHERE graph_id = ? AND expired_at IS NULL AND invalid_at IS NULL""",
                (graph_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_historical_facts(self, graph_id: str) -> List[dict]:
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT * FROM relations
                   WHERE graph_id = ? AND (expired_at IS NOT NULL OR invalid_at IS NOT NULL)""",
                (graph_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Community Detection ──

    def detect_communities(self, graph_id: str,
                           algorithm: str = "louvain") -> List[Community]:
        G = self._get_graph(graph_id)
        if len(G.nodes) == 0:
            return []
        # Convert to undirected for community detection
        UG = G.to_undirected()
        if algorithm == "louvain":
            communities = nx.community.louvain_communities(
                UG, weight="weight", seed=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return [
            Community(id=i, members=sorted(members))
            for i, members in enumerate(communities)
        ]

    # ── Serialization ──

    def snapshot(self, graph_id: str) -> bytes:
        G = self._get_graph(graph_id)
        data = nx.node_link_data(G)
        return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")

    def restore(self, data: bytes) -> str:
        parsed = json.loads(data.decode("utf-8"))
        G = nx.node_link_graph(parsed, directed=True)
        new_id = f"restored-{generate_uuid()[:8]}"
        G.graph["name"] = G.graph.get("name", "Restored")
        with self._lock:
            self._graphs[new_id] = G
        # Sync entities to SQLite
        for nid, ndata in G.nodes(data=True):
            self._upsert_entity_db(
                nid, new_id, ndata.get("name", ""),
                ndata.get("entity_type", "Entity"),
                ndata.get("summary", ""), ndata.get("attributes", {})
            )
        return new_id

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_networkx_graph_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/networkx_graph_store.py backend/tests/test_networkx_graph_store.py
git commit -m "feat: implement NetworkX+SQLite graph store with community detection"
```

---

## Task 3: Local Entity Reader

**Files:**
- Create: `backend/app/services/entity_reader.py`
- Create: `backend/tests/test_entity_reader.py`

This replaces `zep_entity_reader.py`. The entity reader filters entities from a graph, excluding generic "Entity" types and enriching with related edges/nodes.

- [ ] **Step 1: Write failing tests**

`backend/tests/test_entity_reader.py`:
```python
import pytest


@pytest.fixture
def store_with_entities(tmp_db_path):
    from backend.app.services.networkx_graph_store import NetworkXGraphStore
    store = NetworkXGraphStore(db_path=tmp_db_path)
    gid = "test-graph"
    store.create_graph(gid, "Test", "")
    store.add_entity(gid, "Alice", "Person", "A researcher", {"role": "lead"})
    store.add_entity(gid, "ACME", "Organization", "A tech company", {})
    store.add_entity(gid, "GenericThing", "Entity", "Untyped thing", {})
    # Add a relation
    entities = store.list_entities(gid)
    alice = next(e for e in entities if e.name == "Alice")
    acme = next(e for e in entities if e.name == "ACME")
    store.add_relation(gid, alice.uuid, acme.uuid, "WORKS_AT", "Alice works at ACME")
    return store, gid


def test_filter_defined_entities_excludes_generic(store_with_entities):
    from backend.app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid)
    names = {e.name for e in result.entities}
    assert "Alice" in names
    assert "ACME" in names
    assert "GenericThing" not in names


def test_filter_defined_entities_by_type(store_with_entities):
    from backend.app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid, defined_entity_types=["Person"])
    assert all(e.get_entity_type() == "Person" for e in result.entities)


def test_filter_enriches_with_edges(store_with_entities):
    from backend.app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid, enrich_with_edges=True)
    alice = next(e for e in result.entities if e.name == "Alice")
    assert len(alice.related_edges) > 0


def test_filtered_entities_counts(store_with_entities):
    from backend.app.services.entity_reader import EntityReader
    store, gid = store_with_entities
    reader = EntityReader(store)
    result = reader.filter_defined_entities(gid)
    assert result.total_count == 3  # all entities
    assert result.filtered_count == 2  # excluding GenericThing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_entity_reader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement EntityReader**

`backend/app/services/entity_reader.py`:
```python
"""
Local entity reader replacing ZepEntityReader.

Reads entities from a GraphStore, filters by type, enriches with edges/neighbors.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set

from .graph_store import EntityNode
from .networkx_graph_store import NetworkXGraphStore


@dataclass
class FilteredEntities:
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self):
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": sorted(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


GENERIC_TYPES = {"Entity", "Node"}


class EntityReader:
    """Reads and filters entities from a local GraphStore."""

    def __init__(self, store: NetworkXGraphStore):
        self.store = store

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = False,
    ) -> FilteredEntities:
        all_entities = self.store.list_entities(graph_id, limit=10000)
        total_count = len(all_entities)

        # Filter out generic types
        filtered = []
        entity_types = set()
        for entity in all_entities:
            etype = entity.get_entity_type()
            if etype in GENERIC_TYPES:
                continue
            if defined_entity_types and etype not in defined_entity_types:
                continue
            entity_types.add(etype)
            filtered.append(entity)

        # Enrich with edges and neighbor data
        if enrich_with_edges:
            for entity in filtered:
                edges = self.store.get_entity_edges(graph_id, entity.uuid)
                entity.related_edges = [
                    {
                        "direction": "outgoing" if e.source_node_uuid == entity.uuid else "incoming",
                        "edge_name": e.name,
                        "fact": e.fact,
                        "source_node_uuid": e.source_node_uuid,
                        "target_node_uuid": e.target_node_uuid,
                    }
                    for e in edges
                ]
                neighbor_uuids = set()
                for e in edges:
                    other = e.target_node_uuid if e.source_node_uuid == entity.uuid else e.source_node_uuid
                    neighbor_uuids.add(other)
                entity.related_nodes = []
                for nid in neighbor_uuids:
                    try:
                        n = self.store.get_entity(graph_id, nid)
                        entity.related_nodes.append({
                            "uuid": n.uuid, "name": n.name,
                            "labels": n.labels, "summary": n.summary,
                        })
                    except KeyError:
                        pass

        return FilteredEntities(
            entities=filtered,
            entity_types=entity_types,
            total_count=total_count,
            filtered_count=len(filtered),
        )

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        try:
            return self.store.get_entity(graph_id, entity_uuid)
        except KeyError:
            return None

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[EntityNode]:
        result = self.filter_defined_entities(
            graph_id, defined_entity_types=[entity_type], enrich_with_edges=True
        )
        return result.entities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_entity_reader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/entity_reader.py backend/tests/test_entity_reader.py
git commit -m "feat: add local entity reader replacing ZepEntityReader"
```

---

## Task 4: Local Graph Memory Updater

**Files:**
- Create: `backend/app/services/graph_memory_updater.py`
- Create: `backend/tests/test_graph_memory_updater.py`

Replaces `zep_graph_memory_updater.py`. Converts OASIS structured actions directly into graph edges + SQLite text records (zero LLM cost).

- [ ] **Step 1: Write failing tests**

`backend/tests/test_graph_memory_updater.py`:
```python
import pytest
from datetime import datetime


@pytest.fixture
def store(tmp_db_path):
    from backend.app.services.networkx_graph_store import NetworkXGraphStore
    store = NetworkXGraphStore(db_path=tmp_db_path)
    store.create_graph("sim-1", "Sim", "")
    # Add agent entities
    store.add_entity("sim-1", "agent_1", "agent", "User 1", {})
    store.add_entity("sim-1", "agent_2", "agent", "User 2", {})
    return store


@pytest.fixture
def updater(store):
    from backend.app.services.graph_memory_updater import GraphMemoryUpdater
    return GraphMemoryUpdater(graph_id="sim-1", store=store)


def test_create_post_adds_edge_and_text(updater, store):
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "agent_1",
        "agent_name": "User1",
        "action_type": "CREATE_POST",
        "action_args": {"content": "Hello world!", "post_id": "post_1"},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert any(e.name == "POSTED" for e in edges)
    texts = store.search_text("sim-1", "Hello world")
    assert len(texts) >= 1


def test_like_post_adds_edge(updater, store):
    # First create a post
    store.add_entity("sim-1", "post_1", "post", "A post", {})
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "agent_1",
        "agent_name": "User1",
        "action_type": "LIKE_POST",
        "action_args": {"post_id": "post_1", "author": "User2", "content": "Nice"},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert any(e.name == "LIKED" for e in edges)


def test_follow_adds_agent_edge(updater, store):
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "agent_1",
        "agent_name": "User1",
        "action_type": "FOLLOW",
        "action_args": {"target_user_name": "agent_2"},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert any(e.name == "FOLLOWS" for e in edges)


def test_do_nothing_is_skipped(updater, store):
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "agent_1",
        "agent_name": "User1",
        "action_type": "DO_NOTHING",
        "action_args": {},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    edges = store.get_entity_edges("sim-1", _find_uuid(store, "agent_1"))
    assert len(edges) == 0


def test_get_stats(updater):
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "agent_1",
        "agent_name": "User1",
        "action_type": "CREATE_POST",
        "action_args": {"content": "Test", "post_id": "p1"},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    stats = updater.get_stats()
    assert stats["total_processed"] == 1


def _find_uuid(store, name):
    entities = store.list_entities("sim-1")
    return next(e.uuid for e in entities if e.name == name)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_graph_memory_updater.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GraphMemoryUpdater**

`backend/app/services/graph_memory_updater.py`:
```python
"""
Local graph memory updater replacing ZepGraphMemoryUpdater.

Converts OASIS structured actions directly into graph edges + SQLite text records.
Zero LLM cost — all operations are pure data transformations.
"""

import threading
from datetime import datetime
from typing import Optional, Dict, Any

from .networkx_graph_store import NetworkXGraphStore


# Action type → (edge_type, needs_post_node, needs_text_index)
ACTION_MAP = {
    "CREATE_POST":     ("POSTED",        True,  True),
    "LIKE_POST":       ("LIKED",         True,  False),
    "DISLIKE_POST":    ("DISLIKED",      True,  False),
    "REPOST":          ("REPOSTED",      True,  False),
    "QUOTE_POST":      ("QUOTED",        True,  True),
    "FOLLOW":          ("FOLLOWS",       False, False),
    "CREATE_COMMENT":  ("COMMENTED",     True,  True),
    "LIKE_COMMENT":    ("LIKED",         True,  False),
    "DISLIKE_COMMENT": ("DISLIKED",      True,  False),
    "SEARCH_POSTS":    ("SEARCHED",      False, False),
    "SEARCH_USER":     ("SEARCHED_USER", False, False),
    "MUTE":            ("MUTED",         False, False),
}

SKIP_ACTIONS = {"DO_NOTHING", "REFRESH", "TREND"}


class GraphMemoryUpdater:
    """Synchronously records agent actions into GraphStore + TextStore."""

    def __init__(self, graph_id: str, store: NetworkXGraphStore):
        self.graph_id = graph_id
        self.store = store
        self._stats = {"total_processed": 0, "total_skipped": 0}
        self._lock = threading.Lock()

    def add_activity(self, activity: Dict[str, Any]) -> None:
        action_type = activity.get("action_type", "")
        if action_type in SKIP_ACTIONS:
            self._stats["total_skipped"] += 1
            return

        if action_type not in ACTION_MAP:
            self._stats["total_skipped"] += 1
            return

        edge_type, needs_post_node, needs_text = ACTION_MAP[action_type]
        args = activity.get("action_args", {})
        agent_id = activity.get("agent_id", "")
        agent_name = activity.get("agent_name", "")
        platform = activity.get("platform", "")
        round_num = activity.get("round_num", 0)
        timestamp_str = activity.get("timestamp", datetime.now().isoformat())
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now()

        # Find or create agent node
        agent_uuid = self._ensure_agent(agent_id, agent_name)

        # Determine target node
        target_uuid = self._resolve_target(
            action_type, edge_type, needs_post_node, args
        )

        if target_uuid is None:
            self._stats["total_skipped"] += 1
            return

        # Determine weight
        weight = 1.0
        if action_type in ("DISLIKE_POST", "DISLIKE_COMMENT"):
            weight = -1.0

        # Build fact description (Chinese, matching original format)
        fact = self._build_fact(action_type, agent_name, args)

        # Add graph edge
        self.store.add_relation(
            self.graph_id, agent_uuid, target_uuid,
            edge_type, fact,
            attributes={
                "action_type": action_type,
                "weight": weight,
                "tick": round_num,
                "platform": platform,
            },
            temporal={"valid_at": timestamp.isoformat()},
        )

        # Add text record for searchable content
        content = args.get("content", "") or args.get("query", "") or fact
        if content:
            self.store.add_text(
                self.graph_id, agent_id, content,
                action_type, {"agent_name": agent_name, "platform": platform},
                tick=round_num, timestamp=timestamp,
            )

        with self._lock:
            self._stats["total_processed"] += 1

    def _ensure_agent(self, agent_id: str, agent_name: str) -> str:
        """Find existing agent node or create one."""
        entities = self.store.list_entities(self.graph_id, limit=10000)
        for e in entities:
            if e.name == agent_id:
                return e.uuid
        return self.store.add_entity(
            self.graph_id, agent_id, "agent", agent_name, {}
        )

    def _resolve_target(self, action_type: str, edge_type: str,
                        needs_post_node: bool, args: dict) -> Optional[str]:
        """Resolve or create the target node for this action."""
        if action_type == "FOLLOW":
            target_name = args.get("target_user_name", "")
            if not target_name:
                return None
            return self._ensure_entity(target_name, "agent", "")
        elif action_type == "MUTE":
            target_name = args.get("target_user_name", "")
            if not target_name:
                return None
            return self._ensure_entity(target_name, "agent", "")
        elif action_type in ("SEARCH_POSTS", "SEARCH_USER"):
            query = args.get("query", "search")
            return self._ensure_entity(f"query:{query}", "query", query)
        elif needs_post_node:
            post_id = args.get("post_id", args.get("comment_id", ""))
            if not post_id:
                post_id = f"post-{self._stats['total_processed']}"
            node_type = "comment" if "COMMENT" in action_type else "post"
            content = args.get("content", "")[:100]
            return self._ensure_entity(str(post_id), node_type, content)
        return None

    def _ensure_entity(self, name: str, entity_type: str, summary: str) -> str:
        """Find existing entity or create one (dedup handled by store)."""
        return self.store.add_entity(
            self.graph_id, name, entity_type, summary, {}
        )

    def _build_fact(self, action_type: str, agent_name: str, args: dict) -> str:
        """Build Chinese fact description matching zep_graph_memory_updater format."""
        content = args.get("content", "")
        author = args.get("author", "")
        target = args.get("target_user_name", "")
        query = args.get("query", "")

        descriptions = {
            "CREATE_POST": f"发布了一条帖子：「{content[:50]}」",
            "LIKE_POST": f"点赞了{author}的帖子",
            "DISLIKE_POST": f"踩了{author}的帖子",
            "REPOST": f"转发了{author}的帖子",
            "QUOTE_POST": f"引用了{author}的帖子并评论",
            "FOLLOW": f"关注了用户「{target}」",
            "CREATE_COMMENT": f"评论道：「{content[:50]}」",
            "LIKE_COMMENT": f"点赞了{author}的评论",
            "DISLIKE_COMMENT": f"踩了{author}的评论",
            "SEARCH_POSTS": f"搜索了「{query}」",
            "SEARCH_USER": f"搜索了用户「{query}」",
            "MUTE": f"屏蔽了用户「{target}」",
        }
        desc = descriptions.get(action_type, action_type)
        return f"{agent_name}: {desc}"

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)


class GraphMemoryManager:
    """Manages memory updaters for multiple simulations.
    Replaces ZepGraphMemoryManager."""

    _updaters: Dict[str, GraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str,
                       store: NetworkXGraphStore) -> GraphMemoryUpdater:
        with cls._lock:
            if simulation_id in cls._updaters:
                pass  # Reuse or replace
            updater = GraphMemoryUpdater(graph_id=graph_id, store=store)
            cls._updaters[simulation_id] = updater
        return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[GraphMemoryUpdater]:
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str) -> None:
        with cls._lock:
            cls._updaters.pop(simulation_id, None)

    @classmethod
    def stop_all(cls) -> None:
        with cls._lock:
            cls._updaters.clear()

    @classmethod
    def get_all_stats(cls) -> dict:
        return {sid: u.get_stats() for sid, u in cls._updaters.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_graph_memory_updater.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/graph_memory_updater.py backend/tests/test_graph_memory_updater.py
git commit -m "feat: add local graph memory updater replacing ZepGraphMemoryUpdater"
```

---

## Task 5: Search Tools (InsightForge, PanoramaSearch, QuickSearch)

**Files:**
- Create: `backend/app/services/search_tools.py`
- Create: `backend/tests/test_search_tools.py`

Replaces `zep_tools.py` (1735 lines). Implements the three search strategies using GraphStore + TextStore.

- [ ] **Step 1: Write failing tests**

`backend/tests/test_search_tools.py`:
```python
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


@pytest.fixture
def store(tmp_db_path):
    from backend.app.services.networkx_graph_store import NetworkXGraphStore
    store = NetworkXGraphStore(db_path=tmp_db_path)
    gid = "sim-test"
    store.create_graph(gid, "Test Sim", "")
    # Create entities
    a1 = store.add_entity(gid, "Alice", "Person", "政策分析師", {})
    a2 = store.add_entity(gid, "Bob", "Person", "社群媒體用戶", {})
    store.add_relation(gid, a1, a2, "KNOWS", "Alice和Bob是同事")
    # Add searchable text
    store.add_text(gid, "Alice", "央行升息對房市造成衝擊", "CREATE_POST",
                   {}, tick=1, timestamp=datetime.now())
    store.add_text(gid, "Bob", "房價下跌已成趨勢", "CREATE_POST",
                   {}, tick=2, timestamp=datetime.now())
    return store


@pytest.fixture
def tools(store):
    from backend.app.services.search_tools import SearchTools
    mock_llm = MagicMock()
    # Mock LLM to return sub-questions as JSON
    mock_llm.chat_json.return_value = {
        "sub_questions": ["升息影響", "房市衝擊", "央行政策"]
    }
    return SearchTools(graph_id="sim-test", store=store, llm_client=mock_llm)


def test_quick_search_returns_results(tools):
    result = tools.quick_search("房市")
    assert len(result.facts) > 0


def test_panorama_search_returns_all(tools):
    result = tools.panorama_search("概況")
    assert len(result.nodes) > 0 or len(result.edges) > 0


def test_insight_forge_decomposes_query(tools):
    result = tools.insight_forge("央行升息對社會的整體影響是什麼?")
    assert len(result.facts) > 0
    # Verify LLM was called to decompose
    tools.llm_client.chat_json.assert_called_once()


def test_search_result_to_text(tools):
    result = tools.quick_search("房")
    text = result.to_text(query="房市")
    assert "搜索查询" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_search_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SearchTools**

`backend/app/services/search_tools.py`:
```python
"""
Search tools replacing ZepTools (InsightForge, PanoramaSearch, QuickSearch).

Uses NetworkXGraphStore for graph operations and SQLite FTS5 for text search.
"""

from typing import Optional, List, Dict, Any

from .graph_store import EntityNode, RelationEdge, SearchResult
from .networkx_graph_store import NetworkXGraphStore


class SearchTools:
    """Search engine providing InsightForge, PanoramaSearch, QuickSearch."""

    def __init__(self, graph_id: str, store: NetworkXGraphStore,
                 llm_client=None):
        self.graph_id = graph_id
        self.store = store
        self.llm_client = llm_client

    def quick_search(self, query: str, limit: int = 20) -> SearchResult:
        """Fast keyword search over text content and graph entities."""
        # FTS5 text search
        text_results = self.store.search_text(self.graph_id, query, limit=limit)
        facts = [r.get("content", "") for r in text_results]

        # Graph node/edge keyword search
        graph_result = self.store.search(self.graph_id, query, scope="all", limit=limit)
        facts.extend(graph_result.facts)

        # Deduplicate facts
        seen = set()
        unique_facts = []
        for f in facts:
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
                facts.append(f"[活跃] {e.fact}")
        for e in historical_edges:
            if e.fact:
                facts.append(f"[历史] {e.fact}")

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
        for node in all_nodes:
            edges = self.store.get_entity_edges(self.graph_id, node.uuid)
            for edge in edges:
                if edge.fact and edge.fact not in all_facts:
                    all_facts.append(edge.fact)
                if edge not in all_edges:
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

        return SearchResult(
            nodes=unique_nodes[:limit],
            edges=all_edges[:limit],
            facts=unique_facts[:limit],
        )

    def _decompose_query(self, query: str) -> List[str]:
        """Use LLM to break a complex query into sub-questions."""
        if not self.llm_client:
            # Fallback: split by common Chinese punctuation
            return [query]

        messages = [
            {"role": "system", "content": (
                "你是一个搜索查询分析器。将用户的复杂问题分解为3-5个更简单、更具体的子问题。"
                "返回JSON格式：{\"sub_questions\": [\"问题1\", \"问题2\", ...]}"
            )},
            {"role": "user", "content": query},
        ]
        try:
            result = self.llm_client.chat_json(messages, temperature=0.3)
            return result.get("sub_questions", [query])
        except Exception:
            return [query]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_search_tools.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/search_tools.py backend/tests/test_search_tools.py
git commit -m "feat: add local search tools replacing ZepTools"
```

---

## Task 6: Migrate graph_builder.py to LLM Entity Extraction

**Files:**
- Modify: `backend/app/services/graph_builder.py`
- Create: `backend/tests/test_graph_builder.py`

Replace Zep API calls with LLM-based entity extraction + GraphStore storage.

- [ ] **Step 1: Write failing tests**

`backend/tests/test_graph_builder.py`:
```python
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def store(tmp_db_path):
    from backend.app.services.networkx_graph_store import NetworkXGraphStore
    return NetworkXGraphStore(db_path=tmp_db_path)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "entities": [
            {"name": "Elon Musk", "type": "Person", "description": "CEO of Tesla"},
            {"name": "Tesla", "type": "Organization", "description": "Electric car company"},
        ],
        "relations": [
            {"source": "Elon Musk", "target": "Tesla", "type": "CEO_OF", "fact": "Elon Musk is CEO of Tesla"},
        ],
    }
    return llm


def test_build_graph_from_text(store, mock_llm):
    from backend.app.services.graph_builder import GraphBuilderService
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Test Graph")
    builder.add_text_and_extract(graph_id, "Elon Musk is the CEO of Tesla, an electric car company.")
    entities = store.list_entities(graph_id)
    names = {e.name for e in entities}
    assert "Elon Musk" in names
    assert "Tesla" in names
    relations = store.list_relations(graph_id)
    assert len(relations) >= 1


def test_build_graph_deduplicates_entities(store, mock_llm):
    from backend.app.services.graph_builder import GraphBuilderService
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Dedup Test")
    builder.add_text_and_extract(graph_id, "Chunk 1 about Elon Musk.")
    builder.add_text_and_extract(graph_id, "Chunk 2 about Elon Musk.")
    entities = store.list_entities(graph_id)
    elon_count = sum(1 for e in entities if e.name == "Elon Musk")
    assert elon_count == 1  # deduplicated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_graph_builder.py -v`
Expected: FAIL (GraphBuilderService doesn't accept store= yet)

- [ ] **Step 3: Rewrite GraphBuilderService**

Modify `backend/app/services/graph_builder.py`. Key changes:
- Remove all `from zep_cloud` imports
- Replace constructor: accept `store: NetworkXGraphStore` and `llm_client: LLMClient` instead of `api_key`
- Replace `create_graph()`: call `store.create_graph()` instead of `client.graph.create()`
- Remove `set_ontology()` (ontology now embedded in LLM prompt)
- Replace `add_text_batches()` with `add_text_and_extract()`:
  - Chunk text (keep existing TextProcessor logic)
  - For each chunk: LLM call with JSON schema for entity/relation extraction
  - Store extracted entities/relations in GraphStore
- Remove `_wait_for_episodes()` (no async polling needed)
- Replace `_get_graph_info()`: query GraphStore directly
- Replace `get_graph_data()`: query GraphStore directly
- Replace `delete_graph()`: call `store.delete_graph()`

The new `add_text_and_extract()` method should:
1. Call `self.llm_client.chat_json()` with the extraction prompt
2. Iterate returned entities, call `store.add_entity()` for each
3. Iterate returned relations, resolve entity UUIDs, call `store.add_relation()` for each

**LLM extraction prompt** (embed in method):
```python
EXTRACTION_SYSTEM_PROMPT = """你是一个知识图谱实体关系抽取引擎。
从给定文本中抽取实体和关系。

输出JSON格式：
{
  "entities": [{"name": "实体名", "type": "实体类型", "description": "简短描述"}],
  "relations": [{"source": "源实体名", "target": "目标实体名", "type": "关系类型", "fact": "关系描述"}]
}

实体类型可以是：Person, Organization, Location, Event, Product, Policy, 或其他合适的类型。
关系类型应该简短且描述性强。"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_graph_builder.py -v`
Expected: All tests PASS

- [ ] **Step 5: Verify existing GraphInfo dataclass is preserved**

Read `backend/app/services/graph_builder.py` to ensure `GraphInfo` and `to_dict()` still work, as they're used by API endpoints.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/graph_builder.py backend/tests/test_graph_builder.py
git commit -m "feat: migrate graph_builder from Zep to LLM entity extraction + GraphStore"
```

---

## Task 7: Migrate Consumers (simulation_runner, oasis_profile_generator, simulation_manager)

**Files:**
- Modify: `backend/app/services/simulation_runner.py`
- Modify: `backend/app/services/oasis_profile_generator.py`
- Modify: `backend/app/services/simulation_manager.py`

These files consume the services we just rebuilt. Changes are primarily import swaps and constructor argument updates.

- [ ] **Step 1: Migrate simulation_runner.py**

Replace:
```python
from .zep_graph_memory_updater import ZepGraphMemoryManager
```
With:
```python
from .graph_memory_updater import GraphMemoryManager
```

Find all references to `ZepGraphMemoryManager` and replace with `GraphMemoryManager`. The `create_updater` call needs an additional `store` parameter:
```python
# Old:
updater = ZepGraphMemoryManager.create_updater(simulation_id, graph_id)
# New:
updater = GraphMemoryManager.create_updater(simulation_id, graph_id, store=self.store)
```

The `SimulationRunner` class needs a `store` attribute. Add it to the constructor — trace how `SimulationRunner` is instantiated (likely in `simulation_manager.py` or API endpoints) and pass the store through.

- [ ] **Step 2: Migrate oasis_profile_generator.py**

Replace:
```python
from zep_cloud.client import Zep
from .zep_entity_reader import EntityNode, ZepEntityReader
```
With:
```python
from .graph_store import EntityNode
from .entity_reader import EntityReader
from .networkx_graph_store import NetworkXGraphStore
```

Update constructor: replace `self.client = Zep(...)` with `self.store = store` parameter.

Replace secondary Zep enrichment (parallel `graph.search(scope="edges")` and `graph.search(scope="nodes")`) with:
```python
result = self.store.search(self.graph_id, entity_name, scope="all", limit=10)
```

- [ ] **Step 3: Migrate simulation_manager.py**

Replace:
```python
from .zep_entity_reader import ZepEntityReader
```
With:
```python
from .entity_reader import EntityReader
```

Replace:
```python
reader = ZepEntityReader()
filtered = reader.filter_defined_entities(graph_id=..., ...)
```
With:
```python
reader = EntityReader(store=self.store)
filtered = reader.filter_defined_entities(graph_id=..., ...)
```

Add `store` parameter to `SimulationManager` constructor. The store instance should be created once at app startup (in `__init__.py` or `api/` blueprints).

- [ ] **Step 4: Run import check**

Run: `cd /home/endea/MiroFish && python -c "from backend.app.services.simulation_manager import SimulationManager; print('OK')"`
Expected: `OK` (no import errors)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/simulation_runner.py backend/app/services/oasis_profile_generator.py backend/app/services/simulation_manager.py
git commit -m "feat: migrate simulation runner, profile generator, manager to local GraphStore"
```

---

## Task 8: Migrate report_agent.py and search tools wiring

**Files:**
- Modify: `backend/app/services/report_agent.py`

- [ ] **Step 1: Update report_agent.py imports**

Replace:
```python
from .zep_tools import ZepTools  # or however it's imported
```
With:
```python
from .search_tools import SearchTools
```

Find where `ZepTools` is instantiated and replace with:
```python
tools = SearchTools(graph_id=self.graph_id, store=self.store, llm_client=self.llm_client)
```

The `ReportAgent` class needs a `store` parameter in its constructor (replacing `api_key`).

- [ ] **Step 2: Verify tool method signatures match**

The `SearchTools` class exposes `insight_forge()`, `panorama_search()`, `quick_search()` — confirm these match what `report_agent.py` calls. If the original methods have different names or signatures, update `search_tools.py` to match, or update the calls in `report_agent.py`.

Read `report_agent.py` to find exact method calls and adapt.

- [ ] **Step 3: Run import check**

Run: `cd /home/endea/MiroFish && python -c "from backend.app.services.report_agent import ReportAgent; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/report_agent.py
git commit -m "feat: migrate report agent from ZepTools to local SearchTools"
```

---

## Task 9: Migrate API Endpoints and Cleanup

**Files:**
- Modify: `backend/app/api/graph.py`
- Modify: `backend/app/api/simulation.py`
- Modify: `backend/app/api/report.py`
- Modify: `backend/app/services/ontology_generator.py`
- Modify: `backend/app/services/__init__.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/__init__.py`

- [ ] **Step 1: Create shared GraphStore instance**

In `backend/app/__init__.py` (the Flask app factory), create a single `NetworkXGraphStore` instance that all blueprints share:

```python
from .services.networkx_graph_store import NetworkXGraphStore

# Create at module level or in create_app()
graph_store = NetworkXGraphStore(db_path="uploads/graph_store.db")
```

Pass this to service constructors via Flask's `g` object or directly.

- [ ] **Step 2: Migrate api/graph.py**

Remove all `Config.ZEP_API_KEY` checks. Replace:
```python
builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
```
With:
```python
from .. import graph_store
from ..utils.llm_client import LLMClient
builder = GraphBuilderService(store=graph_store, llm_client=LLMClient())
```

- [ ] **Step 3: Migrate api/simulation.py**

Remove `ZEP_API_KEY` checks. Replace `ZepEntityReader()` with `EntityReader(store=graph_store)`.

- [ ] **Step 4: Migrate api/report.py**

Remove `ZEP_API_KEY` checks. Pass `store=graph_store` to `ReportAgent`.

- [ ] **Step 5: Update ontology_generator.py**

Remove the `from zep_cloud.external_clients.ontology import ...` reference.
Remove or relax the max-10 entity type constraint (no longer limited by Zep API).

- [ ] **Step 6: Update services/__init__.py exports**

Replace:
```python
from .zep_entity_reader import ZepEntityReader, EntityNode, FilteredEntities
from .zep_graph_memory_updater import ZepGraphMemoryUpdater, ZepGraphMemoryManager, AgentActivity
```
With:
```python
from .graph_store import EntityNode
from .entity_reader import EntityReader, FilteredEntities
from .graph_memory_updater import GraphMemoryUpdater, GraphMemoryManager
```

- [ ] **Step 7: Remove ZEP config from config.py**

Remove from `backend/app/config.py`:
```python
ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
ZEP_REQUEST_TIMEOUT_SECONDS = float(os.environ.get('ZEP_REQUEST_TIMEOUT_SECONDS', '30'))
```

- [ ] **Step 8: Run full import check**

```bash
cd /home/endea/MiroFish && python -c "
from backend.app.services import EntityReader, FilteredEntities, GraphMemoryUpdater, GraphMemoryManager
from backend.app.services.graph_builder import GraphBuilderService
from backend.app.services.search_tools import SearchTools
from backend.app.services.report_agent import ReportAgent
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 9: Commit**

```bash
git add backend/app/api/ backend/app/services/__init__.py backend/app/services/ontology_generator.py backend/app/config.py backend/app/__init__.py
git commit -m "feat: migrate API endpoints and service exports to local GraphStore"
```

---

## Task 10: Delete Zep Files and Remove Dependency

**Files:**
- Delete: `backend/app/utils/zep_paging.py`
- Delete: `backend/app/services/zep_entity_reader.py`
- Delete: `backend/app/services/zep_graph_memory_updater.py`
- Delete: `backend/app/services/zep_tools.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Verify no remaining Zep imports**

```bash
cd /home/endea/MiroFish && grep -r "from zep_cloud" backend/app/ --include="*.py" || echo "No Zep imports found"
cd /home/endea/MiroFish && grep -r "import zep" backend/app/ --include="*.py" || echo "No Zep imports found"
cd /home/endea/MiroFish && grep -r "from.*zep_entity_reader" backend/app/ --include="*.py" || echo "No old reader imports"
cd /home/endea/MiroFish && grep -r "from.*zep_graph_memory_updater" backend/app/ --include="*.py" || echo "No old updater imports"
cd /home/endea/MiroFish && grep -r "from.*zep_tools" backend/app/ --include="*.py" || echo "No old tools imports"
cd /home/endea/MiroFish && grep -r "from.*zep_paging" backend/app/ --include="*.py" || echo "No old paging imports"
```
Expected: All lines report "No ... found"

- [ ] **Step 2: Delete old Zep files**

```bash
rm backend/app/utils/zep_paging.py
rm backend/app/services/zep_entity_reader.py
rm backend/app/services/zep_graph_memory_updater.py
rm backend/app/services/zep_tools.py
```

- [ ] **Step 3: Update requirements.txt**

Remove:
```
zep-cloud==3.13.0
```

Add (if not already present):
```
networkx>=3.0
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/endea/MiroFish && python -m pytest backend/tests/ -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: remove all Zep Cloud dependencies

BREAKING CHANGE: zep-cloud removed from requirements.
ZEP_API_KEY no longer needed in .env configuration.
All graph operations now use local NetworkX + SQLite."
```

---

## Task 11: Integration Smoke Test

**Files:**
- Create: `backend/tests/test_integration.py`

End-to-end test of the full workflow: document upload → entity extraction → profile generation setup → simulation memory recording → search.

- [ ] **Step 1: Write integration test**

`backend/tests/test_integration.py`:
```python
"""
End-to-end integration test for the de-Zep'd pipeline.
Tests the full flow: graph building → entity reading → memory updates → search.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock


@pytest.fixture
def store(tmp_db_path):
    from backend.app.services.networkx_graph_store import NetworkXGraphStore
    return NetworkXGraphStore(db_path=tmp_db_path)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mock for entity extraction
    llm.chat_json.side_effect = [
        # First call: entity extraction from document
        {
            "entities": [
                {"name": "央行", "type": "Organization", "description": "中央銀行"},
                {"name": "升息政策", "type": "Policy", "description": "央行升息決策"},
                {"name": "房市", "type": "Topic", "description": "房地產市場"},
            ],
            "relations": [
                {"source": "央行", "target": "升息政策", "type": "IMPLEMENTS", "fact": "央行實施升息政策"},
                {"source": "升息政策", "target": "房市", "type": "IMPACTS", "fact": "升息政策影響房市"},
            ],
        },
        # Second call: InsightForge sub-question decomposition
        {"sub_questions": ["升息影響", "房市變化", "央行決策"]},
    ]
    return llm


def test_full_pipeline(store, mock_llm):
    from backend.app.services.graph_builder import GraphBuilderService
    from backend.app.services.entity_reader import EntityReader
    from backend.app.services.graph_memory_updater import GraphMemoryUpdater
    from backend.app.services.search_tools import SearchTools

    # Step 1: Build graph from document
    builder = GraphBuilderService(store=store, llm_client=mock_llm)
    graph_id = builder.create_graph("Integration Test")
    builder.add_text_and_extract(graph_id, "央行宣布升息，預計將對房市造成衝擊。")

    # Step 2: Read entities
    reader = EntityReader(store)
    filtered = reader.filter_defined_entities(graph_id)
    assert filtered.filtered_count >= 3
    assert "Organization" in filtered.entity_types

    # Step 3: Simulate agent memory updates
    updater = GraphMemoryUpdater(graph_id=graph_id, store=store)
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "analyst_1",
        "agent_name": "分析師小王",
        "action_type": "CREATE_POST",
        "action_args": {"content": "央行升息後房價恐下跌20%", "post_id": "p1"},
        "round_num": 1,
        "timestamp": datetime.now().isoformat(),
    })
    updater.add_activity({
        "platform": "twitter",
        "agent_id": "citizen_1",
        "agent_name": "市民小李",
        "action_type": "LIKE_POST",
        "action_args": {"post_id": "p1", "author": "分析師小王", "content": "房價恐下跌"},
        "round_num": 2,
        "timestamp": datetime.now().isoformat(),
    })

    # Step 4: Search
    tools = SearchTools(graph_id=graph_id, store=store, llm_client=mock_llm)
    result = tools.quick_search("房價")
    assert len(result.facts) > 0

    # Step 5: Community detection (need at least 2 connected nodes)
    communities = store.detect_communities(graph_id)
    assert len(communities) >= 1  # at least one community

    # Step 6: Snapshot and restore
    snapshot = store.snapshot(graph_id)
    restored_id = store.restore(snapshot)
    restored_entities = store.list_entities(restored_id)
    assert len(restored_entities) >= 3
```

- [ ] **Step 2: Run integration test**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /home/endea/MiroFish && python -m pytest backend/tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_integration.py
git commit -m "test: add end-to-end integration test for de-Zep pipeline"
```

---

## Summary

| Task | Description | New/Modified Files | Tests |
|------|-------------|-------------------|-------|
| 1 | Data models + protocols | `graph_store.py` | 7 tests |
| 2 | NetworkX+SQLite implementation | `networkx_graph_store.py` | 10+ tests |
| 3 | Local entity reader | `entity_reader.py` | 4 tests |
| 4 | Local memory updater | `graph_memory_updater.py` | 5 tests |
| 5 | Search tools | `search_tools.py` | 4 tests |
| 6 | Migrate graph_builder | `graph_builder.py` | 2 tests |
| 7 | Migrate consumers | `simulation_runner.py`, `oasis_profile_generator.py`, `simulation_manager.py` | import check |
| 8 | Migrate report_agent | `report_agent.py` | import check |
| 9 | Migrate API + cleanup | 7 files | import check |
| 10 | Delete Zep files + dependency | 4 deletions, `requirements.txt` | grep check |
| 11 | Integration smoke test | `test_integration.py` | 1 e2e test |
