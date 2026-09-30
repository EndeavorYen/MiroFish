"""Heavy OASIS dependencies loaded only when used (#65).

``oasis.social_agent.agent_graph`` imports the neo4j driver at module level
(about 80 MB and 0.4 s) for its optional Neo4j graph backend; MiroFish uses
the igraph backend. ``defer_neo4j()`` installs a stand-in whose
``GraphDatabase`` loads the real driver the first time it is called.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types


class _DeferredGraphDatabase:
    def __getattr__(self, name: str):
        return getattr(_real_neo4j().GraphDatabase, name)


def _real_neo4j() -> types.ModuleType:
    if isinstance(sys.modules.get("neo4j"), _StandIn):
        del sys.modules["neo4j"]
    return importlib.import_module("neo4j")


class _StandIn(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):  # import-system and introspection probes
            raise AttributeError(name)
        return getattr(_real_neo4j(), name)


def defer_neo4j() -> None:
    """Before oasis is imported: neo4j loads on first use instead."""

    if "neo4j" in sys.modules:
        return
    spec = importlib.util.find_spec("neo4j")
    if spec is None:
        return
    stand_in = _StandIn("neo4j")
    # A package, so `from neo4j.exceptions import ...` still finds submodules.
    stand_in.__path__ = list(spec.submodule_search_locations or [])
    stand_in.GraphDatabase = _DeferredGraphDatabase()
    sys.modules["neo4j"] = stand_in
