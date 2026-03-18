"""
Local entity reader replacing ZepEntityReader.
Reads entities from a GraphStore, filters by type, enriches with edges/neighbors.
"""
from dataclasses import dataclass
from typing import Optional, List, Set
from .graph_store import EntityNode
from .networkx_graph_store import NetworkXGraphStore

GENERIC_TYPES = {"Entity", "Node"}

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

class EntityReader:
    def __init__(self, store: NetworkXGraphStore):
        self.store = store

    def filter_defined_entities(self, graph_id, defined_entity_types=None, enrich_with_edges=False):
        all_entities = self.store.list_entities(graph_id, limit=10000)
        total_count = len(all_entities)
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

        if enrich_with_edges:
            for entity in filtered:
                edges = self.store.get_entity_edges(graph_id, entity.uuid)
                entity.related_edges = [
                    {"direction": "outgoing" if e.source_node_uuid == entity.uuid else "incoming",
                     "edge_name": e.name, "fact": e.fact,
                     "source_node_uuid": e.source_node_uuid, "target_node_uuid": e.target_node_uuid}
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
                        entity.related_nodes.append({"uuid": n.uuid, "name": n.name, "labels": n.labels, "summary": n.summary})
                    except KeyError:
                        pass

        return FilteredEntities(entities=filtered, entity_types=entity_types,
                                total_count=total_count, filtered_count=len(filtered))

    def get_entity_with_context(self, graph_id, entity_uuid):
        try:
            return self.store.get_entity(graph_id, entity_uuid)
        except KeyError:
            return None

    def get_entities_by_type(self, graph_id, entity_type):
        result = self.filter_defined_entities(graph_id, defined_entity_types=[entity_type], enrich_with_edges=True)
        return result.entities
