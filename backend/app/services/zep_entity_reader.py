"""
Entity Reader & Filter Service
Reads nodes from the local DuckDB graph, filters by entity types
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..config import Config
from ..utils.local_graph_store import LocalGraphStore
from ..utils.logger import get_logger

logger = get_logger('mirofish.zep_entity_reader')


@dataclass
class EntityNode:
    """Entity node data structure"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        for label in self.labels:
            if label not in ("Entity", "Node"):
                return label
        return None


@dataclass
class FilteredEntities:
    """Filtered entity collection"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    Entity Reader & Filter Service

    Reads all nodes from the local DuckDB graph, filters entities
    with meaningful labels (beyond Entity/Node), and enriches with edges.
    """

    def __init__(self, storage_dir: Optional[str] = None, api_key: Optional[str] = None):
        storage_dir = storage_dir or Config.GRAPH_STORAGE_DIR
        self.store = LocalGraphStore(storage_dir)

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching all nodes for graph {graph_id}...")
        nodes = self.store.get_nodes(graph_id)
        logger.info(f"Fetched {len(nodes)} nodes")
        return nodes

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching all edges for graph {graph_id}...")
        edges = self.store.get_edges(graph_id)
        logger.info(f"Fetched {len(edges)} edges")
        return edges

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        try:
            return self.store.get_node_edges(graph_id, node_uuid)
        except Exception as e:
            logger.warning(f"Failed to get edges for node {node_uuid}: {e}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
        max_edges_per_entity: int = 30,
        max_related_nodes: int = 20
    ) -> FilteredEntities:
        logger.info(f"Filtering entities for graph {graph_id}...")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        node_map = {n["uuid"]: n for n in all_nodes}

        edge_index: Dict[str, List[Dict[str, Any]]] = {}
        if enrich_with_edges and all_edges:
            for edge in all_edges:
                src = edge.get("source_node_uuid")
                tgt = edge.get("target_node_uuid")
                if src:
                    edge_index.setdefault(src, []).append({**edge, "_direction": "outgoing"})
                if tgt:
                    edge_index.setdefault(tgt, []).append({**edge, "_direction": "incoming"})

        filtered_entities = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels") or []
            custom_labels = [l for l in labels if l not in ("Entity", "Node")]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching = [l for l in custom_labels if l in defined_entity_types]
                if not matching:
                    continue
                entity_type = matching[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node.get("name", ""),
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids: Set[str] = set()
                node_uuid = node["uuid"]

                node_edges = edge_index.get(node_uuid, [])
                for edge in node_edges[:max_edges_per_entity]:
                    direction = edge["_direction"]
                    if direction == "outgoing":
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge.get("name", ""),
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge.get("target_node_uuid", ""),
                        })
                        related_node_uuids.add(edge.get("target_node_uuid", ""))
                    else:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge.get("name", ""),
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge.get("source_node_uuid", ""),
                        })
                        related_node_uuids.add(edge.get("source_node_uuid", ""))

                entity.related_edges = related_edges

                related_nodes = []
                count = 0
                for related_uuid in related_node_uuids:
                    if count >= max_related_nodes:
                        break
                    if related_uuid and related_uuid in node_map:
                        rn = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": rn["uuid"],
                            "name": rn.get("name", ""),
                            "labels": rn.get("labels", []),
                            "summary": rn.get("summary", ""),
                        })
                        count += 1
                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(f"Filtering complete: {total_count} total, {len(filtered_entities)} matched, "
                    f"types: {entity_types_found}")

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str,
        max_edges_per_entity: int = 30,
        max_related_nodes: int = 20
    ) -> Optional[EntityNode]:
        try:
            node = self.store.get_node(graph_id, entity_uuid)
            if not node:
                return None

            edges = self.get_node_edges(graph_id, entity_uuid)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}

            related_edges = []
            related_node_uuids: Set[str] = set()

            for edge in edges[:max_edges_per_entity]:
                if edge.get("source_node_uuid") == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge.get("name", ""),
                        "fact": edge.get("fact", ""),
                        "target_node_uuid": edge.get("target_node_uuid", ""),
                    })
                    related_node_uuids.add(edge.get("target_node_uuid", ""))
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge.get("name", ""),
                        "fact": edge.get("fact", ""),
                        "source_node_uuid": edge.get("source_node_uuid", ""),
                    })
                    related_node_uuids.add(edge.get("source_node_uuid", ""))

            related_nodes = []
            count = 0
            for related_uuid in related_node_uuids:
                if count >= max_related_nodes:
                    break
                if related_uuid and related_uuid in node_map:
                    rn = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": rn["uuid"],
                        "name": rn.get("name", ""),
                        "labels": rn.get("labels", []),
                        "summary": rn.get("summary", ""),
                    })
                    count += 1

            return EntityNode(
                uuid=node["uuid"],
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Failed to get entity {entity_uuid}: {e}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
