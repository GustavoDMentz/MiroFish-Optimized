"""
Graph Builder Service
Builds a standalone knowledge graph using local DuckDB storage
"""

import os
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.local_graph_store import LocalGraphStore
from ..utils.llm_client import LLMClient
from .text_processor import TextProcessor
from ..utils.locale import t, get_locale, set_locale, get_language_instruction
from ..utils.logger import get_logger

logger = get_logger('mirofish.graph_builder')


@dataclass
class GraphInfo:
    """Graph information"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph Builder Service
    Builds a knowledge graph using local DuckDB storage
    """

    def __init__(self, storage_dir: Optional[str] = None, api_key: Optional[str] = None):
        self.storage_dir = storage_dir or Config.GRAPH_STORAGE_DIR
        self.store = LocalGraphStore(self.storage_dir)
        self.task_manager = TaskManager()
        self._llm: Optional[LLMClient] = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        current_locale = get_locale()

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = 'zh'
    ):
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t('progress.startBuildingGraph')
            )

            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t('progress.graphCreated', graphId=graph_id)
            )

            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t('progress.ontologySet')
            )

            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t('progress.textSplit', count=total_chunks)
            )

            self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.7),
                    message=msg
                )
            )

            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t('progress.fetchingGraphInfo')
            )

            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.store.create_graph(graph_id, name, "MiroFish Social Simulation Graph")
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        self.store.set_ontology(graph_id, ontology)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        episode_uuids = []
        ontology = self.store.get_ontology(graph_id) or {}
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch)) / total_chunks
                progress_callback(
                    t('progress.sendingBatch', current=batch_num, total=total_batches, chunks=len(batch)),
                    progress
                )

            for text in batch:
                ep_uuid = self.store.add_episode(graph_id, text)
                episode_uuids.append(ep_uuid)

            if ontology.get("entity_types") or ontology.get("edge_types"):
                try:
                    extracted = self._extract_entities_from_batch(batch, ontology)
                    self._store_extracted(graph_id, extracted)
                except Exception as e:
                    logger.warning(f"Batch {batch_num} entity extraction failed: {e}")

            time.sleep(0.3)

        return episode_uuids

    def _extract_entities_from_batch(self, texts: List[str], ontology: Dict[str, Any]) -> Dict[str, Any]:
        combined_text = "\n\n".join(texts)

        entity_types_desc = "\n".join(
            f"- {et['name']}: {et.get('description', '')}"
            for et in ontology.get("entity_types", [])
        ) or "- Entity (generic)"

        edge_types_desc = "\n".join(
            f"- {rt['name']}: {rt.get('description', '')}"
            for rt in ontology.get("edge_types", [])
        ) or "- RELATED_TO"

        user_prompt = f"""Extract entities and relationships from the following text, using ONLY the given ontology types.

Entity Types (only use these):
{entity_types_desc}

Relationship Types (only use these):
{edge_types_desc}

Text:
{combined_text[:4000]}

Return JSON format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Entity Type", "summary": "One sentence description", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source": "Source Entity Name", "target": "Target Entity Name", "type": "Relationship Type", "fact": "Fact description"}}
  ]
}}

Rules:
- Use only the entity types and relationship types defined in the ontology.
- Entity names should be concrete (names of people, locations, organizations, etc.).
- The 'fact' field should be a concise statement of fact.
- If no matches are found, return an empty list.

{get_language_instruction()}"""

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.1
            )
            return result if isinstance(result, dict) else {"entities": [], "relationships": []}
        except Exception as e:
            logger.warning(f"Entity extraction LLM call failed: {e}")
            return {"entities": [], "relationships": []}

    def _store_extracted(self, graph_id: str, extracted: Dict[str, Any]):
        entities = extracted.get("entities", []) or []
        relationships = extracted.get("relationships", []) or []

        name_to_uuid: Dict[str, str] = {}

        for entity in entities:
            name = (entity.get("name") or "").strip()
            if not name:
                continue
            entity_type = entity.get("type") or "Entity"
            summary = entity.get("summary") or ""
            attributes = entity.get("attributes") or {}

            labels = [entity_type, "Entity"] if entity_type != "Entity" else ["Entity"]
            node_uuid = self.store.upsert_node(
                graph_id=graph_id,
                name=name,
                labels=labels,
                summary=summary,
                attributes=attributes,
            )
            name_to_uuid[name.lower()] = node_uuid

        for rel in relationships:
            source_name = (rel.get("source") or "").strip()
            target_name = (rel.get("target") or "").strip()
            rel_type = rel.get("type") or "RELATED_TO"
            fact = rel.get("fact") or ""

            if not source_name or not target_name or not fact:
                continue

            source_uuid = name_to_uuid.get(source_name.lower()) or \
                self.store.upsert_node(graph_id, source_name, ["Entity"])
            name_to_uuid[source_name.lower()] = source_uuid

            target_uuid = name_to_uuid.get(target_name.lower()) or \
                self.store.upsert_node(graph_id, target_name, ["Entity"])
            name_to_uuid[target_name.lower()] = target_uuid

            self.store.add_fact_edge(
                graph_id=graph_id,
                source_uuid=source_uuid,
                target_uuid=target_uuid,
                name=rel_type,
                fact=fact,
            )

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600
    ):
        if progress_callback:
            progress_callback(t('progress.processingComplete',
                                completed=len(episode_uuids),
                                total=len(episode_uuids)), 1.0)

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        nodes = self.store.get_nodes(graph_id)
        edges = self.store.get_edges(graph_id)

        entity_types = set()
        for node in nodes:
            for label in (node.get("labels") or []):
                if label not in ("Entity", "Node"):
                    entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        nodes = self.store.get_nodes(graph_id)
        edges = self.store.get_edges(graph_id)

        node_map = {n["uuid"]: n.get("name", "") for n in nodes}

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": edge.get("uuid", ""),
                "name": edge.get("name", ""),
                "fact": edge.get("fact", ""),
                "fact_type": edge.get("name", ""),
                "source_node_uuid": edge.get("source_node_uuid", ""),
                "target_node_uuid": edge.get("target_node_uuid", ""),
                "source_node_name": node_map.get(edge.get("source_node_uuid", ""), ""),
                "target_node_name": node_map.get(edge.get("target_node_uuid", ""), ""),
                "attributes": edge.get("attributes", {}),
                "created_at": edge.get("created_at"),
                "valid_at": edge.get("valid_at"),
                "invalid_at": edge.get("invalid_at"),
                "expired_at": edge.get("expired_at"),
                "episodes": [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges_data,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def delete_graph(self, graph_id: str):
        self.store.delete_graph(graph_id)
