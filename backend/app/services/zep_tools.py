"""
Zep Tools Service (local graph store edition)
Wraps graph search, node/edge reading into tools for the Report Agent.

Core tools:
1. InsightForge - Deep insight retrieval, auto-generates sub-queries
2. PanoramaSearch - Broad search including expired content
3. QuickSearch - Simple keyword search
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..config import Config
from ..utils.local_graph_store import LocalGraphStore
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t, get_language_instruction

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant items"]

        if self.facts:
            text_parts.append("\n### Relevant Facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node info"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }

    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge info"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relation: {source} --[{self.name}]--> {target}\nFact: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "Present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """Deep insight retrieval result"""
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)

    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        text_parts = [
            f"## Deep Insight Analysis",
            f"Query: {self.query}",
            f"Scenario: {self.simulation_requirement}",
            f"\n### Stats",
            f"- Facts: {self.total_facts}",
            f"- Entities: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}"
        ]

        if self.sub_queries:
            text_parts.append(f"\n### Sub-queries")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        if self.semantic_facts:
            text_parts.append(f"\n### Key Facts")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")

        if self.entity_insights:
            text_parts.append(f"\n### Core Entities")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: \"{entity.get('summary')}\"")

        if self.relationship_chains:
            text_parts.append(f"\n### Relationship Chains")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """Broad search result, includes expired content"""
    query: str

    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)

    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        text_parts = [
            f"## Panorama Search Results",
            f"Query: {self.query}",
            f"\n### Stats",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Active facts: {self.active_count}",
            f"- Historical facts: {self.historical_count}"
        ]

        if self.active_facts:
            text_parts.append(f"\n### Active Facts")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")

        if self.historical_facts:
            text_parts.append(f"\n### Historical/Expired Facts")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")

        if self.all_nodes:
            text_parts.append(f"\n### Entities (top 25 by relevance)")
            query_lower = self.query.lower()
            keywords = [w.strip() for w in query_lower.replace(',', ' ').split() if len(w.strip()) > 1]

            def node_relevance(node) -> int:
                score = 0
                name_lower = node.name.lower()
                if query_lower in name_lower:
                    score += 100
                for kw in keywords:
                    if kw in name_lower:
                        score += 10
                return score

            sorted_nodes = sorted(self.all_nodes, key=node_relevance, reverse=True)
            for node in sorted_nodes[:25]:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Single agent interview result"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key Quotes:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """Interview result with multiple agent interviews"""
    interview_topic: str
    interview_questions: List[str]

    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)

    selection_reasoning: str = ""
    summary: str = ""

    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        text_parts = [
            "## Deep Interview Report",
            f"**Topic:** {self.interview_topic}",
            f"**Interviewees:** {self.interviewed_count} / {self.total_agents} agents",
            "\n### Selection Rationale",
            self.selection_reasoning or "(Auto-selected)",
            "\n---",
            "\n### Interview Transcript",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interviews)\n\n---")

        text_parts.append("\n### Summary")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Graph search tools service backed by LocalGraphStore (DuckDB).

    Core tools:
    1. insight_forge - Deep insight retrieval with auto sub-query generation
    2. panorama_search - Broad search including expired content
    3. quick_search - Simple keyword search
    4. interview_agents - Interview simulation agents

    Base tools:
    - search_graph - Keyword search
    - get_all_nodes - All nodes
    - get_all_edges - All edges with temporal info
    - get_node_detail - Single node detail
    - get_node_edges - Node's edges
    - get_entities_by_type - Filter by type
    - get_entity_summary - Entity relationship summary
    """

    def __init__(self, storage_dir: Optional[str] = None, api_key: Optional[str] = None,
                 llm_client: Optional[LLMClient] = None):
        storage_dir = storage_dir or Config.GRAPH_STORAGE_DIR
        self.store = LocalGraphStore(storage_dir)
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))
        return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        logger.info(t("console.usingLocalSearch", query=query[:30]))

        try:
            raw = self.store.search(graph_id, query, limit=limit, scope=scope)

            facts = raw.get("facts", [])
            edges_result = raw.get("edges", [])
            nodes_result = raw.get("nodes", [])

            logger.info(t("console.localSearchComplete", count=len(facts)))
        except Exception as e:
            logger.error(t("console.localSearchFailed", error=str(e)))
            facts, edges_result, nodes_result = [], [], []

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        logger.info(t("console.fetchingAllNodes", graphId=graph_id))

        nodes = self.store.get_nodes(graph_id)
        result = [
            NodeInfo(
                uuid=n.get("uuid", ""),
                name=n.get("name", ""),
                labels=n.get("labels") or [],
                summary=n.get("summary", ""),
                attributes=n.get("attributes") or {},
            )
            for n in nodes
        ]

        logger.info(t("console.fetchedNodes", count=len(result)))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        logger.info(t("console.fetchingAllEdges", graphId=graph_id))

        edges = self.store.get_edges(graph_id)
        result = []
        for e in edges:
            edge_info = EdgeInfo(
                uuid=e.get("uuid", ""),
                name=e.get("name", ""),
                fact=e.get("fact", ""),
                source_node_uuid=e.get("source_node_uuid", ""),
                target_node_uuid=e.get("target_node_uuid", ""),
            )
            if include_temporal:
                edge_info.created_at = e.get("created_at")
                edge_info.valid_at = e.get("valid_at")
                edge_info.invalid_at = e.get("invalid_at")
                edge_info.expired_at = e.get("expired_at")
            result.append(edge_info)

        logger.info(t("console.fetchedEdges", count=len(result)))
        return result

    def get_node_detail(self, node_uuid: str, graph_id: str = "") -> Optional[NodeInfo]:
        logger.info(t("console.fetchingNodeDetail", uuid=node_uuid[:8]))

        try:
            if graph_id:
                n = self.store.get_node(graph_id, node_uuid)
                if n:
                    return NodeInfo(
                        uuid=n.get("uuid", ""),
                        name=n.get("name", ""),
                        labels=n.get("labels") or [],
                        summary=n.get("summary", ""),
                        attributes=n.get("attributes") or {},
                    )
            return None
        except Exception as e:
            logger.error(t("console.fetchNodeDetailFailed", error=str(e)))
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        logger.info(t("console.fetchingNodeEdges", uuid=node_uuid[:8]))

        try:
            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)

            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result

        except Exception as e:
            logger.warning(t("console.fetchNodeEdgesFailed", error=str(e)))
            return []

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        logger.info(t("console.fetchingEntitiesByType", type=entity_type))

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:
            if entity_type in node.labels:
                filtered.append(node)

        logger.info(t("console.foundEntitiesByType", count=len(filtered), type=entity_type))
        return filtered

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        logger.info(t("console.fetchingEntitySummary", name=entity_name))

        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )

        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        logger.info(t("console.fetchingGraphStats", graphId=graph_id))

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        logger.info(t("console.fetchingSimContext", requirement=simulation_requirement[:50]))

        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )

        stats = self.get_graph_statistics(graph_id)

        all_nodes = self.get_all_nodes(graph_id)

        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }

    # ========== Core retrieval tools ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        logger.info(t("console.insightForgeStart", query=query[:50]))

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )

        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))

        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)

        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        entity_insights = []
        node_map = {}

        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid, graph_id=graph_id)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")

                    related_facts = [
                        f for f in all_facts
                        if node.name.lower() in f.lower()
                    ]

                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"Failed to get node {uuid}: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')

                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(t("console.insightForgeComplete", facts=result.total_facts, entities=result.total_entities, relationships=result.total_relationships))
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        system_prompt = f"""You are a professional problem analysis expert. Your task is to decompose a complex query into multiple sub-queries that can be independently observed in the simulated world.

Requirements:
1. Each sub-query must be concrete enough to find relevant Agent behaviors or events in the simulated world.
2. Sub-queries should cover different dimensions of the original query (e.g. who, what, why, how, when, where).
3. Sub-queries must be relevant to the simulation scenario.
4. Return JSON format: {{"sub_queries": ["sub-query 1", "sub-query 2", ...]}}

{get_language_instruction()}"""

        user_prompt = f"""Simulation Requirement Context:
{simulation_requirement}

{f"Report Context: {report_context[:500]}" if report_context else ""}

Please decompose the following query into {max_queries} sub-queries:
{query}

Return the list of sub-queries in JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            return [
                query,
                f"Key participants in {query}",
                f"Causes and impacts of {query}",
                f"Development process of {query}"
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        logger.info(t("console.panoramaSearchStart", query=query[:50]))

        result = PanoramaResult(query=query)

        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]

            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(t("console.panoramaSearchComplete", active=result.active_count, historical=result.historical_count))
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        logger.info(t("console.quickSearchStart", query=query[:50]))

        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )

        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        from .simulation_runner import SimulationRunner

        logger.info(t("console.interviewAgentsStart", requirement=interview_requirement[:50]))

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )

        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = "No agent profile files found for interview."
            return result

        result.total_agents = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))

        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(t("console.selectedAgentsForInterview", count=len(selected_agents), indices=selected_indices))

        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(t("console.generatedInterviewQuestions", count=len(result.interview_questions)))

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Based on your persona, all past memories and actions, "
            "answer the following questions directly in plain text.\n"
            "Response requirements:\n"
            "1. Answer in natural language directly, do not call any tools\n"
            "2. Do not return JSON or tool call format\n"
            "3. Do not use Markdown headings (like #, ##, ###)\n"
            "4. Answer each question in order, start each with 「Question X:」(X is the question number)\n"
            "5. Separate answers with blank lines\n"
            "6. Each answer should be at least 2-3 sentences\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}\n\n{get_language_instruction()}"

        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt
                })

            logger.info(t("console.callingBatchInterviewApi", count=len(interviews_request)))

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )

            logger.info(t("console.interviewApiReturned", count=api_result.get('interviews_count', 0), success=api_result.get('success')))

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning(t("console.interviewApiReturnedFailure", error=error_msg))
                result.summary = f"Interview API call failed: {error_msg}. Please check the OASIS simulation environment status."
                return result

            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                twitter_text = twitter_response if twitter_response else " (No response from this platform) "
                reddit_text = reddit_response if reddit_response else " (No response from this platform) "
                response_text = f"[Twitter Platform Response]\n{twitter_text}\n\n[Reddit Platform Response]\n{reddit_text}"

                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'(问题|Question|Pergunta)\d+[：:]\s*', '', clean_text, flags=re.IGNORECASE)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = f"Interview failed: {str(e)}. The simulation environment may have closed. Please ensure the OASIS environment is running."
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"An error occurred during the interview process: {str(e)}"
            return result

        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )

        logger.info(t("console.interviewAgentsComplete", count=result.interviewed_count))
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    # ── Methods called by interview_agents (from original) ─────────────

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        import glob as _glob
        profile_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "simulations", simulation_id, "profiles"
        ) if not os.path.isabs(simulation_id) else simulation_id

        if not os.path.isdir(profile_dir):
            profile_dir = os.path.join(Config.SIMULATION_STORAGE_DIR or "", simulation_id, "profiles")
        if not os.path.isdir(profile_dir):
            return []

        profiles = []
        for fpath in sorted(_glob.glob(os.path.join(profile_dir, "*.json"))):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    profiles.extend(data)
                else:
                    profiles.append(data)
            except Exception as e:
                logger.warning(f"Failed to load profile {fpath}: {e}")
        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5
    ) -> tuple:
        system_prompt = f"""You are an interview planner. Based on the interview requirement and the list of available agents, select the most relevant agents to interview.

Return JSON format:
{{
  "selected_indices": [0, 3, 5],
  "reasoning": "Brief explanation of why these agents were selected"
}}

{get_language_instruction()}"""

        profile_summary = "\n".join(
            f"Index {i}: Name={p.get('name', p.get('username', 'Unknown'))}, "
            f"Type={p.get('profession', p.get('entity_type', 'Unknown'))}, "
            f"Bio={p.get('bio', '')[:100]}"
            for i, p in enumerate(profiles[:50])
        )

        user_prompt = f"""Interview Requirement: {interview_requirement}

{f"Simulation Background: {simulation_requirement}" if simulation_requirement else ""}

Available Agents ({min(len(profiles), 50)} shown):
{profile_summary}

Select up to {max_agents} most relevant agents for this interview and explain why."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "")

            selected_agents = [profiles[i] for i in indices if 0 <= i < len(profiles)]
            return selected_agents, indices, reasoning

        except Exception as e:
            logger.warning(f"LLM agent selection failed: {e}, using first {max_agents}")
            indices = list(range(min(max_agents, len(profiles))))
            selected_agents = [profiles[i] for i in indices]
            return selected_agents, indices, "Auto-selected (LLM unavailable)"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str = "",
        selected_agents: List[Dict[str, Any]] = None
    ) -> List[str]:
        system_prompt = f"""You are an interview question designer. Generate insightful interview questions based on the interview requirement and the selected agents' profiles.

Return JSON format:
{{"questions": ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]}}

{get_language_instruction()}"""

        agent_descriptions = ""
        if selected_agents:
            agent_descriptions = "\n".join(
                f"- {a.get('name', a.get('username', 'Unknown'))}: {a.get('profession', '')} - {a.get('bio', '')[:100]}"
                for a in selected_agents
            )

        user_prompt = f"""Interview Requirement: {interview_requirement}

{f"Simulation Background: {simulation_requirement}" if simulation_requirement else ""}

{f"Selected Agents:\n{agent_descriptions}" if agent_descriptions else ""}

Generate 5 insightful interview questions that will produce meaningful responses from the agents."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4
            )

            questions = response.get("questions", [])
            return [str(q) for q in questions[:5]]

        except Exception as e:
            logger.warning(f"LLM question generation failed: {e}, using defaults")
            return [
                f"What is your perspective on {interview_requirement}?",
                f"How does this issue affect you and your community?",
                f"What actions would you recommend regarding this situation?",
                f"What are the potential outcomes you foresee?",
                f"Is there anything else you would like to add?"
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        summarised = "\n\n".join(
            f"### {inv.agent_name} ({inv.agent_role})\n"
            f"Key quotes: {'; '.join(inv.key_quotes[:3])}"
            for inv in interviews
        )
        return f"Interview summary for: {interview_requirement}\n\n{summarised[:2000]}"
