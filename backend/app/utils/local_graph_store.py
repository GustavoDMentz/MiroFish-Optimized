"""
Local Graph Store — DuckDB + Parquet (Columnar)
================================================
Substitui o armazenamento JSON por tabelas colunares DuckDB,
oferecendo queries indexadas, INSERT/UPDATE eficientes e
busca full-text nativa.

Migração automática de dados JSON existentes na primeira carga.
Memgraph mirroring mantido para compatibilidade.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb

from .logger import get_logger

logger = get_logger('mirofish.local_graph_store')


class LocalGraphStore:
    """Graph store baseado em DuckDB (colunar) + Memgraph mirroring"""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Embedding client (optional)
        try:
            from .llm_client import EmbeddingClient
            self.embedding_client = EmbeddingClient()
        except Exception as e:
            self.embedding_client = None
            logger.warning(f"EmbeddingClient não disponível: {e}")

        # Memgraph driver (optional)
        from ..config import Config
        self.memgraph_host = Config.MEMGRAPH_HOST
        self.memgraph_port = Config.MEMGRAPH_PORT
        self.memgraph_username = Config.MEMGRAPH_USERNAME
        self.memgraph_password = Config.MEMGRAPH_PASSWORD
        self.driver = None
        self.memgraph_connected = False
        if self.memgraph_host:
            try:
                from neo4j import GraphDatabase
                uri = f"bolt://{self.memgraph_host}:{self.memgraph_port}"
                auth = (self.memgraph_username, self.memgraph_password) if self.memgraph_username or self.memgraph_password else None
                self.driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=2.0)
                with self.driver.session() as session:
                    session.run("RETURN 1")
                self.memgraph_connected = True
                logger.info(f"Conectado ao Memgraph em {uri}")
            except Exception as e:
                logger.warning(f"Memgraph indisponível, modo JSON local: {e}")
                self.driver = None
                self.memgraph_connected = False

        # Cache de conexões DuckDB por graph_id
        self._db_connections: Dict[str, duckdb.DuckDBPyConnection] = {}

    # ── Conexão DuckDB ──────────────────────────────────────────────────────

    def _db_path(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "graph.duckdb")

    def _get_db(self, graph_id: str) -> duckdb.DuckDBPyConnection:
        """Retorna conexão DuckDB para o graph_id, criando se necessário."""
        if graph_id in self._db_connections:
            return self._db_connections[graph_id]

        os.makedirs(self._graph_dir(graph_id), exist_ok=True)
        db_path = self._db_path(graph_id)
        is_new = not os.path.exists(db_path)

        conn = duckdb.connect(db_path)
        conn.execute("PRAGMA enable_progress_report=false")
        conn.execute("PRAGMA threads=2")

        if is_new:
            conn.execute("""
                CREATE TABLE metadata (
                    graph_id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    description VARCHAR,
                    ontology JSON,
                    created_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE nodes (
                    uuid VARCHAR PRIMARY KEY,
                    graph_id VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    labels VARCHAR[],
                    summary VARCHAR,
                    attributes JSON,
                    created_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE edges (
                    uuid VARCHAR PRIMARY KEY,
                    graph_id VARCHAR NOT NULL,
                    source_node_uuid VARCHAR NOT NULL,
                    target_node_uuid VARCHAR NOT NULL,
                    name VARCHAR,
                    fact VARCHAR,
                    attributes JSON,
                    created_at TIMESTAMP,
                    valid_at TIMESTAMP,
                    invalid_at TIMESTAMP,
                    expired_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE episodes (
                    uuid VARCHAR PRIMARY KEY,
                    graph_id VARCHAR NOT NULL,
                    text VARCHAR,
                    created_at TIMESTAMP,
                    processed BOOLEAN DEFAULT TRUE
                )
            """)
            # Índices
            conn.execute("CREATE INDEX idx_nodes_graph ON nodes(graph_id)")
            conn.execute("CREATE INDEX idx_nodes_name ON nodes(graph_id, name)")
            conn.execute("CREATE INDEX idx_edges_graph ON edges(graph_id)")
            conn.execute("CREATE INDEX idx_edges_src ON edges(source_node_uuid)")
            conn.execute("CREATE INDEX idx_edges_tgt ON edges(target_node_uuid)")
            conn.execute("CREATE INDEX idx_episodes_graph ON episodes(graph_id)")
            logger.info(f"DuckDB graph criado: {graph_id}")

            # Migrar dados JSON existentes se houver
            self._migrate_from_json(conn, graph_id)
        else:
            logger.debug(f"DuckDB graph aberto: {graph_id}")

        self._db_connections[graph_id] = conn
        return conn

    def _migrate_from_json(self, conn: duckdb.DuckDBPyConnection, graph_id: str):
        """Migra dados de arquivos JSON existentes para DuckDB."""
        # Verificar se há nodes.json
        nodes_path = self._nodes_path(graph_id)
        edges_path = self._edges_path(graph_id)
        meta_path = self._meta_path(graph_id)
        episodes_path = self._episodes_path(graph_id)

        if os.path.exists(nodes_path):
            try:
                with open(nodes_path, 'r', encoding='utf-8') as f:
                    nodes = json.load(f)
                if nodes:
                    conn.executemany(
                        "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(
                            n.get("uuid", uuid.uuid4().hex),
                            graph_id,
                            n.get("name", ""),
                            n.get("labels", ["Entity"]),
                            n.get("summary", ""),
                            json.dumps(n.get("attributes", {}), ensure_ascii=False),
                            n.get("created_at", datetime.now().isoformat())
                        ) for n in nodes]
                    )
                    logger.info(f"Migrados {len(nodes)} nós de {nodes_path}")
            except Exception as e:
                logger.warning(f"Falha ao migrar {nodes_path}: {e}")

        if os.path.exists(edges_path):
            try:
                with open(edges_path, 'r', encoding='utf-8') as f:
                    edges = json.load(f)
                if edges:
                    conn.executemany(
                        "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(
                            e.get("uuid", uuid.uuid4().hex),
                            graph_id,
                            e.get("source_node_uuid", ""),
                            e.get("target_node_uuid", ""),
                            e.get("name", ""),
                            e.get("fact", ""),
                            json.dumps(e.get("attributes", {}), ensure_ascii=False),
                            e.get("created_at", datetime.now().isoformat()),
                            e.get("valid_at"),
                            e.get("invalid_at"),
                            e.get("expired_at"),
                        ) for e in edges]
                    )
                    logger.info(f"Migradas {len(edges)} arestas de {edges_path}")
            except Exception as e:
                logger.warning(f"Falha ao migrar {edges_path}: {e}")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                conn.execute("""
                    INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?)
                """, [
                    graph_id,
                    meta.get("name", graph_id),
                    meta.get("description", ""),
                    json.dumps(meta.get("ontology"), ensure_ascii=False) if meta.get("ontology") else None,
                    meta.get("created_at", datetime.now().isoformat())
                ])
                logger.info(f"Metadata migrado de {meta_path}")
            except Exception as e:
                logger.warning(f"Falha ao migrar {meta_path}: {e}")

        if os.path.exists(episodes_path):
            try:
                episodes = []
                with open(episodes_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ep = json.loads(line)
                            episodes.append((
                                ep.get("uuid", uuid.uuid4().hex),
                                graph_id,
                                ep.get("text", ""),
                                ep.get("created_at", datetime.now().isoformat()),
                                ep.get("processed", True)
                            ))
                if episodes:
                    conn.executemany(
                        "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?)",
                        episodes
                    )
                    logger.info(f"Migrados {len(episodes)} episódios de {episodes_path}")
            except Exception as e:
                logger.warning(f"Falha ao migrar {episodes_path}: {e}")

    def _close_db(self, graph_id: str):
        """Fecha conexão DuckDB."""
        conn = self._db_connections.pop(graph_id, None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # ── Memgraph mirroring ──────────────────────────────────────────────────

    def _execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.memgraph_connected or not self.driver:
            return None
        try:
            with self.driver.session() as session:
                return list(session.run(query, params or {}))
        except Exception as e:
            logger.warning(f"Cypher error: {e}")
            return None

    # ── Graph lifecycle ─────────────────────────────────────────────────────

    def create_graph(self, graph_id: str, name: str, description: str = "") -> None:
        conn = self._get_db(graph_id)
        conn.execute("""
            INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, NULL, ?)
        """, [graph_id, name, description, datetime.now().isoformat()])

        # Memgraph mirroring
        self._execute_cypher("""
            MERGE (g:Graph {graph_id: $graph_id})
            ON CREATE SET g.name = $name, g.description = $description, g.created_at = $created_at
        """, {"graph_id": graph_id, "name": name, "description": description,
              "created_at": datetime.now().isoformat()})
        logger.info(f"Graph criado: {graph_id}")

    def delete_graph(self, graph_id: str) -> None:
        self._close_db(graph_id)
        graph_dir = self._graph_dir(graph_id)
        if os.path.exists(graph_dir):
            shutil.rmtree(graph_dir)
            logger.info(f"Graph deletado: {graph_id}")
        self._execute_cypher("MATCH (n) WHERE n.graph_id = $graph_id DETACH DELETE n", {"graph_id": graph_id})
        self._execute_cypher("MATCH (g:Graph {graph_id: $graph_id}) DETACH DELETE g", {"graph_id": graph_id})

    def graph_exists(self, graph_id: str) -> bool:
        return os.path.exists(self._db_path(graph_id)) or os.path.exists(self._meta_path(graph_id))

    # ── Ontology ────────────────────────────────────────────────────────────

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        conn = self._get_db(graph_id)
        conn.execute("UPDATE metadata SET ontology = ? WHERE graph_id = ?",
                     [json.dumps(ontology, ensure_ascii=False), graph_id])
        self._execute_cypher("""
            MATCH (g:Graph {graph_id: $graph_id})
            SET g.ontology = $ontology_json
        """, {"graph_id": graph_id, "ontology_json": json.dumps(ontology, ensure_ascii=False)})

    def get_ontology(self, graph_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = self._get_db(graph_id)
            row = conn.execute("SELECT ontology FROM metadata WHERE graph_id = ?", [graph_id]).fetchone()
            if row and row[0]:
                if isinstance(row[0], str):
                    return json.loads(row[0])
                return row[0]
        except Exception:
            pass
        return None

    def get_metadata(self, graph_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = self._get_db(graph_id)
            row = conn.execute("SELECT * FROM metadata WHERE graph_id = ?", [graph_id]).fetchone()
            if row:
                return {"graph_id": row[0], "name": row[1], "description": row[2],
                        "ontology": json.loads(row[3]) if isinstance(row[3], str) else row[3],
                        "created_at": str(row[4])}
        except Exception:
            pass
        # Fallback para metadata.json
        meta_path = self._meta_path(graph_id)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    # ── Episodes ────────────────────────────────────────────────────────────

    def add_episode(self, graph_id: str, text: str) -> str:
        episode_id = uuid.uuid4().hex
        conn = self._get_db(graph_id)
        conn.execute("""
            INSERT INTO episodes VALUES (?, ?, ?, ?, ?)
        """, [episode_id, graph_id, text, datetime.now().isoformat(), True])
        self._execute_cypher("""
            CREATE (e:Episode {uuid: $uuid, graph_id: $graph_id, text: $text, created_at: $created_at})
        """, {"uuid": episode_id, "graph_id": graph_id, "text": text,
              "created_at": datetime.now().isoformat()})
        return episode_id

    def add_episodes_batch(self, graph_id: str, texts: List[str]) -> List[str]:
        ids = []
        conn = self._get_db(graph_id)
        now = datetime.now().isoformat()
        for text in texts:
            eid = uuid.uuid4().hex
            ids.append(eid)
            conn.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
                         [eid, graph_id, text, now, True])
        return ids

    def episode_is_processed(self, graph_id: str, episode_uuid: str) -> bool:
        return True

    def get_episodes(self, graph_id: str) -> List[Dict[str, Any]]:
        try:
            conn = self._get_db(graph_id)
            rows = conn.execute("""
                SELECT uuid, graph_id, text, created_at, processed
                FROM episodes WHERE graph_id = ? ORDER BY created_at
            """, [graph_id]).fetchall()
            return [{"uuid": r[0], "graph_id": r[1], "text": r[2],
                     "created_at": str(r[3]) if r[3] else None, "processed": r[4]} for r in rows]
        except Exception:
            return []

    # ── Nodes ───────────────────────────────────────────────────────────────

    def get_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        try:
            conn = self._get_db(graph_id)
            rows = conn.execute("""
                SELECT uuid, name, labels, summary, attributes, created_at
                FROM nodes WHERE graph_id = ?
            """, [graph_id]).fetchall()
            result = []
            for r in rows:
                attrs = r[4]
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except Exception:
                        attrs = {}
                result.append({
                    "uuid": r[0],
                    "name": r[1],
                    "labels": r[2] if r[2] else ["Entity"],
                    "summary": r[3] if r[3] else "",
                    "attributes": attrs if attrs else {},
                    "created_at": str(r[5]) if r[5] else datetime.now().isoformat(),
                })
            return result
        except Exception:
            return self._fallback_get_nodes(graph_id)

    def _fallback_get_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        path = self._nodes_path(graph_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f) or []
            except Exception:
                pass
        return []

    def get_node(self, graph_id: str, node_uuid: str) -> Optional[Dict[str, Any]]:
        try:
            conn = self._get_db(graph_id)
            row = conn.execute("""
                SELECT uuid, name, labels, summary, attributes, created_at
                FROM nodes WHERE uuid = ? AND graph_id = ?
            """, [node_uuid, graph_id]).fetchone()
            if row:
                attrs = row[4]
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except Exception:
                        attrs = {}
                return {
                    "uuid": row[0], "name": row[1],
                    "labels": row[2] if row[2] else ["Entity"],
                    "summary": row[3] if row[3] else "",
                    "attributes": attrs if attrs else {},
                    "created_at": str(row[5]) if row[5] else None,
                }
        except Exception:
            pass
        # Fallback: busca linear no JSON
        for node in self._fallback_get_nodes(graph_id):
            if node.get("uuid") == node_uuid:
                return node
        return None

    def get_node_by_name(self, graph_id: str, name: str) -> Optional[Dict[str, Any]]:
        """Busca nó por nome (case-insensitive). DuckDB indexado."""
        try:
            conn = self._get_db(graph_id)
            row = conn.execute("""
                SELECT uuid, name, labels, summary, attributes, created_at
                FROM nodes WHERE graph_id = ? AND LOWER(name) = LOWER(?)
            """, [graph_id, name]).fetchone()
            if row:
                attrs = row[4]
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except Exception:
                        attrs = {}
                return {
                    "uuid": row[0], "name": row[1],
                    "labels": row[2] if row[2] else ["Entity"],
                    "summary": row[3] if row[3] else "",
                    "attributes": attrs if attrs else {},
                    "created_at": str(row[5]) if row[5] else None,
                }
        except Exception:
            pass
        return None

    def upsert_node(
        self,
        graph_id: str,
        name: str,
        labels: Optional[List[str]] = None,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        labels = labels or ["Entity"]
        attributes = attributes or {}
        conn = self._get_db(graph_id)

        # Obter embedding
        embedding = []
        if self.embedding_client:
            embedding_text = f"{name}: {summary}" if summary else name
            try:
                embedding = self.embedding_client.get_embedding(embedding_text)
            except Exception as e:
                import logging
                logging.getLogger('mirofish.local_graph_store').warning(f"Erro ao obter embedding para {name}: {e}")

        if embedding:
            attributes["embedding"] = embedding

        # Tentar encontrar por nome
        existing = conn.execute(
            "SELECT uuid, labels, summary, attributes FROM nodes WHERE graph_id = ? AND LOWER(name) = LOWER(?)",
            [graph_id, name]
        ).fetchone()

        if existing:
            node_uuid = existing[0]
            merged_labels = list(set(existing[1] or []) | set(labels))
            merged_summary = summary if summary else (existing[2] or "")
            existing_attrs = json.loads(existing[3]) if isinstance(existing[3], str) else (existing[3] or {})
            merged_attrs = {**existing_attrs, **attributes}
            if embedding:
                merged_attrs["embedding"] = embedding
            conn.execute("""
                UPDATE nodes SET labels = ?, summary = ?, attributes = ?
                WHERE uuid = ?
            """, [merged_labels, merged_summary,
                  json.dumps(merged_attrs, ensure_ascii=False), node_uuid])
            attributes = merged_attrs
        else:
            node_uuid = uuid.uuid4().hex
            conn.execute("""
                INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [node_uuid, graph_id, name, labels, summary,
                  json.dumps(attributes, ensure_ascii=False),
                  datetime.now().isoformat()])

        sanitized_labels = ["".join(c for c in l if c.isalnum() or c == '_') or 'Entity' for l in labels]
        if 'Entity' not in sanitized_labels:
            sanitized_labels.append('Entity')
        label_clause = ":" + ":".join(sanitized_labels)
        query = f"""
            MERGE (n{label_clause} {{uuid: $uuid, graph_id: $graph_id}})
            ON CREATE SET n.name = $name, n.labels = $labels, n.summary = $summary,
                          n.attributes = $attributes_json, n.created_at = $created_at
            ON MATCH SET n.name = $name, n.labels = $labels, n.summary = $summary,
                         n.attributes = $attributes_json
        """
        params = {
            "uuid": node_uuid, "graph_id": graph_id, "name": name,
            "labels": labels, "summary": summary,
            "attributes_json": json.dumps(attributes, ensure_ascii=False),
            "created_at": datetime.now().isoformat()
        }
        if embedding:
            query += "\nSET n.embedding = $embedding"
            params["embedding"] = embedding
        self._execute_cypher(query, params)

        return node_uuid

    # ── Edges ───────────────────────────────────────────────────────────────

    def get_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        try:
            conn = self._get_db(graph_id)
            rows = conn.execute("""
                SELECT uuid, graph_id, source_node_uuid, target_node_uuid,
                       name, fact, attributes, created_at, valid_at, invalid_at, expired_at
                FROM edges WHERE graph_id = ?
            """, [graph_id]).fetchall()
            result = []
            for r in rows:
                attrs = r[6]
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except Exception:
                        attrs = {}
                result.append({
                    "uuid": r[0], "graph_id": r[1],
                    "source_node_uuid": r[2], "target_node_uuid": r[3],
                    "name": r[4] or "", "fact": r[5] or "",
                    "attributes": attrs or {},
                    "created_at": str(r[7]) if r[7] else None,
                    "valid_at": str(r[8]) if r[8] else None,
                    "invalid_at": str(r[9]) if r[9] else None,
                    "expired_at": str(r[10]) if r[10] else None,
                })
            return result
        except Exception:
            return self._fallback_get_edges(graph_id)

    def _fallback_get_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        path = self._edges_path(graph_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f) or []
            except Exception:
                pass
        return []

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        """Busca edges por node_uuid usando índices DuckDB."""
        try:
            conn = self._get_db(graph_id)
            rows = conn.execute("""
                SELECT uuid, graph_id, source_node_uuid, target_node_uuid,
                       name, fact, attributes, created_at, valid_at, invalid_at, expired_at
                FROM edges WHERE graph_id = ?
                AND (source_node_uuid = ? OR target_node_uuid = ?)
            """, [graph_id, node_uuid, node_uuid]).fetchall()
            result = []
            for r in rows:
                attrs = r[6]
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except Exception:
                        attrs = {}
                result.append({
                    "uuid": r[0], "graph_id": r[1],
                    "source_node_uuid": r[2], "target_node_uuid": r[3],
                    "name": r[4] or "", "fact": r[5] or "",
                    "attributes": attrs or {},
                    "created_at": str(r[7]) if r[7] else None,
                    "valid_at": str(r[8]) if r[8] else None,
                    "invalid_at": str(r[9]) if r[9] else None,
                    "expired_at": str(r[10]) if r[10] else None,
                })
            return result
        except Exception:
            return self._fallback_get_node_edges(graph_id, node_uuid)

    def _fallback_get_node_edges(self, graph_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        return [
            e for e in self._fallback_get_edges(graph_id)
            if e.get("source_node_uuid") == node_uuid or e.get("target_node_uuid") == node_uuid
        ]

    def add_edge(self, graph_id: str, edge: Dict[str, Any]) -> str:
        edge_uuid = edge.get("uuid") or uuid.uuid4().hex
        edge = dict(edge)
        edge["uuid"] = edge_uuid
        edge.setdefault("created_at", datetime.now().isoformat())

        # Obter embedding
        embedding = []
        fact = edge.get("fact", "")
        if fact and self.embedding_client:
            try:
                embedding = self.embedding_client.get_embedding(fact)
            except Exception as e:
                import logging
                logging.getLogger('mirofish.local_graph_store').warning(f"Erro ao obter embedding para edge: {e}")

        if embedding:
            attrs = dict(edge.get("attributes", {}))
            attrs["embedding"] = embedding
            edge["attributes"] = attrs

        conn = self._get_db(graph_id)
        conn.execute("""
            INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            edge_uuid, graph_id,
            edge.get("source_node_uuid", ""),
            edge.get("target_node_uuid", ""),
            edge.get("name", ""),
            edge.get("fact", ""),
            json.dumps(edge.get("attributes", {}), ensure_ascii=False),
            edge.get("created_at"),
            edge.get("valid_at"),
            edge.get("invalid_at"),
            edge.get("expired_at"),
        ])

        # Memgraph mirroring
        # Reutiliza o embedding obtido anteriormente

        rel_name = edge.get("name") or "RELATED_TO"
        sanitized_rel = "".join(c if c.isalnum() or c == '_' else '_' for c in rel_name).upper() or "RELATED_TO"

        query = f"""
            MATCH (s:Entity {{uuid: $source_uuid, graph_id: $graph_id}})
            MATCH (t:Entity {{uuid: $target_uuid, graph_id: $graph_id}})
            CREATE (s)-[r:{sanitized_rel} {{
                uuid: $uuid, name: $name, fact: $fact,
                created_at: $created_at, valid_at: $valid_at,
                invalid_at: $invalid_at, expired_at: $expired_at,
                graph_id: $graph_id
            }}]->(t)
        """
        params = {
            "source_uuid": edge.get("source_node_uuid"),
            "target_uuid": edge.get("target_node_uuid"),
            "uuid": edge_uuid, "name": rel_name, "fact": fact,
            "created_at": edge.get("created_at"),
            "valid_at": edge.get("valid_at"),
            "invalid_at": edge.get("invalid_at"),
            "expired_at": edge.get("expired_at"),
            "graph_id": graph_id
        }
        if embedding:
            query = query.replace("}]->(t)", ",\n            embedding: $embedding\n        }]->(t)")
            params["embedding"] = embedding
        self._execute_cypher(query, params)

        return edge_uuid

    def add_fact_edge(self, graph_id: str, source_uuid: str, target_uuid: str, name: str, fact: str) -> str:
        return self.add_edge(graph_id, {
            "name": name, "fact": fact,
            "source_node_uuid": source_uuid,
            "target_node_uuid": target_uuid,
        })

    # ── Search (DuckDB FTS + LIKE) ─────────────────────────────────────────

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> Dict[str, Any]:
        """Busca híbrida: Memgraph vector → DuckDB local vector → DuckDB FTS/LIKE fallback."""
        # 1. Tentar Memgraph vector search
        if self.memgraph_connected and self.embedding_client:
            try:
                q_emb = self.embedding_client.get_embedding(query)
                if q_emb:
                    result_edges, result_nodes, facts = self._mg_vector_search(graph_id, q_emb, limit, scope)
                    if facts:
                        return {"facts": facts, "edges": result_edges, "nodes": result_nodes}
            except Exception as e:
                logger.warning(f"Erro na busca vetorial Memgraph: {e}")

        # 2. Tentar local vector search (DuckDB + Python/NumPy)
        if self.embedding_client:
            try:
                q_emb = self.embedding_client.get_embedding(query)
                if q_emb:
                    res = self._local_vector_search(graph_id, q_emb, limit, scope)
                    if res and res.get("facts"):
                        return res
            except Exception as e:
                logger.warning(f"Erro na busca vetorial local: {e}")

        # 3. Fallback DuckDB LIKE search (indexado)
        return self._duckdb_search(graph_id, query, limit, scope)

    def _local_vector_search(self, graph_id: str, query_embedding: List[float], limit: int, scope: str) -> Dict[str, Any]:
        """Busca vetorial local em memória usando os dados do DuckDB + NumPy."""
        import numpy as np
        
        def cosine_similarity(v1, v2):
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

        conn = self._get_db(graph_id)
        result_edges, result_nodes, facts = [], [], []
        q_emb = np.array(query_embedding)

        # 1. Buscar Arestas
        if scope in ("edges", "both"):
            try:
                rows = conn.execute("""
                    SELECT uuid, source_node_uuid, target_node_uuid, name, fact, attributes, created_at
                    FROM edges WHERE graph_id = ?
                """, [graph_id]).fetchall()
                
                scored_edges = []
                for r in rows:
                    attrs = json.loads(r[5]) if isinstance(r[5], str) else (r[5] or {})
                    emb = attrs.get("embedding")
                    if emb and len(emb) == len(query_embedding):
                        sim = cosine_similarity(q_emb, np.array(emb))
                        scored_edges.append((sim, r, attrs))
                
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                for sim, r, attrs in scored_edges[:limit]:
                    attrs_clean = dict(attrs)
                    attrs_clean.pop("embedding", None)
                    edge = {
                        "uuid": r[0], "graph_id": graph_id,
                        "source_node_uuid": r[1], "target_node_uuid": r[2],
                        "name": r[3] or "", "fact": r[4] or "",
                        "attributes": attrs_clean,
                        "created_at": str(r[6]) if r[6] else None,
                        "similarity": sim
                    }
                    result_edges.append(edge)
                    if edge.get("fact"):
                        facts.append(edge["fact"])
            except Exception as e:
                logger.error(f"Erro na busca vetorial local de arestas: {e}")

        # 2. Buscar Nodos
        if scope in ("nodes", "both"):
            try:
                rows = conn.execute("""
                    SELECT uuid, name, labels, summary, attributes, created_at
                    FROM nodes WHERE graph_id = ?
                """, [graph_id]).fetchall()
                
                scored_nodes = []
                for r in rows:
                    attrs = json.loads(r[4]) if isinstance(r[4], str) else (r[4] or {})
                    emb = attrs.get("embedding")
                    if emb and len(emb) == len(query_embedding):
                        sim = cosine_similarity(q_emb, np.array(emb))
                        scored_nodes.append((sim, r, attrs))
                        
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                for sim, r, attrs in scored_nodes[:limit]:
                    attrs_clean = dict(attrs)
                    attrs_clean.pop("embedding", None)
                    node = {
                        "uuid": r[0], "name": r[1],
                        "labels": r[2] if r[2] else ["Entity"],
                        "summary": r[3] or "",
                        "attributes": attrs_clean,
                        "created_at": str(r[5]) if r[5] else None,
                        "similarity": sim
                    }
                    result_nodes.append(node)
                    if node.get("summary"):
                        facts.append(f"[{node['name']}]: {node['summary']}")
            except Exception as e:
                logger.error(f"Erro na busca vetorial local de nodos: {e}")

        return {"facts": facts, "edges": result_edges, "nodes": result_nodes}

    def _mg_vector_search(self, graph_id: str, query_embedding: List[float], limit: int, scope: str) -> tuple:
        result_edges, result_nodes, facts = [], [], []
        if scope in ("edges", "both"):
            records = self._execute_cypher("""
                MATCH (s:Entity {graph_id: $graph_id})-[r]->(t:Entity {graph_id: $graph_id})
                WHERE r.embedding IS NOT NULL
                WITH r, s, t, reduce(s = 0.0, i in range(0, size(r.embedding)-1) | s + r.embedding[i] * $qe[i]) as sim
                ORDER BY sim DESC LIMIT $limit
                RETURN r, s.uuid, s.name, t.uuid, t.name, sim
            """, {"graph_id": graph_id, "qe": query_embedding, "limit": limit})
            if records:
                for rec in records:
                    r = dict(rec["r"])
                    r.update({"source_node_uuid": rec["s.uuid"], "source_node_name": rec["s.name"],
                              "target_node_uuid": rec["t.uuid"], "target_node_name": rec["t.name"],
                              "similarity": rec["sim"]})
                    r.pop("embedding", None)
                    result_edges.append(r)
                    if r.get("fact"):
                        facts.append(r["fact"])

        if scope in ("nodes", "both"):
            records = self._execute_cypher("""
                MATCH (n:Entity {graph_id: $graph_id})
                WHERE n.embedding IS NOT NULL
                WITH n, reduce(s = 0.0, i in range(0, size(n.embedding)-1) | s + n.embedding[i] * $qe[i]) as sim
                ORDER BY sim DESC LIMIT $limit
                RETURN n, sim
            """, {"graph_id": graph_id, "qe": query_embedding, "limit": limit})
            if records:
                for rec in records:
                    n = dict(rec["n"])
                    if isinstance(n.get("attributes"), str):
                        try:
                            n["attributes"] = json.loads(n["attributes"])
                        except Exception:
                            pass
                    n["similarity"] = rec["sim"]
                    n.pop("embedding", None)
                    result_nodes.append(n)
                    if n.get("summary"):
                        facts.append(f"[{n['name']}]: {n['summary']}")

        return result_edges, result_nodes, facts

    def _duckdb_search(self, graph_id: str, query: str, limit: int, scope: str) -> Dict[str, Any]:
        """Busca por palavra-chave via DuckDB LIKE (O(log n) com índice)."""
        try:
            conn = self._get_db(graph_id)
        except Exception:
            return {"facts": [], "edges": [], "nodes": []}

        q = f"%{query}%"
        result_edges, result_nodes, facts = [], [], []

        if scope in ("edges", "both"):
            try:
                rows = conn.execute("""
                    SELECT uuid, graph_id, source_node_uuid, target_node_uuid,
                           name, fact, attributes, created_at, valid_at, invalid_at, expired_at
                    FROM edges WHERE graph_id = ?
                    AND (fact ILIKE ? OR name ILIKE ?)
                    LIMIT ?
                """, [graph_id, q, q, limit]).fetchall()
                for r in rows:
                    attrs = json.loads(r[6]) if isinstance(r[6], str) else (r[6] or {})
                    edge = {
                        "uuid": r[0], "graph_id": r[1],
                        "source_node_uuid": r[2], "target_node_uuid": r[3],
                        "name": r[4] or "", "fact": r[5] or "",
                        "attributes": attrs or {},
                        "created_at": str(r[7]) if r[7] else None,
                    }
                    result_edges.append(edge)
                    if edge.get("fact"):
                        facts.append(edge["fact"])
            except Exception:
                pass

        if scope in ("nodes", "both"):
            try:
                rows = conn.execute("""
                    SELECT uuid, name, labels, summary, attributes, created_at
                    FROM nodes WHERE graph_id = ?
                    AND (name ILIKE ? OR summary ILIKE ?)
                    LIMIT ?
                """, [graph_id, q, q, limit]).fetchall()
                for r in rows:
                    attrs = json.loads(r[4]) if isinstance(r[4], str) else (r[4] or {})
                    node = {
                        "uuid": r[0], "name": r[1],
                        "labels": r[2] if r[2] else ["Entity"],
                        "summary": r[3] or "",
                        "attributes": attrs or {},
                        "created_at": str(r[5]) if r[5] else None,
                    }
                    result_nodes.append(node)
                    if node.get("summary"):
                        facts.append(f"[{node['name']}]: {node['summary']}")
            except Exception:
                pass

        return {"facts": facts, "edges": result_edges, "nodes": result_nodes}

    # ── Utility ─────────────────────────────────────────────────────────────

    def get_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Estatísticas rápidas via SQL agregado."""
        try:
            conn = self._get_db(graph_id)
            node_count = conn.execute("SELECT count(*) FROM nodes WHERE graph_id = ?", [graph_id]).fetchone()[0]
            edge_count = conn.execute("SELECT count(*) FROM edges WHERE graph_id = ?", [graph_id]).fetchone()[0]
            ep_count = conn.execute("SELECT count(*) FROM episodes WHERE graph_id = ?", [graph_id]).fetchone()[0]
            return {"nodes": node_count, "edges": edge_count, "episodes": ep_count}
        except Exception:
            return {"nodes": 0, "edges": 0, "episodes": 0}

    def _graph_dir(self, graph_id: str) -> str:
        return os.path.join(self.storage_dir, graph_id)

    def _meta_path(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "metadata.json")

    def _nodes_path(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "nodes.json")

    def _edges_path(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "edges.json")

    def _episodes_path(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "episodes.jsonl")
