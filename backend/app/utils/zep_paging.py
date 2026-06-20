"""
Graph pagination read tool (deprecated stub)

Previously wrapped Zep Cloud pagination logic.
Data is now stored in local DuckDB via LocalGraphStore.
This module is kept to avoid breaking old imports.
"""

from __future__ import annotations

from ..utils.logger import get_logger

logger = get_logger('mirofish.zep_paging')


def fetch_all_nodes(client, graph_id: str, **kwargs) -> list:
    """Deprecated: use LocalGraphStore.get_nodes() instead"""
    logger.warning("fetch_all_nodes is deprecated, use LocalGraphStore.get_nodes()")
    return []


def fetch_all_edges(client, graph_id: str, **kwargs) -> list:
    """Deprecated: use LocalGraphStore.get_edges() instead"""
    logger.warning("fetch_all_edges is deprecated, use LocalGraphStore.get_edges()")
    return []
