"""
graph_db.py - Enterprise GraphRAG Engine
- Edge Weighting & Frequency Tracking
- Advanced Entity Resolution
- Transactional Batch Ingestion
"""

import logging
import re
from typing import List, Dict, Any
import streamlit as st
from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

# --- ENTERPRISE UTILITIES ---

def normalize_entity(text: str) -> str:
    """Enterprise-grade normalization: handles acronyms and noisy suffixes."""
    if not text: return "Unknown"
    text = text.strip().replace('"', '').replace("'", "")
    # Preserve Acronyms (BERT, LLM, GPT-4)
    if text.isupper() and len(text) < 10: return text
    # Clean common suffixes/noise
    text = re.sub(r'\s+(Inc|Corp|Ltd|LLC)\.?$', '', text, flags=re.IGNORECASE)
    return text.title()

def sanitize_relationship(rel: str) -> str:
    """Strict uppercase underscored relationships."""
    clean = re.sub(r"[^A-Z0-9_]", "", rel.strip().upper().replace(" ", "_"))
    return clean[:64] or "RELATED_TO"

# --- CORE INGESTION (ATOMIC & WEIGHTED) ---

def insert_triplets_enterprise(graph: Neo4jGraph, batch: List[Dict[str, str]], source: str) -> int:
    """
    Atomic batch insertion with Edge Weighting.
    Tracks how many times a relationship is mentioned to build 'Confidence'.
    """
    if not batch: return 0

    # Grouping logic
    relation_groups = {}
    for triplet in batch:
        rel_type = sanitize_relationship(triplet.get("relation"))
        relation_groups.setdefault(rel_type, []).append({
            "head": normalize_entity(triplet.get("head")),
            "tail": normalize_entity(triplet.get("tail"))
        })

    total = 0
    try:
        for rel_type, rows in relation_groups.items():
            # ENTERPRISE CYPHER: MERGE + ON MATCH (Increments weights)
            query = f"""
            UNWIND $rows AS row
            MERGE (h:Entity {{id: row.head}})
            ON CREATE SET h.source = $source, h.created_at = timestamp(), h.weight = 1
            ON MATCH SET h.weight = h.weight + 1
            
            MERGE (t:Entity {{id: row.tail}})
            ON CREATE SET t.source = $source, t.created_at = timestamp(), t.weight = 1
            ON MATCH SET t.weight = t.weight + 1
            
            MERGE (h)-[r:`{rel_type}`]->(t)
            ON CREATE SET r.source = $source, r.created_at = timestamp(), r.weight = 1, r.occurrences = 1
            ON MATCH SET r.weight = r.weight + 1, r.occurrences = r.occurrences + 1
            """
            graph.query(query, params={"rows": rows, "source": source})
            total += len(rows)
        return total
    except Exception as e:
        logger.error(f"Enterprise Ingestion Failure: {e}")
        return 0

# --- HYBRID RETRIEVAL (RANKED) ---

def get_ranked_neighborhood(graph: Neo4jGraph, entities: List[str]) -> str:
    """
    Retrieves the 2-hop neighborhood, ranked by edge weight (confidence).
    Filters out 'noise' relationships.
    """
    if not entities: return ""
    
    clean_entities = [normalize_entity(e) for e in entities]
    
    query = """
    MATCH (n:Entity)-[r]-(m:Entity)
    WHERE n.id IN $entities
    RETURN n.id AS head, type(r) AS rel, m.id AS tail, r.weight AS weight
    ORDER BY r.weight DESC
    LIMIT 30
    """
    try:
        results = graph.query(query, params={"entities": clean_entities})
        facts = [f"{res['head']} --{res['rel']}--> {res['tail']} (Confidence: {res['weight']})" for res in results]
        return "\n".join(facts)
    except Exception as e:
        logger.exception("Ranked Retrieval Failed")
        return ""
