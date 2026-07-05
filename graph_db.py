import re
import logging
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


# 1. CANONICAL GRAPH CONTRACT

ALLOWED_RELATIONS = {
    "USES", "IMPLEMENTS", "PART_OF", "RELATED_TO",
    "TRAINED_ON", "EVALUATES", "CREATES", "DESCRIBES"
}

# 2. ENTITY RESOLUTION (DETERMINISTIC + SAFE)


def normalize_entity(text: str) -> str:
    """
    Single canonical rule:
    - preserve acronyms
    - preserve versioned models
    - otherwise title-case normalized string
    """
    if not text:
        return "UNKNOWN"

    t = text.strip()

    # Preserve technical tokens
    if t.isupper() or any(c.isdigit() for c in t) or len(t) <= 4:
        return t

    t = re.sub(r"\s+", " ", t)
    return t.title()


# 3. RELATION SAFETY LAYER (ZERO TRUST INPUT)


def sanitize_relation(rel: str) -> str:
    if not rel:
        return "RELATED_TO"

    r = re.sub(r"[^A-Z0-9_]", "", rel.upper().replace(" ", "_"))

    return r if r in ALLOWED_RELATIONS else "RELATED_TO"


# 4. INGESTION ENGINE (ATOMIC + CLEAN + CONSISTENT)


UPSERT_QUERY = """
UNWIND $rows AS row

MERGE (h:Entity {id: row.head})
ON CREATE SET h.created_at = timestamp(), h.count = 1
ON MATCH SET h.count = h.count + 1

MERGE (t:Entity {id: row.tail})
ON CREATE SET t.created_at = timestamp(), t.count = 1
ON MATCH SET t.count = t.count + 1

MERGE (h)-[r:`{rel}`]->(t)
ON CREATE SET r.count = 1, r.weight = row.weight
ON MATCH SET r.count = r.count + 1, r.weight = r.weight + row.weight
"""

def ingest_triplets(
    graph: Neo4jGraph,
    triplets: List[Dict[str, Any]],
    source: str
) -> int:

    if not triplets:
        return 0

    grouped: Dict[str, List[Dict]] = {}

    for t in triplets:
        rel = sanitize_relation(t.get("relation"))

        grouped.setdefault(rel, []).append({
            "head": normalize_entity(t.get("head")),
            "tail": normalize_entity(t.get("tail")),
            "weight": float(t.get("confidence", 1.0))
        })

    total = 0

    for rel, rows in grouped.items():
        query = UPSERT_QUERY.format(rel=rel)
        graph.query(query, params={"rows": rows, "source": source})
        total += len(rows)

    return total

# 5. GRAPH RAG RETRIEVAL (CLEAN MULTI-HOP CORE)


def retrieve_graph_context(
    graph: Neo4jGraph,
    entities: List[str],
    max_hops: int = 2
) -> str:

    if not entities:
        return ""

    clean = [normalize_entity(e) for e in entities]

    query = f"""
    MATCH path = (n:Entity)-[*1..{max_hops}]-(m:Entity)
    WHERE n.id IN $entities
    WITH relationships(path) AS rels
    UNWIND rels AS r
    RETURN DISTINCT
        startNode(r).id + ' -- ' + type(r) + ' --> ' + endNode(r).id AS fact
    LIMIT 40
    """

    try:
        res = graph.query(query, params={"entities": clean})
        return "\n".join(r["fact"] for r in res)
    except Exception as e:
        logger.exception("Graph retrieval failed")
        return ""

