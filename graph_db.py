import logging, json, os, threading, re
from concurrent.futures import ThreadPoolExecutor
from portalocker import Lock
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st
from langchain_neo4j import Neo4jGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from schemas import TripletList

logger = logging.getLogger(__name__)
CACHE_DIR = "storage/cache/triplets"
ALLOWED_RELATIONS = {"USES", "IMPLEMENTS", "PART_OF", "AUTHOR_OF", "TRAINED_ON", "EVALUATES", "DESCRIBES"}
# Token-bucket rate limiting for Gemini (10 RPM)
GEMINI_SEMAPHORE = threading.BoundedSemaphore(value=2)

class EntityResolver:
    def normalize(self, text: str) -> str:
        if not text: return "UNKNOWN"
        t = text.strip()
        if t.isupper() or any(c.isdigit() for c in t) or len(t) <= 4: return t
        return re.sub(r"\s+", " ", t).title()

resolver = EntityResolver()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_query(graph, query, params=None):
    return graph.query(query, params=params)

def initialize_graph(graph: Neo4jGraph):
    safe_query(graph, "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.id, e.ns) IS UNIQUE")

def populate_graph(graph, docs, llm, ns: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    parser = PydanticOutputParser(pydantic_object=TripletList)
    prompt = PromptTemplate.from_template("Extract Knowledge triplets as JSON. Text: {text}\n{format_instructions}")

    def process_chunk(doc):
        from data_processing import get_chunk_hash
        cid = get_chunk_hash(doc.page_content)
        cp = os.path.join(CACHE_DIR, f"{cid}.json")
        
        with Lock(cp + ".lock", timeout=5):
            if os.path.exists(cp):
                with open(cp, "r") as f: return json.load(f)
        
        with GEMINI_SEMAPHORE:
            try:
                res = llm.invoke(prompt.format(text=doc.page_content, format_instructions=parser.get_format_instructions()))
                data = parser.parse(res.content).triplets
                triplets = [t.model_dump() for t in data]
                with open(cp + ".tmp", "w") as f: json.dump(triplets, f)
                os.replace(cp + ".tmp", cp)
                return triplets
            except: return []

    with ThreadPoolExecutor(max_workers=2) as exe:
        results = [t for sub in list(exe.map(process_chunk, docs)) for t in sub]

    # Batch Ingestion by Relation
    for rel in ALLOWED_RELATIONS:
        rows = [{"h": resolver.normalize(t['head']), "t": resolver.normalize(t['tail']), "w": t.get('confidence', 1.0)} 
                for t in results if t.get('relation', '').upper().replace(" ", "_") == rel]
        if rows:
            q = f"UNWIND $rows AS r MERGE (h:Entity {{id: r.h, ns: $ns}}) MERGE (t:Entity {{id: r.t, ns: $ns}}) MERGE (h)-[rel:`{rel}`]->(t) ON CREATE SET rel.weight = r.w"
            safe_query(graph, q, {"rows": rows, "ns": ns})

def retrieve_facts(graph, query_entities: list, ns: str) -> str:
    q = """
    MATCH path = (n:Entity)-[*1..2]-(m:Entity)
    WHERE n.id IN $ents AND n.ns = $ns AND m.ns = $ns
    UNWIND relationships(path) AS r
    RETURN DISTINCT startNode(r).id + ' --' + type(r) + '--> ' + endNode(r).id AS f
    LIMIT 30
    """
    res = safe_query(graph, q, {"ents": [resolver.normalize(e) for e in query_entities], "ns": ns})
    return "\n".join([r['f'] for r in res])
