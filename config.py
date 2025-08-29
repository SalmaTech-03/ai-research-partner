# config.py (Final Neo4j Version)

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- LLM and EMBEDDING MODELS ---
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- NEW: NEO4J DATABASE ---
NEO4J_URI = st.secrets.get("NEO4J_URI")
NEO4J_USERNAME = st.secrets.get("NEO4J_USERNAME")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD")

# --- GLOBAL INITIALIZATIONS ---
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

#### B. `graph_db.py` (The Biggest Change)

# graph_db.py (Final Neo4j Version)

from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from schemas import TripletList
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
import streamlit as st

@st.cache_resource
def get_neo4j_graph():
    """Initializes and returns the Neo4j graph object."""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

def format_cypher_relationship(rel: str) -> str:
    """Sanitizes a string for use as a Cypher relationship type."""
    # Replace spaces and hyphens with underscores, convert to uppercase
    return rel.replace(' ', '_').replace('-', '_').upper()

def populate_graph_from_docs(graph, docs, llm, source_name: str):
    """
    Extracts entities and relationships and populates the Neo4j graph using MERGE.
    """
    # Create an index for faster lookups on the 'id' property of nodes
    graph.query("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)")

    extraction_prompt = PromptTemplate.from_template(
        """
        You are an expert data analyst... (Prompt is the same as before)
        TEXT:
        {chunk}
        """
    )
    
    extraction_chain = extraction_prompt | llm | JsonOutputParser()
    
    st.write(f"Extracting knowledge from '{source_name}' and populating graph...")
    progress_bar = st.progress(0)
    
    for i, doc in enumerate(docs):
        try:
            extracted_json = extraction_chain.invoke({"chunk": doc.page_content})
            validated_triplets = TripletList.parse_obj(extracted_json)
            
            for triplet in validated_triplets.triplets:
                # Use MERGE to create nodes and relationships without duplicates
                # MERGE is Neo4j's equivalent of UPSERT
                cypher_query = """
                MERGE (h:Entity {id: $head})
                ON CREATE SET h.source = $source
                MERGE (t:Entity {id: $tail})
                ON CREATE SET t.source = $source
                MERGE (h)-[r:`{relation}`]->(t)
                ON CREATE SET r.source = $source
                """
                
                # Format the relationship type dynamically
                formatted_query = cypher_query.format(relation=format_cypher_relationship(triplet.relation))
                
                graph.query(
                    formatted_query,
                    params={
                        "head": triplet.head,
                        "tail": triplet.tail,
                        "source": source_name,
                    }
                )

            progress_bar.progress((i + 1) / len(docs), text=f"Processing chunk {i+1}/{len(docs)}")
        
        except Exception as e:
            st.error(f"Failed to process chunk {i+1}. Error: {e}")
            continue
            
    st.success(f"Knowledge from '{source_name}' has been added to the graph!")

def get_graph_qa_chain(graph, llm):
    """Creates and returns a question-answering chain for the Neo4j graph."""
    graph.refresh_schema() # Important to update schema for the QA chain
    return GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        verbose=True,
        allow_dangerous_requests=True
    )