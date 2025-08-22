# graph_db.py

from arango import ArangoClient
from langchain_arangodb import ArangoGraph, ArangoGraphQAChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from schemas import TripletList
from config import ARANGO_HOST, ARANGO_DATABASE, ARANGO_USER, ARANGO_PASSWORD
import streamlit as st
import re

@st.cache_resource
def get_arangodb_graph():
    """Initializes and returns the ArangoDB graph object."""
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(ARANGO_DATABASE, username=ARANGO_USER, password=ARANGO_PASSWORD)
    return ArangoGraph(db)

def sanitize_for_key(text):
    """Sanitizes a string to be a valid ArangoDB _key."""
    sanitized = re.sub(r'[^a-zA-Z0-9_:.@-]', '_', text)
    return sanitized if sanitized else "empty"

def populate_graph_from_docs(graph, docs, llm, source_name: str):
    """
    Extracts entities and relationships and populates the ArangoDB graph.
    """
    if not graph.db.has_graph("knowledge_graph"):
        st.info("Creating new graph 'knowledge_graph'...")
        graph.db.create_graph(
            "knowledge_graph",
            edge_definitions=[{
                "edge_collection": "Relationships",
                "from_vertex_collections": ["Entities"],
                "to_vertex_collections": ["Entities"],
            }]
        )

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
                aql_query = """
                LET head_key = @head_key
                LET tail_key = @tail_key
                
                UPSERT { _key: head_key } INSERT { _key: head_key, id: @head, source: @source } UPDATE { source: @source } IN Entities
                UPSERT { _key: tail_key } INSERT { _key: tail_key, id: @tail, source: @source } UPDATE { source: @source } IN Entities
                UPSERT { _from: @from, _to: @to, label: @label } INSERT { _from: @from, _to: @to, label: @label, source: @source } UPDATE { source: @source } IN Relationships
                """
                
                # --- THIS IS THE FINAL, CORRECTED FUNCTION CALL ---
                # LangChain's wrapper expects parameters as direct keyword arguments.
                graph.query(
                    aql_query,
                    head=triplet.head,
                    tail=triplet.tail,
                    label=triplet.relation,
                    source=source_name,
                    head_key=sanitize_for_key(triplet.head),
                    tail_key=sanitize_for_key(triplet.tail),
                    from_="Entities/" + sanitize_for_key(triplet.head), # Use from_ to avoid Python keyword conflict
                    to="Entities/" + sanitize_for_key(triplet.tail),
                )

            progress_bar.progress((i + 1) / len(docs), text=f"Processing chunk {i+1}/{len(docs)}")
        
        except Exception as e:
            st.error(f"Failed to process chunk {i+1}. Error: {e}")
            continue
            
    st.success(f"Knowledge from '{source_name}' has been added to the graph!")

def get_graph_qa_chain(graph, llm):
    """Creates and returns a question-answering chain for the ArangoDB graph."""
    return ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        graph_name="knowledge_graph",
        verbose=True,
        allow_dangerous_requests=True
    )