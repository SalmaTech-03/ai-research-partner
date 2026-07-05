# graph_db.py

from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from schemas import TripletList
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
import streamlit as st


@st.cache_resource
def get_neo4j_graph():
    """
    Initialize and return the Neo4j graph connection.
    """
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )


def format_relationship(relation: str) -> str:
    """
    Convert relationship names into valid Neo4j relationship types.
    Example:
        "works at" -> WORKS_AT
    """
    return (
        relation.upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def populate_graph_from_docs(graph, docs, llm, source_name: str):
    """
    Extract entities and relationships from documents and populate Neo4j.
    """

    graph.query(
        """
        CREATE INDEX IF NOT EXISTS
        FOR (n:Entity)
        ON (n.id)
        """
    )

    extraction_prompt = PromptTemplate.from_template(
        """
You are an expert knowledge graph extraction system.

Extract entities and relationships from the following text.

Return ONLY valid JSON matching the required schema.

TEXT:
{chunk}
"""
    )

    extraction_chain = extraction_prompt | llm | JsonOutputParser()

    st.info(f"Building knowledge graph from '{source_name}'...")
    progress_bar = st.progress(0)

    total_docs = len(docs)

    for i, doc in enumerate(docs):

        try:

            extracted_json = extraction_chain.invoke(
                {"chunk": doc.page_content}
            )

            validated_triplets = TripletList.parse_obj(extracted_json)

            for triplet in validated_triplets.triplets:

                relation = format_relationship(triplet.relation)

                query = f"""
                MERGE (h:Entity {{id:$head}})
                ON CREATE SET
                    h.source=$source

                MERGE (t:Entity {{id:$tail}})
                ON CREATE SET
                    t.source=$source

                MERGE (h)-[r:`{relation}`]->(t)
                ON CREATE SET
                    r.source=$source
                """

                graph.query(
                    query,
                    params={
                        "head": triplet.head,
                        "tail": triplet.tail,
                        "source": source_name,
                    },
                )

            progress_bar.progress(
                (i + 1) / total_docs,
                text=f"Processing chunk {i+1}/{total_docs}",
            )

        except Exception as e:
            st.error(f"Failed to process chunk {i+1}: {e}")
            continue

    st.success("Knowledge graph created successfully.")


def get_graph_qa_chain(graph, llm):
    """
    Create the Cypher QA chain.
    """

    graph.refresh_schema()

    return GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        verbose=True,
        allow_dangerous_requests=True,
    )
