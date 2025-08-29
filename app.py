# app.py

import streamlit as st
from config import get_llm, get_embeddings_model
from data_processing import process_uploaded_pdf, process_url
from vector_store import create_faiss_vector_store
# Import the correct Neo4j functions from graph_db
from graph_db import get_neo4j_graph, populate_graph_from_docs, get_graph_qa_chain
from qa_chain import generate_response
from visualization import visualize_graph_from_query
from google.api_core.exceptions import ResourceExhausted

st.set_page_config(page_title="🧠 AI Research Partner", layout="wide")
st.title("🧠 AI Research Partner")

# --- Initialize Models and Connections ---
llm = get_llm()
embeddings = get_embeddings_model()
# Initialize the Neo4j graph connection
graph = get_neo4j_graph()

# --- Session State Management ---
if "docs" not in st.session_state:
    st.session_state.docs = []
if "vector_retriever" not in st.session_state:
    st.session_state.vector_retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = set()

# --- UI: Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("1. Add Data Sources")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    url_input = st.text_input("Or enter a website URL")

    if st.button("Process Sources"):
        if not uploaded_files and not url_input:
            st.warning("Please upload a PDF or enter a URL.")
        else:
            new_docs = []
            with st.spinner("Processing documents... This may take a few minutes."):
                try:
                    # Clear the entire Neo4j database before processing new files
                    st.info("Clearing old graph data...")
                    graph.query("MATCH (n) DETACH DELETE n")

                    if uploaded_files:
                        for file in uploaded_files:
                            if file.name not in st.session_state.processed_sources:
                                st.info(f"Processing PDF: {file.name}")
                                pdf_docs = process_uploaded_pdf(file)
                                populate_graph_from_docs(graph, pdf_docs, llm, file.name)
                                new_docs.extend(pdf_docs)
                                st.session_state.processed_sources.add(file.name)
                    
                    if url_input and url_input not in st.session_state.processed_sources:
                        st.info(f"Processing URL: {url_input}")
                        url_docs = process_url(url_input)
                        populate_graph_from_docs(graph, url_docs, llm, url_input)
                        new_docs.extend(url_docs)
                        st.session_state.processed_sources.add(url_input)

                except ResourceExhausted:
                    st.error("API Quota Reached during processing. Please try again tomorrow.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

            if new_docs:
                st.session_state.docs.extend(new_docs)
                st.info("Creating/Updating vector store...")
                st.session_state.vector_retriever = create_faiss_vector_store(st.session_state.docs, embeddings)
                st.success("All new documents processed successfully!")
            else:
                st.info("No new documents to process.")

    st.header("Processed Sources")
    st.markdown(f"**{len(st.session_state.processed_sources)}** sources loaded.")
    with st.expander("View Sources"):
        for source in st.session_state.processed_sources:
            st.write(f"- {source}")

# --- Main UI: Chat Interface ---
st.header("2. Ask Your Research Question")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            tab_list = [
                "✅ Main Answer", " perspectives", "🔬 Analytical Insights", 
                "💡 Creative Insights", "🔎 Recommendations", "📚 Sources & Details"
            ]
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)
            with tab1: st.markdown(message["content"]["answer"])
            with tab2: st.markdown(message["content"]["perspectives"])
            with tab3: st.markdown(message["content"]["analytical_insights"])
            with tab4: st.markdown(message["content"]["creative_insights"])
            with tab5: st.markdown(message["content"]["recommendations"])
            with tab6:
                st.text_area("Semantic Context", message["content"]["semantic_sources"], height=200, key=f"semantic_{i}")
        else:
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.vector_retriever:
        st.warning("Please process at least one document before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Performing multi-stage analysis and recommendations..."):
                try:
                    graph_qa_chain = get_graph_qa_chain(graph, llm)
                    result = generate_response(
                        prompt, st.session_state.vector_retriever, graph_qa_chain, llm
                    )
                    
                    tab_list = [
                        "✅ Main Answer", " perspectives", "🔬 Analytical Insights", 
                        "💡 Creative Insights", "🔎 Recommendations", "📚 Sources & Details"
                    ]
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)

                    with tab1:
                        st.markdown(result["answer"])
                        st.download_button("Download Answer", result["answer"], "answer.md", "text/markdown")
                    with tab2:
                        st.markdown(result["perspectives"])
                    with tab3:
                        st.markdown(result["analytical_insights"])
                    with tab4:
                        st.markdown(result["creative_insights"])
                    with tab5:
                        st.markdown(result["recommendations"])
                    with tab6:
                        st.subheader("Document Sources (from Vector Search)")
                        st.text_area("Semantic Context", result["semantic_sources"], height=200)
                        st.subheader("Knowledge Graph Context")
                        st.code(result["graph_source"], language="sql") # Displaying the Cypher query result
                        st.subheader("Visual Knowledge Map")
                        # Note: Visualization may not work as well with the Cypher chain's text output.
                        # This is a known area for future improvement.
                        st.warning("Visualization is experimental and may not render for all queries.")
                        # visualize_graph_from_query(graph, result["graph_source"]) # Commented out for stability

                    st.session_state.messages.append({"role": "assistant", "content": result})

                except ResourceExhausted:
                    error_msg = "**API Quota Reached:** Apologies, the daily free usage limit has been reached. Please try again after midnight PT."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})