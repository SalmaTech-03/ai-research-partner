# app.py

import streamlit as st
from config import get_llm, get_embeddings_model
from data_processing import process_uploaded_pdf # Simplified import
from vector_store import create_faiss_vector_store
from graph_db import get_arangodb_graph, populate_graph_from_docs, get_graph_qa_chain
from qa_chain import generate_response
from visualization import visualize_graph_from_query
from google.api_core.exceptions import ResourceExhausted

st.set_page_config(page_title="🧠 AI Research Partner", layout="wide")
st.title("🧠 AI Research Partner")

# --- Initialize Models and Connections ---
llm = get_llm()
embeddings = get_embeddings_model()
graph = get_arangodb_graph()

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
    st.header("1. Upload Documents")
    
    # PDF Uploader is now the only input method
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            new_docs = []
            with st.spinner("Processing documents... This may take a few minutes."):
                try:
                    for file in uploaded_files:
                        if file.name not in st.session_state.processed_sources:
                            st.info(f"Processing PDF: {file.name}")
                            pdf_docs = process_uploaded_pdf(file)
                            if pdf_docs: # Ensure docs were processed
                                populate_graph_from_docs(graph, pdf_docs, llm, file.name)
                                new_docs.extend(pdf_docs)
                                st.session_state.processed_sources.add(file.name)
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

    st.header("Processed Documents")
    st.markdown(f"**{len(st.session_state.processed_sources)}** documents loaded.")
    with st.expander("View Documents"):
        for source in st.session_state.processed_sources:
            st.write(f"- {source}")

# --- Main UI: Chat Interface ---
st.header("2. Ask Your Research Question")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            # Re-display the tabbed output from history
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
                        st.code(result["graph_source"], language="sql")
                        st.subheader("Visual Knowledge Map")
                        visualize_graph_from_query(graph, result["graph_source"])

                    st.session_state.messages.append({"role": "assistant", "content": result})

                except ResourceExhausted:
                    error_msg = "**API Quota Reached:** Apologies, the daily free usage limit has been reached. Please try again after midnight PT."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})