import streamlit as st
from config import get_llm, get_embeddings, settings
from data_processing import get_namespace, process_pdf, process_url
from graph_db import get_neo4j_graph, populate_graph, initialize_graph
from vector_store import get_vector_retriever
from qa_chain import RAGOrchestrator

st.set_page_config(page_title="AI Research Partner", layout="wide")

# Simple Access Control
if st.sidebar.text_input("SaaS Token", type="password") != settings.app_token:
    st.info("Authentication Required.")
    st.stop()

llm, embeddings, graph = get_llm(), get_embeddings(), get_neo4j_graph()
initialize_graph(graph)
orch = RAGOrchestrator()

if "ns" not in st.session_state: st.session_state.ns = None

with st.sidebar:
    st.header("Ingestion")
    file = st.file_uploader("Upload PDF", type="pdf")
    if file and st.button("Analyze"):
        ns = get_namespace(file.getvalue())
        st.session_state.ns = ns
        docs = process_pdf(file)
        populate_graph(graph, docs, llm, ns)
        st.session_state.retriever = get_vector_retriever(docs, embeddings, ns)
        st.success("Ready.")

if st.session_state.ns:
    if prompt := st.chat_input("Research query..."):
        retriever = st.session_state.get("retriever") or get_vector_retriever([], embeddings, st.session_state.ns)
        res = orch.generate_response(prompt, retriever, graph, llm, st.session_state.ns)
        st.markdown(res["answer"])
        st.caption(f"Reasoning Latency: {res['latency']:.2f}s")
