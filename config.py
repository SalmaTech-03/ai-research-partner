# config.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- LLM and EMBEDDING MODELS ---
LLM_MODEL_NAME = "gemini-2.5-pro"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- NEW: ARANGODB DATABASE ---
ARANGO_HOST = st.secrets.get("ARANGO_HOST")
ARANGO_DATABASE = st.secrets.get("ARANGO_DATABASE")
ARANGO_USER = st.secrets.get("ARANGO_USER")
ARANGO_PASSWORD = st.secrets.get("ARANGO_PASSWORD")

# --- GLOBAL INITIALIZATIONS ---
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
