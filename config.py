# config.py

"""
Central configuration for the AI Research Partner.

Responsibilities
----------------
1. Load API keys and secrets
2. Initialize Gemini LLM
3. Initialize HuggingFace embeddings
4. Validate required configuration
5. Cache expensive resources
"""

from __future__ import annotations

import logging

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------

LLM_MODEL_NAME = "gemini-1.5-flash"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TEMPERATURE = 0.0

# ---------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID")

NEO4J_URI = st.secrets.get("NEO4J_URI")

NEO4J_USERNAME = st.secrets.get("NEO4J_USERNAME")

NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD")

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_configuration() -> None:
    """
    Validate required configuration before the application starts.
    """

    missing = []

    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")

    if not NEO4J_URI:
        missing.append("NEO4J_URI")

    if not NEO4J_USERNAME:
        missing.append("NEO4J_USERNAME")

    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise RuntimeError(
            "Missing required Streamlit secrets:\n\n"
            + "\n".join(f"- {item}" for item in missing)
        )


# ---------------------------------------------------------------------
# Cached LLM
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns a cached Gemini LLM instance.
    """

    validate_configuration()

    logger.info("Loading Gemini model...")

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        convert_system_message_to_human=True,
        max_retries=3,
        timeout=120,
    )


# ---------------------------------------------------------------------
# Cached Embedding Model
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Returns cached embedding model.
    """

    logger.info("Loading embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )
