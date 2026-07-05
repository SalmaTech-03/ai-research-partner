# data_processing.py

"""
Document processing utilities.

Responsibilities
----------------
1. Load PDF documents
2. Split documents into semantic chunks
3. Preserve metadata
4. Clean temporary files
5. Prepare documents for FAISS and Neo4j
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
)

# ---------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------

def process_uploaded_pdf(uploaded_file) -> List[Document]:
    """
    Load an uploaded PDF and split it into chunks.
    """

    if uploaded_file is None:
        return []

    logger.info("Processing PDF: %s", uploaded_file.name)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        for page in documents:
            page.metadata["source"] = uploaded_file.name

        chunks = text_splitter.split_documents(documents)

        logger.info(
            "Generated %d chunks from %s",
            len(chunks),
            uploaded_file.name,
        )

        return chunks

    except Exception:
        logger.exception("Failed to process PDF.")
        return []

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                logger.warning("Unable to remove temporary file.")

# ---------------------------------------------------------------------
# Future Extensions
# ---------------------------------------------------------------------

def process_url(url: str):
    """
    Placeholder for website ingestion.

    Replace this later with:
    - WebBaseLoader
    - FireCrawl
    - Jina Reader
    - Crawl4AI

    depending on your preferred crawler.
    """

    raise NotImplementedError(
        "URL processing has not yet been implemented."
    )
