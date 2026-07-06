import hashlib, tempfile, os, logging
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

logger = logging.getLogger(__name__)

def get_namespace(file_bytes: bytes) -> str:
    """Full SHA-256 for idempotent multi-tenancy."""
    return hashlib.sha256(file_bytes).hexdigest()

def get_chunk_hash(content: str, version: str = "v1") -> str:
    return hashlib.sha256(f"{version}:{content}".encode()).hexdigest()

def process_pdf(uploaded_file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        loader = PyPDFLoader(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(loader.load())
        for d in chunks: d.metadata["source"] = uploaded_file.name
        return chunks
    finally:
        if os.path.exists(path): os.remove(path)

def process_url(url: str) -> List[Document]:
    try:
        loader = WebBaseLoader(url)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        return splitter.split_documents(loader.load())
    except Exception as e:
        logger.error(f"Scrape Failed: {e}")
        return []
