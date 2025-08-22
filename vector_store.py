# vector_store.py

from langchain_community.vectorstores import FAISS

def create_faiss_vector_store(docs, embeddings):
    """
    Creates a FAISS vector store from document chunks and returns a retriever.
    """
    if not docs:
        return None
        
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()