import os
from langchain_community.vectorstores import FAISS

STORAGE_DIR = "storage/vector"

def get_vector_retriever(docs, embeddings, ns: str):
    path = os.path.join(STORAGE_DIR, ns)
    if os.path.exists(os.path.join(path, "index.faiss")):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever()
    
    os.makedirs(path, exist_ok=True)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(path)
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 10})
