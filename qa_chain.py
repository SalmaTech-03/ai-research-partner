import json, time
from sentence_transformers import CrossEncoder
from graph_db import retrieve_facts

class RAGOrchestrator:
    def __init__(self):
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def generate_response(self, query, retriever, graph, llm, ns):
        start = time.time()
        # 1. Intent Extraction
        p = f"Extract tech entities from: '{query}' as JSON: {{'entities':[]}}"
        ents = json.loads(llm.invoke(p).content).get("entities", [])
        
        # 2. Parallel Retrieval
        chunks = [d.page_content for d in retriever.invoke(query)]
        facts = retrieve_facts(graph, ents, ns).split("\n")
        
        # 3. Neural Reranking
        candidates = list(set(chunks + facts))
        scores = self.reranker.predict([[query, c] for c in candidates])
        optimized_context = [candidates[i] for i in scores.argsort()[-5:][::-1]]
        
        # 4. Synthesis
        res = llm.invoke(f"Use Context to answer: {query}\n\nContext:\n{optimized_context}")
        return {"answer": res.content, "latency": time.time() - start}
