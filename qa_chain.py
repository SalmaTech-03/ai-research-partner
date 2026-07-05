"""
qa_chain.py

High-reliability Retrieval Augmented Generation (RAG) pipeline.

Responsibilities
----------------
1. Fail-safe multi-source retrieval (Semantic, Graph, Web).
2. Robust context aggregation with 'None' safety.
3. Protected LLM invocation (Quota/Timeout resilience).
4. Strict grounding to eliminate hallucination.
5. Polymorphic API support for LangChain version compatibility.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Any

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

# -----------------------------------------------------------------------------
# Logging & Constants
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

MAX_RETRIEVED_DOCUMENTS = 5
WEB_SEARCH_KEYWORDS = [
    "latest", "recent", "new", "today", "2024", "2025", "2026",
    "research", "paper", "papers", "dataset", "datasets",
    "benchmark", "survey", "news", "trend", "state of the art", "sota",
]

# Professional Default Strings
DEFAULT_CONFIDENCE = "Medium"
DEFAULT_NO_GRAPH = "Knowledge graph context is currently unavailable."
DEFAULT_NO_DOCS = "No relevant document knowledge found in uploaded files."
DEFAULT_NO_WEB = "Live web search was not required for this inquiry."
DEFAULT_NO_REC = "No specific recommendations could be derived from the provided context."

# -----------------------------------------------------------------------------
# Google Search Tool
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_google_search_tool() -> Tool:
    """Returns a cached Google Search tool with environment-safe config."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    cse_id = st.secrets.get("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        logger.error("Search credentials missing in st.secrets")
        raise ValueError("Search API credentials are not configured.")

    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_CSE_ID"] = cse_id

    search = GoogleSearchAPIWrapper()

    return Tool(
        name="Google Search",
        description="Searches Google for recent information.",
        func=search.run,
    )

# -----------------------------------------------------------------------------
# Intent Detection
# -----------------------------------------------------------------------------
def requires_web_search(query: str) -> bool:
    """Decide whether the user's question needs live web information."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in WEB_SEARCH_KEYWORDS)

# -----------------------------------------------------------------------------
# Context Retrieval Functions
# -----------------------------------------------------------------------------
def retrieve_semantic_context(query: str, retriever: Any) -> str:
    """Retrieve relevant chunks from FAISS with polymorphic API support."""
    logger.info("Starting semantic retrieval...")
    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)
            
        docs = list(docs or [])[:MAX_RETRIEVED_DOCUMENTS]
        
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            context_parts.append(f"Source: {source}\nContent: {doc.page_content}")
            
        logger.info("Semantic context retrieved (Chunks: %d).", len(context_parts))
        return "\n\n".join(context_parts) if context_parts else DEFAULT_NO_DOCS
    except Exception:
        logger.exception("Semantic retrieval failed.")
        return DEFAULT_NO_DOCS

def retrieve_graph_context(query: str, graph_chain: Any) -> str:
    """Retrieve knowledge from Neo4j with polymorphic response parsing."""
    logger.info("Starting graph retrieval...")
    try:
        if hasattr(graph_chain, "invoke"):
            response = graph_chain.invoke({"query": query})
        else:
            response = graph_chain.run(query)

        if isinstance(response, dict):
            logger.info("Graph context retrieved (Dict).")
            return str(
                response.get("result") or 
                response.get("answer") or 
                response.get("output") or 
                response
            )

        logger.info("Graph context retrieved (Str).")
        return str(response) if response else DEFAULT_NO_GRAPH
    except Exception:
        logger.exception("Graph retrieval failed.")
        return DEFAULT_NO_GRAPH

def retrieve_web_context(query: str) -> str:
    """Retrieve info from Google Search with Tool API abstraction."""
    if not requires_web_search(query):
        logger.info("Skipping web search: Static intent detected.")
        return DEFAULT_NO_WEB

    logger.info("Starting Google Search...")
    try:
        search_tool = get_google_search_tool()
        search_query = f"Latest research and industry trends for: {query}"
        
        if hasattr(search_tool, "invoke"):
            result = search_tool.invoke(search_query)
        else:
            result = search_tool.run(search_query)
        
        if isinstance(result, dict):
            return str(result.get("output", result))

        logger.info("Web context retrieved.")
        return str(result)
    except Exception as e:
        logger.exception("Google search failed: %s", e)
        return "Live web search results currently unavailable."

# -----------------------------------------------------------------------------
# Master Prompt & Parsing
# -----------------------------------------------------------------------------
MASTER_PROMPT = PromptTemplate.from_template(
"""
You are an expert AI Research Assistant.

### CORE GROUNDING INSTRUCTIONS:
- Answer ONLY using the provided context.
- If the answer is not in the context, respond with: "I don't have enough information from the provided context to answer this."
- Never infer facts or use prior training knowledge.
- Never fabricate citations or information.

-------------------------------------------------------
AVAILABLE CONTEXT
-------------------------------------------------------
{context}

-------------------------------------------------------
QUESTION
-------------------------------------------------------
{question}

-------------------------------------------------------
INSTRUCTIONS
-------------------------------------------------------
Structure your response using EXACTLY the following Markdown structure.

# Main Answer
Provide the most complete answer possible.

# Citations
Specifically state which parts came from:
- Uploaded Documents
- Knowledge Graph
- Live Web Search

---
# Beginner Perspective
Explain simply for a non-technical audience.

---
# Expert Perspective
Provide technical details suitable for researchers.

---
# Interdisciplinary Perspective
Connect this topic to broader domains.

---
# Conflict Detection
Identify contradictions between sources. If none, say "No conflicts detected."

---
# Knowledge Gaps
Identify missing information needed for higher confidence.

---
# Trend Analysis
Summarize current research trends.

---
# Creative Ideas
Suggest innovative research directions.

---
# Recommendations
Recommend specific Papers, Authors, Datasets, or Tools.

---
# Confidence
High, Medium, or Low + one sentence explanation.
"""
)

SECTION_HEADERS = {
    "answer": "# Main Answer",
    "citations": "# Citations",
    "beginner": "# Beginner Perspective",
    "expert": "# Expert Perspective",
    "interdisciplinary": "# Interdisciplinary Perspective",
    "conflict": "# Conflict Detection",
    "gaps": "# Knowledge Gaps",
    "trends": "# Trend Analysis",
    "ideas": "# Creative Ideas",
    "recommendations": "# Recommendations",
    "confidence": "# Confidence",
}

def extract_section(text: str, start_header: str, end_headers: List[str]) -> str:
    """Extract a specific markdown section from the raw text."""
    if start_header not in text:
        return "Section not available."
    try:
        start_pos = text.index(start_header) + len(start_header)
        end_pos = len(text)
        for header in end_headers:
            pos = text.find(header, start_pos)
            if pos != -1:
                end_pos = min(end_pos, pos)
        return text[start_pos:end_pos].strip()
    except ValueError:
        return "Formatting error in generation."

def parse_llm_response(response: Any) -> Dict[str, str]:
    """Parse unified response into structured components."""
    text = response.content if hasattr(response, "content") else str(response)
    keys = list(SECTION_HEADERS.keys())
    headers = list(SECTION_HEADERS.values())
    parsed = {}
    for i, key in enumerate(keys):
        next_headers = headers[i + 1:]
        parsed[key] = extract_section(text, headers[i], next_headers)
    return parsed

# -----------------------------------------------------------------------------
# Public API (Protected Orchestration)
# -----------------------------------------------------------------------------
def generate_response(query: str, vector_retriever: Any, graph_qa_chain: Any, llm: Any) -> Dict[str, Any]:
    """Main fail-safe orchestration point."""
    logger.info("="*40)
    logger.info("New Research Inquiry: %s", query)

    # 1. Protected Multi-source Retrieval
    if vector_retriever:
        semantic_context = retrieve_semantic_context(query, vector_retriever)
    else:
        semantic_context = DEFAULT_NO_DOCS

    if graph_qa_chain:
        graph_context = retrieve_graph_context(query, graph_qa_chain)
    else:
        graph_context = DEFAULT_NO_GRAPH

    web_context = retrieve_web_context(query)

    # 2. Context Aggregation
    combined_context = f"""
[DATASET: UPLOADED DOCUMENTS]
{semantic_context}

[DATASET: KNOWLEDGE GRAPH]
{graph_context}

[DATASET: LIVE WEB RESULTS]
{web_context}
"""

    # 3. Protected LLM Generation
    logger.info("Invoking LLM...")
    try:
        chain = MASTER_PROMPT | llm
        response = chain.invoke({"context": combined_context, "question": query})
        parsed = parse_llm_response(response)
    except Exception:
        logger.exception("LLM generation phase failed.")
        return {
            "answer": "The AI Research Partner is currently unable to generate an answer. Please check your API quota or network connection.",
            "perspectives": "", "analytical_insights": "", "creative_insights": "", "recommendations": "",
            "semantic_sources": semantic_context, "graph_source": graph_context, "web_source": web_context
        }

    # 4. Refined Output Formatting
    main_answer = parsed.get("answer", "No answer generated.")
    citations = parsed.get("citations", "No citation information provided.")

    return {
        "answer": f"{main_answer}\n\n---\n\n### Data Sources\n{citations}",
        "perspectives": (
            f"## Beginner Perspective\n{parsed.get('beginner')}\n\n"
            f"--- \n"
            f"## Expert Perspective\n{parsed.get('expert')}\n\n"
            f"--- \n"
            f"## Interdisciplinary Perspective\n{parsed.get('interdisciplinary')}"
        ),
        "analytical_insights": (
            f"## Conflict Detection\n{parsed.get('conflict')}\n\n"
            f"## Knowledge Gaps\n{parsed.get('gaps')}\n\n"
            f"## Trend Analysis\n{parsed.get('trends')}\n\n"
            f"**Confidence Score:** {parsed.get('confidence', DEFAULT_CONFIDENCE)}"
        ),
        "creative_insights": parsed.get("ideas", "No suggestions available."),
        "recommendations": parsed.get("recommendations", DEFAULT_NO_REC),
        "semantic_sources": semantic_context,
        "graph_source": graph_context,
        "web_source": web_context
    }
