# qa_chain.py

from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
import os
import streamlit as st

@st.cache_resource
def get_google_search_tool():
    """Initializes the Google Search tool."""
    os.environ["GOOGLE_CSE_ID"] = st.secrets.get("GOOGLE_CSE_ID")
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY")
    
    search = GoogleSearchAPIWrapper()
    return Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )

def generate_response(query, vector_retriever, graph_qa_chain, llm):
    """
    Generates a main response and a full suite of analytical, creative, and recommendation insights.
    """
    search_tool = get_google_search_tool()

    # 1. Retrieve context from all sources
    semantic_context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
        for doc in vector_retriever.get_relevant_documents(query)
    )
    try:
        graph_context = graph_qa_chain.run(query)
    except Exception:
        graph_context = "No information found in the knowledge graph for this query."
    
    web_context = search_tool.run(f"latest research papers, datasets, and influential authors on the topic of: {query}")
        
    combined_context = (
        f"--- CONTEXT FROM YOUR DOCUMENTS (SEMANTIC SEARCH) ---\n{semantic_context}\n\n"
        f"--- CONTEXT FROM YOUR KNOWLEDGE GRAPH ---\n{graph_context}\n\n"
        f"--- CONTEXT FROM A LIVE GOOGLE SEARCH ---\n{web_context}"
    )
    
    # 2. Generate Main Answer
    main_answer_template = """
    You are an expert research assistant. Your goal is to provide a comprehensive and accurate answer by synthesizing information from multiple sources.
    Answer the user's question based ONLY on the provided context below.
    If the context is insufficient, state that clearly. Do not use external knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ACCURATE AND SYNTHESIZED ANSWER:
    """
    main_prompt = PromptTemplate(template=main_answer_template, input_variables=["context", "question"])
    main_chain = main_prompt | llm
    main_answer = main_chain.invoke({"context": combined_context, "question": query}).content
    
    # 3. Generate Multi-Perspective Summaries
    perspectives_template = """
    Based on the following CONTEXT, provide three distinct summaries of the answer to the QUESTION.
    Structure your response EXACTLY as follows, using Markdown for headings:
    ### Beginner's Perspective
    (A simple, high-level summary using easy-to-understand language.)
    ### Expert's Perspective
    (A detailed, technical summary for an expert in the field.)
    ### Interdisciplinary Perspective
    (A summary connecting the ideas to broader concepts from other domains.)

    CONTEXT:
    {context}
    QUESTION:
    {question}
    """
    perspectives_prompt = PromptTemplate(template=perspectives_template, input_variables=["context", "question"])
    perspectives_chain = perspectives_prompt | llm
    perspectives = perspectives_chain.invoke({"context": combined_context, "question": query}).content

    # 4. Generate Analytical Insights
    analytical_template = """
    You are a critical research analyst. Based on the CONTEXT, provide the following insights for the QUESTION. Use these exact Markdown headings:
    ### Conflict Detection
    (Identify contradictions between sources. If none, state that.)
    ### Gap Finder
    (Identify what important information is missing. If none, state that.)
    ### Trend Insights
    (Identify emerging trends based on the live web search.)

    CONTEXT:
    {context}
    QUESTION:
    {question}
    """
    analytical_prompt = PromptTemplate(template=analytical_template, input_variables=["context", "question"])
    analytical_chain = analytical_prompt | llm
    analytical_insights = analytical_chain.invoke({"context": combined_context, "question": query}).content

    # 5. Generate Creative Insights
    creative_template = """
    You are a creative strategist. Based on the CONTEXT, provide the following insights for the QUESTION. Use these exact Markdown headings:
    ### New Idea Suggestions
    (Suggest a novel connection or new research direction.)
    ### Scenario Answers
    (Address the question as a "what if" scenario and provide a plausible prediction.)

    CONTEXT:
    {context}
    QUESTION:
    {question}
    """
    creative_prompt = PromptTemplate(template=creative_template, input_variables=["context", "question"])
    creative_chain = creative_prompt | llm
    creative_insights = creative_chain.invoke({"context": combined_context, "question": query}).content

    # 6. Generate Recommendations
    recommendation_template = """
    You are a research librarian. Based on the live web search CONTEXT and the user's QUESTION, recommend related materials. Use these exact Markdown headings:
    ### Related Papers or Articles
    (List 2-3 relevant papers or articles with brief explanations.)
    ### Key Datasets
    (Suggest 1-2 relevant public datasets. If none, state that.)
    ### Influential Authors or Labs
    (List 1-2 key researchers or labs in this topic area.)

    CONTEXT (Live Web Search):
    {web_context}
    QUESTION:
    {question}
    """
    recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=["web_context", "question"])
    recommendation_chain = recommendation_prompt | llm
    recommendations = recommendation_chain.invoke({"web_context": web_context, "question": query}).content

    # 7. Return all generated content
    return {
        "answer": main_answer,
        "perspectives": perspectives,
        "analytical_insights": analytical_insights,
        "creative_insights": creative_insights,
        "recommendations": recommendations,
        "semantic_sources": semantic_context,
        "graph_source": str(graph_context)
    }