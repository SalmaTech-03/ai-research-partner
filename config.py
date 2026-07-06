import streamlit as st
from pydantic_settings import BaseSettings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

class Settings(BaseSettings):
    # Credentials pulled from Streamlit Secrets
    google_api_key: str = st.secrets["GOOGLE_API_KEY"]
    neo4j_uri: str = st.secrets["NEO4J_URI"]
    neo4j_user: str = st.secrets["NEO4J_USERNAME"]
    neo4j_password: str = st.secrets["NEO4J_PASSWORD"]
    app_token: str = st.secrets.get("APP_TOKEN", "research_dev_2024")

settings = Settings()

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=settings.google_api_key)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
