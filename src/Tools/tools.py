import os
import json
import streamlit as st
from huggingface_hub import login

from langchain_community.document_loaders import (
    WebBaseLoader, CSVLoader, TextLoader, PyPDFLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

# Shared embedding model
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# Document retrievers are stored in-memory per chat session
DOCUMENT_RETRIEVERS = {}  # key = session_id, value = retriever


# -------------------------------------------------------------------
# Utility: Build Retriever from text documents
# -------------------------------------------------------------------
def build_retriever_from_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, emb)
    return vectordb.as_retriever()


# -------------------------------------------------------------------
# Upload Processing
# -------------------------------------------------------------------
def process_uploaded_files(files, session_id):
    """Build or update retriever for uploaded files."""
    docs = []

    for file in files:
        ext = file.name.lower()

        if ext.endswith(".pdf"):
            loader = PyPDFLoader(file)
        elif ext.endswith(".csv"):
            loader = CSVLoader(file, encoding="utf8")
        elif ext.endswith(".txt"):
            loader = TextLoader(file, encoding="utf8")
        else:
            continue

        docs.extend(loader.load())

    # Build new retriever for this session
    DOCUMENT_RETRIEVERS[session_id] = build_retriever_from_docs(docs)


# -------------------------------------------------------------------
# Document Query Tool
# -------------------------------------------------------------------
@tool
def document_tool(query: str, session_id: str) -> str:
    """
    Query the user's uploaded documents.
    """
    retriever = DOCUMENT_RETRIEVERS.get(session_id)

    if not retriever:
        return "No documents uploaded yet. Please upload a file from the sidebar."

    results = retriever.invoke(query)
    return "\n\n".join([r.page_content[:700] for r in results])


# -------------------------------------------------------------------
# Existing Tools
# -------------------------------------------------------------------
@tool
def nasa_tool(query: str) -> str:
    loader = WebBaseLoader(
        web_path="https://www.nasa.gov/",
        requests_per_second=2,
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )
    docs = loader.load()
    retriever = build_retriever_from_docs(docs)
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


@tool
def api_tool(query: str) -> str:
    loader = TextLoader("assets/knowledge_base1.txt", encoding="utf8")
    retriever = build_retriever_from_docs(loader.load())
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


@tool
def satellite_data_tool(query: str) -> str:
    loader = CSVLoader("assets/knowledge_base3.csv", encoding="utf8")
    retriever = build_retriever_from_docs(loader.load())
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


@tool
def sensors_data_tool(query: str) -> str:
    loader = CSVLoader("assets/sensor_raw_data.csv", encoding="utf8")
    retriever = build_retriever_from_docs(loader.load())
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


# -------------------------------------------------------------------
# Expose all tools (doc tool included)
# -------------------------------------------------------------------
def get_tools():
    tavily = TavilySearchResults(max_results=3)
    return [
        nasa_tool,
        api_tool,
        satellite_data_tool,
        sensors_data_tool,
        document_tool,
        tavily,
    ]
