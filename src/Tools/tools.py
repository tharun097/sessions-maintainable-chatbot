"""
Tools module for the Q&A Chatbot.

This module defines the tools that the LangGraph agent can call.
Tools include:
 - nasa_tool: search NASA website content
 - api_tool: search a local REST API knowledge base
 - satellite_data_tool: search a satellite CSV knowledge base
 - sensors_data_tool: search a sensor CSV dataset
 - document_tool: query documents uploaded by the user (session-scoped)
 - get_tools / create_tool_node: helpers to expose tools to the graph

All tools use a shared HuggingFace embedding model and Chroma vector store
for retrieval. Document retrievers built from uploaded user files are stored
in-memory per-session (DOCUMENT_RETRIEVERS).
"""

from __future__ import annotations

import os
import json
import tempfile
from typing import List, Dict, Optional

import streamlit as st
from huggingface_hub import login

from langchain_community.document_loaders import (
    WebBaseLoader,
    CSVLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

# -------------------------------------------------------------------------
# ENV + AUTH
# -------------------------------------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

# -------------------------------------------------------------------------
# EMBEDDINGS (Shared for All Tools)
# -------------------------------------------------------------------------
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# In-memory per-session document retrievers
DOCUMENT_RETRIEVERS: Dict[str, "VectorStoreRetriever"] = {}  # type: ignore[name-defined]


# -------------------------------------------------------------------------
# Helper: Build a Retriever
# -------------------------------------------------------------------------
def build_retriever_from_docs(docs) -> "VectorStoreRetriever":  # type: ignore[name-defined]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, emb)
    return vectordb.as_retriever()


def build_retriever(loader) -> "VectorStoreRetriever":  # type: ignore[name-defined]
    docs = loader.load()
    return build_retriever_from_docs(docs)


# -------------------------------------------------------------------------
# File Loader Helper (Fix for Streamlit UploadedFile)
# -------------------------------------------------------------------------
def load_uploaded_file(file) -> List:
    """
    Safely load Streamlit UploadedFile using langchain loaders.

    Saves UploadFile into a temporary file path so loaders accept it.
    """
    suffix = ""
    if file.name.lower().endswith(".pdf"):
        suffix = ".pdf"
    elif file.name.lower().endswith(".csv"):
        suffix = ".csv"
    elif file.name.lower().endswith(".txt"):
        suffix = ".txt"
    else:
        return []  # unsupported

    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    # Use the correct loader
    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".csv":
        loader = CSVLoader(tmp_path, encoding="utf8")
    else:
        loader = TextLoader(tmp_path, encoding="utf8")

    return loader.load()


# -------------------------------------------------------------------------
# Document Upload Handler
# -------------------------------------------------------------------------
def process_uploaded_files(
    files: List[st.runtime.uploaded_file_manager.UploadedFile], session_id: str
) -> None:
    """
    Process uploaded files and build/update a session-scoped retriever.

    Supported file types:
     - PDF (.pdf)
     - CSV (.csv)
     - Plain text (.txt)
    """
    all_docs = []

    for file in files:
        loaded_docs = load_uploaded_file(file)
        all_docs.extend(loaded_docs)

    # If nothing was loaded (unsupported types or empty list) → skip
    if not all_docs:
        return

    DOCUMENT_RETRIEVERS[session_id] = build_retriever_from_docs(all_docs)


# -------------------------------------------------------------------------
# Document Tool
# -------------------------------------------------------------------------
@tool
def document_tool(query: str, session_id: str) -> str:
    """
    Query the documents uploaded by the user for a specific session.

    This tool is intended to be called by the LLM using the session_id that
    the app injects into the tool call. If no documents have been uploaded
    for the session, the tool will ask the user to upload first.

    Args:
        query: The query string to search across the uploaded documents.
        session_id: The session identifier used to locate the session retriever.

    Returns:
        A string containing concatenated top document chunks (truncated).
    """
    retriever = DOCUMENT_RETRIEVERS.get(session_id)
    if not retriever:
        return (
            "No documents uploaded yet for this session. Please upload files via the sidebar."
        )

    results = retriever.invoke(query)
    return "\n\n".join([r.page_content[:800] for r in results])


# -------------------------------------------------------------------------
# Domain Tools (NASA, REST API, Satellite, Sensor)
# -------------------------------------------------------------------------
@tool
def nasa_tool(query: str) -> str:
    """
    Search the NASA website and return relevant excerpts.

    Args:
        query: The natural language query to run against NASA website content.

    Returns:
        A string with relevant document excerpts (truncated).
    """
    loader = WebBaseLoader(
        web_path="https://www.nasa.gov/",
        requests_per_second=2,
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )
    retriever = build_retriever(loader)
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


@tool
def api_tool(query: str) -> str:
    """
    Search a local REST API knowledge base text file.

    Args:
        query: The question/query to search inside the REST API KB.

    Returns:
        A string of top-matching text chunks.
    """
    loader = TextLoader("assets/knowledge_base1.txt", encoding="utf8")
    retriever = build_retriever(loader)
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


@tool
def satellite_data_tool(query: str) -> str:
    """
    Search the satellite CSV dataset.

    Args:
        query: The query string to search inside satellite knowledge CSV.

    Returns:
        String containing matching rows / excerpts (truncated).
    """
    loader = CSVLoader("assets/knowledge_base3.csv", encoding="utf8")
    retriever = build_retriever(loader)
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


@tool
def sensors_data_tool(query: str) -> str:
    """
    Search the sensors raw data CSV and return relevant excerpts.

    Args:
        query: The question / search string for the sensors dataset.

    Returns:
        Concatenated matching rows or snippets (truncated).
    """
    loader = CSVLoader("assets/sensor_raw_data.csv", encoding="utf8")
    retriever = build_retriever(loader)
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content[:500] for d in results])


# -------------------------------------------------------------------------
# Tools Exposure
# -------------------------------------------------------------------------
def get_tools() -> List:
    tavily = TavilySearchResults(max_results=3)
    return [
        nasa_tool,
        api_tool,
        satellite_data_tool,
        sensors_data_tool,
        document_tool,  # upload-aware
        tavily,
    ]


def create_tool_node(tools: List) -> ToolNode:
    return ToolNode(tools=tools)
