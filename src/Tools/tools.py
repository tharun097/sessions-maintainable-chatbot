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
import streamlit as st
from typing import List, Dict, Optional

from huggingface_hub import login

from langchain_community.document_loaders import WebBaseLoader, CSVLoader, TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

# -------------------------------------------------------------------------
# Environment & Auth
# -------------------------------------------------------------------------
# Ensure secrets exist in Streamlit secrets. Adjust as needed for your env.
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
# Login to HF - required for downloading sentence-transformers models
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

# -------------------------------------------------------------------------
# Shared embedding model
# -------------------------------------------------------------------------
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# In-memory per-session retrievers built from user uploads
DOCUMENT_RETRIEVERS: Dict[str, "VectorStoreRetriever"] = {}  # type: ignore[name-defined]


# -------------------------------------------------------------------------
# Helper: build_retriever
# -------------------------------------------------------------------------
def build_retriever(loader) -> "VectorStoreRetriever":  # type: ignore[name-defined]
    """
    Build a vectorstore retriever from a LangChain document loader.

    Steps:
    1. Load documents using provided loader.
    2. Split documents into chunks.
    3. Create embeddings with the shared HuggingFace model.
    4. Index chunks in Chroma and return an as_retriever() object.

    Args:
        loader: A LangChain document loader instance (has .load()).

    Returns:
        A VectorStoreRetriever-like object that supports .invoke(query).
    """
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_splits = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(doc_splits, emb)
    return vectordb.as_retriever()


# -------------------------------------------------------------------------
# Document upload handling (session-aware)
# -------------------------------------------------------------------------
def process_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile], session_id: str) -> None:
    """
    Process uploaded files and build/update a session-scoped retriever.

    Supported file types:
     - PDF (.pdf)
     - CSV (.csv)
     - Plain text (.txt)

    The function loads each file via a suitable LangChain loader, combines
    all documents, builds a Chroma vector store, and stores the retriever in
    the DOCUMENT_RETRIEVERS dict under the session_id.

    Args:
        files: List of Streamlit UploadedFile objects from file_uploader.
        session_id: The chat session identifier (string).
    """
    docs = []
    for f in files:
        fname = f.name.lower()
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(f)
        elif fname.endswith(".csv"):
            loader = CSVLoader(f, encoding="utf8")
        elif fname.endswith(".txt"):
            loader = TextLoader(f, encoding="utf8")
        else:
            # skip unsupported file type; you can extend to docx, json, etc.
            continue
        docs.extend(loader.load())

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(chunks, emb)
        DOCUMENT_RETRIEVERS[session_id] = vectordb.as_retriever()


# -------------------------------------------------------------------------
# Document tool (session-aware)
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
        return "No documents uploaded yet for this session. Please upload files via the sidebar."

    results = retriever.invoke(query)
    # join the top results into a human-friendly string, truncate each chunk
    return "\n\n".join([r.page_content[:800] for r in results])


# -------------------------------------------------------------------------
# Domain tools
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
# Tools exposure helpers
# -------------------------------------------------------------------------
def get_tools() -> List:
    """
    Return the list of tool callables / BaseTool objects to expose to LangGraph.

    The order does not matter; document_tool is included so the LLM can query
    user-uploaded documents. TavilySearchResults is included as a BaseTool.
    """
    tavily = TavilySearchResults(max_results=3)
    return [nasa_tool, api_tool, satellite_data_tool, sensors_data_tool, document_tool, tavily]


def create_tool_node(tools: List) -> ToolNode:
    """
    Create and return a LangGraph ToolNode configured with the provided tools.

    Args:
        tools: A list of tool callables and/or BaseTool objects (as returned by get_tools()).

    Returns:
        ToolNode instance suitable for adding to the graph.
    """
    return ToolNode(tools=tools)
