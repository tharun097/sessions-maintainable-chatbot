import os
import json
import streamlit as st
from huggingface_hub import login

from langchain_community.document_loaders import WebBaseLoader, CSVLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

# -------------------------------------------------
# ENV SETUP
# -------------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

# -------------------------------------------------
# COMMON FUNCTION: Build Retriever from Loader
# -------------------------------------------------
def build_retriever(loader):
    """Loads docs → splits → embeddings → vectorDB → retriever"""
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    vectordb = Chroma.from_documents(doc_splits, embeddings)
    return vectordb.as_retriever()


# -------------------------------------------------
# NASA TOOL
# -------------------------------------------------
@tool
def nasa_tool(query: str) -> str:
    """Search information about NASA"""
    loader = WebBaseLoader(
        web_path="https://www.nasa.gov/",
        requests_per_second=2,
        bs_kwargs={},
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )

    retriever = build_retriever(loader)
    results = retriever.invoke(query)

    return "\n\n".join([d.page_content[:500] for d in results])


# -------------------------------------------------
# REST API TOOL
# -------------------------------------------------
@tool
def api_tool(query: str) -> str:
    """Search knowledge base of REST APIs"""
    loader = TextLoader("assets/knowledge_base1.txt", encoding="utf8")

    retriever = build_retriever(loader)
    results = retriever.invoke(query)

    return "\n\n".join([d.page_content[:500] for d in results])


# -------------------------------------------------
# SATELLITE TOOL
# -------------------------------------------------
@tool
def satellite_data_tool(query: str) -> str:
    """Search satellite dataset"""
    loader = CSVLoader("assets/knowledge_base3.csv", encoding="utf8")

    retriever = build_retriever(loader)
    results = retriever.invoke(query)

    return "\n\n".join([d.page_content[:500] for d in results])


# -------------------------------------------------
# SENSOR TOOL
# -------------------------------------------------
@tool
def sensors_data_tool(query: str) -> str:
    """Search sensor raw dataset"""
    loader = CSVLoader("assets/sensor_raw_data.csv", encoding="utf8")

    retriever = build_retriever(loader)
    results = retriever.invoke(query)

    return "\n\n".join([d.page_content[:500] for d in results])


# -------------------------------------------------
# Define Tools List
# -------------------------------------------------
def get_tools():
    tavily = TavilySearchResults(max_results=3)
    return [nasa_tool, api_tool, satellite_data_tool, sensors_data_tool, tavily]


# -------------------------------------------------
# Tool Node
# -------------------------------------------------
def create_tool_node(tools):
    return ToolNode(tools=tools)
