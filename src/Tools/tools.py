import os
import json
import streamlit as st
from huggingface_hub import login

from langchain_community.document_loaders import WebBaseLoader, TextLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode


# ------------- AUTH -------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])


# ------------- Helper -------------
def load_vector_retriever(loader):
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(loader.load())

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    vectordb = Chroma.from_documents(docs, embeddings)
    return vectordb.as_retriever()


# ---------------- NASA TOOL ----------------
@tool
def nasa_tool(query: str) -> str:
    """Search NASA website content."""
    loader = WebBaseLoader("https://www.nasa.gov/")
    retriever = load_vector_retriever(loader)
    results = retriever.get_relevant_documents(query)
    return json.dumps([d.page_content[:400] for d in results], indent=2)


# ---------------- API TOOL ----------------
@tool
def api_tool(query: str) -> str:
    """Search REST API knowledge base."""
    loader = TextLoader("assets/knowledge_base1.txt", encoding="utf8")
    retriever = load_vector_retriever(loader)
    results = retriever.get_relevant_documents(query)
    return json.dumps([d.page_content[:400] for d in results], indent=2)


# ---------------- SATELLITE TOOL ----------------
@tool
def satellite_data_tool(query: str) -> str:
    """Search satellite details dataset."""
    loader = CSVLoader("assets/knowledge_base3.csv", encoding="utf8")
    retriever = load_vector_retriever(loader)
    results = retriever.get_relevant_documents(query)
    return json.dumps([d.page_content[:400] for d in results], indent=2)


# ---------------- SENSOR TOOL ----------------
@tool
def sensors_data_tool(query: str) -> str:
    """Search sensor raw dataset."""
    loader = CSVLoader("assets/sensor_raw_data.csv", encoding="utf8")
    retriever = load_vector_retriever(loader)
    results = retriever.get_relevant_documents(query)
    return json.dumps([d.page_content[:400] for d in results], indent=2)


# ---------------- GET TOOLS ----------------
def get_tools():
    return [
        nasa_tool,
        api_tool,
        satellite_data_tool,
        sensors_data_tool,
        TavilySearchResults(max_results=3),  # BaseTool
    ]


def create_tool_node(tools):
    return ToolNode(tools=tools)
