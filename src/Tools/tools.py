from langchain_community.document_loaders import WebBaseLoader, CSVLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from huggingface_hub import login
import os
import streamlit as st


# ------------------------------------------
# AUTH
# ------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])


# ------------------------------------------
# HELPER — build retriever tool
# ------------------------------------------
def build_retriever_tool(docs, name, desc):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    return create_retriever_tool(
        retriever=retriever,
        name=name,
        description=desc,
    )


# ------------------------------------------
# TOOLS (no execution at import time)
# ------------------------------------------
def nasa_tool():
    loader = WebBaseLoader(
        web_path="https://www.nasa.gov/",
        requests_per_second=2,
        bs_kwargs={},
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(loader.load())

    return build_retriever_tool(
        docs,
        "nasa_data_knowledge_base",
        "Search and run information about NASA",
    )


def api_tool():
    loader = TextLoader("assets/knowledge_base1.txt", encoding="utf8")
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(loader.load())

    return build_retriever_tool(
        docs,
        "rest_api_knowledge_base",
        "Search and run information about REST APIs",
    )


def satellite_data_tool():
    loader = CSVLoader("assets/knowledge_base3.csv", encoding="utf8")
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(loader.load())

    return build_retriever_tool(
        docs,
        "satellite_knowledge_base",
        "Search information about satellites",
    )


def sensors_data_tool():
    loader = CSVLoader("assets/sensor_raw_data.csv", encoding="utf8")
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(loader.load())

    return build_retriever_tool(
        docs,
        "sensor_data_knowledge_base",
        "Search and run information about sensor raw data",
    )


# ------------------------------------------
# THIS IS THE REAL FIX — Create tools only HERE
# ------------------------------------------
def get_tools():
    tools = [
        nasa_tool(),
        api_tool(),
        satellite_data_tool(),
        sensors_data_tool(),
        TavilySearchResults(max_results=3),  # already BaseTool
    ]
    return tools


def create_tool_node(tools):
    return ToolNode(tools=tools)
