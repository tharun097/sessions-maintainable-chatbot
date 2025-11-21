from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
# from langchain.tools import create_retriever_tool
# from langchain.tools.retriever import create_retriever_tool
import streamlit as st
from langchain_core.tools import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
import os 
from huggingface_hub import login
# from dotenv import load_dotenv 
# load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["TAVILY_API_KEY"]=st.secrets["TAVILY_API_KEY"]
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
# ---------------- NASA TOOL ----------------
def nasa_tool():
    web_content_loader = WebBaseLoader(
        web_path="https://www.nasa.gov/",
        requests_per_second=2,
        bs_kwargs={},
        bs_get_text_kwargs={"separator": "\n", "strip": True}
    )
    web_data = web_content_loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(web_data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    return create_retriever_tool(
        retriever=retriever,
        name="nasa_data_knowledge_base",
        description="Search and run information about NASA"
    )


# ---------------- API TOOL ----------------
def api_tool():
    loader = TextLoader(
        file_path="assets/knowledge_base1.txt",
        encoding="utf8"
    )
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    return create_retriever_tool(
        retriever=retriever,
        name="rest_api_knowledge_base",
        description="Search and run information about REST APIs"
    )


# ---------------- SATELLITE TOOL ----------------
def satellite_data_tool():
    loader = CSVLoader(
        file_path="assets/knowledge_base3.csv",
        encoding="utf8"
    )
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    return create_retriever_tool(
        retriever=retriever,
        name="satellite_knowledge_base",
        description="Search information about satellites"
    )


# ---------------- SENSOR TOOL ----------------
def sensors_data_tool():
    loader = CSVLoader(
        file_path="assets/sensor_raw_data.csv",
        encoding="utf8"
    )
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    return create_retriever_tool(
        retriever=retriever,
        name="sensor_data_knowledge_base",
        description="Search and run information about sensor raw data"
    )
def get_tools():
    """Returns a list of available tools."""
    tavily_tool = TavilySearchResults(max_results=3)
    tools = [nasa_tool(), api_tool(), satellite_data_tool(), sensors_data_tool(), tavily_tool]
    return tools

def create_tool_node(tools):
    """Creates a ToolNode for the given tool."""
    return ToolNode(tools=tools)
    
