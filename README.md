# 🤖 AI Chatbot with LangGraph, LangChain & Groq  
### A Production-grade Multi-Tool, Multi-Session AI Assistant with Streaming, Memory

This project is an end-to-end intelligent chatbot application built using **Groq Llama-3.1**, **LangGraph workflow**, vector search (ChromaDB), and a clean **ChatGPT-style Streamlit UI**.

It includes multi-session chat, tool-augmented reasoning, domain-specific knowledge retrieval, and user document analysis — all running extremely fast thanks to **Groq’s low-latency inference**.

---

## 🚀 Features

### 🧠 **Graph-based AI Agent (LangGraph)**
- Agent workflow built on LangGraph  
- Handles tool routing automatically  
- Memory-enabled conversations with `MemorySaver`  
- Supports tool → LLM → tool cycles  

### 🔧 **Integrated Tools**
The agent can call domain-specific tools:

| Tool | Purpose |
|------|---------|
| `nasa_tool` | NASA website content retrieval |
| `api_tool` | REST API local knowledge base |
| `satellite_data_tool` | Satellite dataset queries |
| `sensors_data_tool` | Sensor raw data retrieval |
| `tavily_search` | Internet search results |

All tools use:
- HuggingFace Embeddings (MiniLM-L6-v2)  
- ChromaDB vector store  

### 📄 **Document Analysis (Outside LangGraph)**
Users can upload:
- PDF  
- TXT  
- MD  
- Word files  

And chat with the content using Groq streaming.

### 💬 **Multi-Session Chat UI**
- Sidebar to switch sessions  
- Each session stores:
  - Chat history  
  - Memory  
  - Uploaded file text  

### ⚡ **Ultra-Fast Streaming**
- Groq API used for chat streaming  
- Nearly instant response generation  

### 🎨 **Modern Streamlit UI**
- Chat-like layout  
- Session management  
- Tool output panels  
- File uploader  

---


---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| LLM | Groq Llama-3.1 (8B Instant) |
| Agent Framework | LangGraph |
| Tools | LangChain Tools |
| Memory | MemorySaver |
| Embeddings | MiniLM-L6-v2 |
| Vector DB | Chroma |
| Frontend | Streamlit |
| File Parsing | PyPDFLoader, CSVLoader, TextLoader |
| Language | Python 3.10+ |

---

## 📥 Installation Guide

### 1️⃣ Clone the Repository  

git clone https://github.com/tharun097/sessions-maintainable-chatbot.git

### 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Add Secrets

Create .streamlit/secrets.toml:

GROQ_API_KEY="your_groq_api_key"
TAVILY_API_KEY="your_tavily_api_key"
HUGGINGFACEHUB_API_TOKEN="your_hf_token"
LANGCHAIN_API_KEY="your_langchain_key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT_NAME="Q&A Chatbot"

### ▶️ How to Run the App
streamlit run app.py

### 📄 Directory Structure
<img width="624" height="310" alt="image" src="https://github.com/user-attachments/assets/db85d0af-c222-42f7-b38e-44773c3937f6" />

