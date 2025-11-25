import streamlit as st
from langchain_groq import ChatGroq
import pdfplumber

st.title("📄 Document Question Answering (Groq LLM)")
st.write("Upload a document and ask questions. Powered by **Groq Llama Models**.")

# API Key Input
groq_api_key = st.text_input("🔑 Enter your GROQ API Key", type="password")
if not groq_api_key:
    st.info("Please enter your GROQ API key.", icon="🗝️")
    st.stop()

# LLM (Groq)
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    streaming=True
)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload your document (.pdf, .txt, .md)",
    type=["pdf", "txt", "md"]
)

# Extract text function
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    # TXT or MD
    return uploaded_file.read().decode("utf-8", errors="ignore")

# UI text input
question = st.text_area(
    "❓ Your question about the document:",
    disabled=not uploaded_file
)

# Run LLM only if we have a document & question
if uploaded_file and question:
    
    document_text = extract_text_from_file(uploaded_file)

    messages = [
        {
            "role": "user",
            "content": f"Here is the document content:\n\n{document_text}\n\n---\nQuestion: {question}"
        }
    ]

    async def stream_response():
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    st.write("🤖 **Answer:**")
    st.write_stream(stream_response())
