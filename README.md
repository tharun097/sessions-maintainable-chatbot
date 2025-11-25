# 📄 Document Q&A App  
### Built with **Groq LLMs** + **Streamlit** + **Real-time Streaming**

A simple app where you can upload a document (PDF/TXT/MD/Word) and ask questions about it.  
The app sends the document to a Groq-powered LLM and streams the answer in real time.

---

## 🚀 Features
- Upload documents (`.pdf`, `.txt`, `.md`, `.word`)
- Ask any question about the file
- Real-time streaming responses (ChatGPT-style)
- Clean, minimal Streamlit UI
- Powered by **Groq’s ultra-fast inference**

---

## 🛠️ Tech Used
- **Python**
- **Streamlit**
- **Groq Chat Completions API**
- **GROQAPI-compatible client**
- **Real-time streaming output**

---

## ▶️ How to Run

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Add your Groq API Key

Use the input box in the app or store in .streamlit/secrets.toml:

GROQ_API_KEY="your_groq_key"

### 3️⃣ Run the app
streamlit run app.py

### 📌 Usage

- Upload a document

- Enter your question

- Watch the LLM stream back the answer

### Ask more questions based on the same document

- 📂 Example Questions

    - “Give a short summary of this document.”

    - “Extract key points.”

    - “What are the main topics discussed?”

    - “Rewrite this professionally.”
