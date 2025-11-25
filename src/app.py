import os
import uuid
import streamlit as st
from LLM.groqllm import GroqLLM
from Graph_Workflow.graph import Graphbuilder
from Tools.tools import process_uploaded_files, DOCUMENT_RETRIEVERS

# --- Setup ---
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
api_key = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Q&A Chatbot", page_icon="💬")


# --- Load graph once ---
@st.cache_resource
def load_graph():
    model = GroqLLM(llm="llama-3.1-8b-instant", api_key=api_key).load_llm()
    return Graphbuilder(model).get_chatbot_graph()

graph = load_graph()


# --- Session setup ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = str(uuid.uuid4())

if "meta" not in st.session_state:
    st.session_state.meta = {}

session_id = st.session_state.current_session


# --- Sidebar ---
st.sidebar.title("📁 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents", type=["pdf", "txt", "csv"], accept_multiple_files=True
)

if uploaded_files:
    process_uploaded_files(uploaded_files, session_id)
    st.sidebar.success("Documents processed successfully.")


# --- Chat UI ---
st.title("Q&A Chatbot with File Upload + Tools")

if session_id not in st.session_state.sessions:
    st.session_state.sessions[session_id] = []

# Display history
for msg in st.session_state.sessions[session_id]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# --- Chat input box ---
user_msg = st.chat_input("Ask something...")

if user_msg:
    # store message
    st.session_state.sessions[session_id].append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # inject session_id for document tool
            response = graph.invoke(
                {"messages": user_msg, "session_id": session_id},
                config={"configurable": {"thread_id": session_id}},
            )

            messages = response["messages"]
            last = messages[-1]

            # TOOL CALL
            if hasattr(last, "tool_calls") and last.tool_calls:
                call = last.tool_calls[0]
                st.info(f"🛠 Tool: `{call['name']}`")
                st.code(call["args"])

                # show tool outputs
                for m in messages:
                    if m.__class__.__name__ == "ToolMessage":
                        st.code(m.content)

            # NORMAL RESPONSE
            bot = last.content or "(no output)"
            st.write(bot)

    st.session_state.sessions[session_id].append(
        {"role": "assistant", "content": bot}
    )

    st.rerun()
