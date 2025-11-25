import os 
import uuid
import streamlit as st

from LLM.groqllm import GroqLLM
from Graph_Workflow.graph import Graphbuilder
from Tools.tools import process_uploaded_files   # <-- ADDED


# ------------------ ENV ------------------
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT_NAME"] = "Q&A Chatbot"

api_key = st.secrets["GROQ_API_KEY"]
if not api_key:
    st.error("❌ GROQ_API_KEY missing in st.secrets")
    st.stop()


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Q&A Chatbot", page_icon="💬")


# ------------------ LOAD GRAPH ------------------
@st.cache_resource
def load_graph():
    model = GroqLLM(llm="llama-3.1-8b-instant", api_key=api_key).load_llm()
    return Graphbuilder(model).get_chatbot_graph()

graph = load_graph()


# ------------------ SESSION INIT ------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = str(uuid.uuid4())

if "meta" not in st.session_state:
    st.session_state.meta = {}


def new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = []
    st.session_state.meta[new_id] = {"welcome_shown": False}
    st.session_state.current_session = new_id


# ------------------ SIDEBAR ------------------
st.sidebar.title("💬 Chat Sessions")
st.sidebar.button("➕ New Chat", on_click=new_chat)

# LIST chat sessions
for sid in list(st.session_state.sessions.keys()):
    if st.sidebar.button(f"Chat {sid[:6]}..."):
        st.session_state.current_session = sid

session_id = st.session_state.current_session

# Ensure session exists
if session_id not in st.session_state.sessions:
    st.session_state.sessions[session_id] = []

if session_id not in st.session_state.meta:
    st.session_state.meta[session_id] = {"welcome_shown": False}


# ------------------ FILE UPLOAD FEATURE ------------------
st.sidebar.subheader("📁 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, CSV, TXT",
    accept_multiple_files=True,
    type=["pdf", "csv", "txt"]
)

if uploaded_files:
    try:
        process_uploaded_files(uploaded_files, session_id)
        st.sidebar.success("📄 Documents uploaded & indexed!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to process file: {e}")


# ------------------ PAGE TITLE ------------------
st.title("Q&A Chatbot with Memory & Tools + File Upload")


# ------------------ WELCOME MESSAGE ------------------
if not st.session_state.meta[session_id]["welcome_shown"]:
    welcome_msg = (
        "👋 **Welcome to your new chat!**\n\n"
        "You can ask anything, and I may use tools, databases, or your **uploaded files**.\n\n"
        "Try: *“Summarize my uploaded document”*."
    )
    st.session_state.sessions[session_id].append({
        "role": "assistant",
        "content": welcome_msg
    })
    st.session_state.meta[session_id]["welcome_shown"] = True


# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state.sessions[session_id]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ------------------ CHAT INPUT ------------------
user_query = st.chat_input("Ask anything...")

if user_query:

    # Save user msg
    st.session_state.sessions[session_id].append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.write(user_query)

    # BOT RESPONSE
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = graph.invoke(
                {"messages": user_query, "session_id": session_id},
                config={"configurable": {"thread_id": session_id}}
            )

            messages = response["messages"]
            last_msg = messages[-1]

            # ------------------ TOOL CALL DETECTION ------------------
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_call = last_msg.tool_calls[0]
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})

                st.write(f"🔧 **Tool Call:** `{tool_name}`")
                st.code(tool_args)

                # TOOL OUTPUT / TOOL ERROR
                for m in messages:
                    if m.__class__.__name__ == "ToolMessage":
                        if getattr(m, "status", None) == "error":
                            st.error("❌ Tool Execution Failed")
                            st.code(m.content)
                        else:
                            st.info("🛠️ Tool Output:")
                            st.code(m.content)

            # ------------------ NORMAL RESPONSE ------------------
            bot_response = last_msg.content or "(No response available)"
            st.write(bot_response)

    # Save bot msg
    st.session_state.sessions[session_id].append({
        "role": "assistant",
        "content": bot_response
    })

    st.rerun()
