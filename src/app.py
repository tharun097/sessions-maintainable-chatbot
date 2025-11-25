import os
import uuid
import streamlit as st

from LLM.groqllm import GroqLLM
from Graph_Workflow.graph import Graphbuilder
from Tools.tools import process_uploaded_files

# ----------------------------------------------------
# INIT KEYS
# ----------------------------------------------------
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT_NAME"] = "Q&A Chatbot"


# ----------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Q&A Chatbot", page_icon="💬")

groq_key = st.secrets["GROQ_API_KEY"]
if not groq_key:
    st.error("❌ GROQ_API_KEY missing in secrets!")
    st.stop()


# ----------------------------------------------------
# LOAD LLM + GRAPH
# ----------------------------------------------------
@st.cache_resource
def load_graph():
    model = GroqLLM(llm="llama-3.1-8b-instant", api_key=groq_key).load_llm()
    return Graphbuilder(model).get_chatbot_graph()


graph = load_graph()


# ----------------------------------------------------
# SESSION INIT
# ----------------------------------------------------
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


# ----------------------------------------------------
# SIDEBAR (Sessions + File Upload)
# ----------------------------------------------------
st.sidebar.title("💬 Chat Sessions")
st.sidebar.button("➕ New Chat", on_click=new_chat)

# show buttons for each session
for sid in list(st.session_state.sessions.keys()):
    if st.sidebar.button(f"Chat {sid[:6]}..."):
        st.session_state.current_session = sid

session_id = st.session_state.current_session

# ensure session initialized
if session_id not in st.session_state.sessions:
    st.session_state.sessions[session_id] = []
if session_id not in st.session_state.meta:
    st.session_state.meta[session_id] = {"welcome_shown": False}

# file uploader
uploaded_files = st.sidebar.file_uploader(
    "📄 Upload documents (PDF, CSV, TXT)",
    type=["pdf", "csv", "txt"],
    accept_multiple_files=True,
)

# process uploaded files
if uploaded_files:
    process_uploaded_files(uploaded_files, session_id)
    st.sidebar.success("📚 Documents processed successfully!")


# ----------------------------------------------------
# PAGE HEADER
# ----------------------------------------------------
st.title("Q&A Chatbot with Memory, Tools & Document Analysis")


# ----------------------------------------------------
# WELCOME MESSAGE
# ----------------------------------------------------
if not st.session_state.meta[session_id]["welcome_shown"]:
    welcome = (
        "👋 **Welcome!**\n\n"
        "You can ask questions, search knowledge bases, or upload documents and chat with them!"
    )
    st.session_state.sessions[session_id].append({"role": "assistant", "content": welcome})
    st.session_state.meta[session_id]["welcome_shown"] = True


# ----------------------------------------------------
# SHOW MESSAGE HISTORY
# ----------------------------------------------------
for msg in st.session_state.sessions[session_id]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ----------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------
user_query = st.chat_input("Type your message...")

if user_query:

    # save + show user message
    st.session_state.sessions[session_id].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = graph.invoke(
                {"messages": user_query, "session_id": session_id},
                config={"configurable": {"thread_id": session_id}},
            )

            messages = response["messages"]
            last_msg = messages[-1]

            # TOOL CALL HANDLING
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_call = last_msg.tool_calls[0]

                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})

                st.write(f"🔧 **Tool Call:** `{tool_name}`")
                st.code(tool_args)

                # print tool output
                for m in messages:
                    if m.__class__.__name__ == "ToolMessage":
                        if getattr(m, "status", None) == "error":
                            st.error(f"❌ Tool `{tool_name}` failed")
                            st.code(m.content)
                        else:
                            st.info(f"🛠 Tool Output:")
                            st.code(m.content)

                        # save tool output
                        st.session_state.sessions[session_id].append(
                            {"role": "assistant", "content": f"Tool Output:\n```\n{m.content}\n```"}
                        )

            # assistant reply
            bot_response = last_msg.content or "(No response)"
            st.write(bot_response)

    # save response
    st.session_state.sessions[session_id].append({"role": "assistant", "content": bot_response})

    st.rerun()
