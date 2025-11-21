from dotenv import load_dotenv
import os 
from LLM.groqllm import GroqLLM
import streamlit as st
import uuid
from Graph_Workflow.graph import Graphbuilder

# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")
api_key = st.secrets["GROQ_API_KEY"]
if not api_key:
    st.error("❌ GROQ_API_KEY missing in st secrets")
    st.stop()

st.set_page_config(page_title="Q&A Chatbot", page_icon="💬")


# -------------------------------------------------------------------
# Load Model + Graph Once
# -------------------------------------------------------------------
@st.cache_resource
def load_graph():
    model = GroqLLM(llm="llama-3.1-8b-instant", api_key=api_key).load_llm()
    return Graphbuilder(model).get_chatbot_graph()

graph = load_graph()


# -------------------------------------------------------------------
# Session Init
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("💬 Chat Sessions")
st.sidebar.button("➕ New Chat", on_click=new_chat)

for sid in list(st.session_state.sessions.keys()):
    if st.sidebar.button(f"Chat {sid[:6]}..."):
        st.session_state.current_session = sid

session_id = st.session_state.current_session

if session_id not in st.session_state.sessions:
    st.session_state.sessions[session_id] = []

if session_id not in st.session_state.meta:
    st.session_state.meta[session_id] = {"welcome_shown": False}


# -------------------------------------------------------------------
# Page Title
# -------------------------------------------------------------------
st.title("Q&A Chatbot with Memory & Smart Tool Handling")


# -------------------------------------------------------------------
# Dynamic Welcome Message
# -------------------------------------------------------------------
if not st.session_state.meta[session_id]["welcome_shown"]:
    welcome_msg = (
        "👋 **Welcome to your new chat!**\n\n"
        "Ask me anything — I can use tools, fetch external data, or answer normally."
    )
    st.session_state.sessions[session_id].append({
        "role": "assistant",
        "content": welcome_msg
    })
    st.session_state.meta[session_id]["welcome_shown"] = True


# -------------------------------------------------------------------
# Display Chat History
# -------------------------------------------------------------------
for msg in st.session_state.sessions[session_id]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------------------------------------------------
# Chat Input
# -------------------------------------------------------------------
user_query = st.chat_input("Ask anything...")

if user_query:

    # Save user message
    st.session_state.sessions[session_id].append({
        "role": "user",
        "content": user_query
    })

    # Display user message instantly
    with st.chat_message("user"):
        st.write(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = graph.invoke(
                {"messages": user_query},
                config={"configurable": {"thread_id": session_id}}
            )

            messages = response["messages"]
            last_msg = messages[-1]

            # -------------------------------------------------------------------
            # 1️⃣ TOOL CALL DETECTED
            # -------------------------------------------------------------------
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_call = last_msg.tool_calls[0]
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})

                st.write(f"🔧 **Tool Call Detected:** `{tool_name}`")
                st.write(f"📦 **Arguments:** `{tool_args}`")
                st.write("⏳ Running tool...")

                # -------------------------------------------------------------------
                # 2️⃣ TOOL OUTPUT (SUCCESS CASE)
                # -------------------------------------------------------------------
                for m in messages:
                    if m.__class__.__name__ == "ToolMessage" and getattr(m, "status", None) != "error":
                        st.info("🛠️ **Tool Output:**")
                        st.code(m.content)

                        # Save tool output to history
                        st.session_state.sessions[session_id].append({
                            "role": "assistant",
                            "content": f"🛠️ Tool Output:\n\n```\n{m.content}\n```"
                        })
                        break

                # -------------------------------------------------------------------
                # 3️⃣ TOOL ERROR (FAILURE CASE)
                # -------------------------------------------------------------------
                for m in messages:
                    if m.__class__.__name__ == "ToolMessage" and getattr(m, "status", None) == "error":
                        st.error("❌ **Tool Execution Failed**")
                        st.warning(f"**Tool:** `{m.name}`")
                        st.code(m.content)

                        st.session_state.sessions[session_id].append({
                            "role": "assistant",
                            "content": f"❌ Tool `{tool_name}` failed:\n```\n{m.content}\n```"
                        })
                        st.rerun()


            # -------------------------------------------------------------------
            # 4️⃣ NORMAL ASSISTANT RESPONSE
            # -------------------------------------------------------------------
            bot_response = last_msg.content or "(No response available)"
            st.write(bot_response)

    # Save bot message
    st.session_state.sessions[session_id].append({
        "role": "assistant",
        "content": bot_response
    })

    st.rerun()
