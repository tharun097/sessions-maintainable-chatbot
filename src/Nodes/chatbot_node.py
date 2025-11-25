from State.state import State
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ChatbotNode:
    def __init__(self, llm):
        self.llm = llm

    def chatbot(self, tools):
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):

            messages = state["messages"]

            # ------------------------------------------
            # 1. Prevent infinite recursion
            # If last message is ToolMessage → stop here
            # ------------------------------------------
            if isinstance(messages[-1], ToolMessage):
                return {"messages": []}  # no new LLM call

            # ------------------------------------------
            # 2. Normal LLM call
            # ------------------------------------------
            ai_response = llm_with_tools.invoke(messages)
            return {"messages": ai_response}

        return chatbot_node
