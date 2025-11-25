from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from State.state import State
from Nodes.chatbot_node import ChatbotNode
from Tools.tools import get_tools


class Graphbuilder:
    def __init__(self, model):
        self.llm = model
        self.memory = MemorySaver()
        self.graph = StateGraph(State)

    def get_chatbot_graph(self):
        tools = get_tools()
        tool_node = ToolNode(tools)

        chatbot = ChatbotNode(self.llm).chatbot(tools)

        self.graph.add_node("chatbot", chatbot)
        self.graph.add_node("tools", tool_node)

        self.graph.add_edge(START, "chatbot")
        self.graph.add_conditional_edges("chatbot", tools_condition)
        self.graph.add_edge("tools", "chatbot")
        self.graph.add_edge("chatbot", END)

        return self.graph.compile(checkpointer=self.memory)
