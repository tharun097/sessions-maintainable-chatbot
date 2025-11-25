from langgraph.graph import StateGraph, START, END
from State.state import State
from Tools.tools import get_tools, document_tool
from langgraph.prebuilt import ToolNode, tools_condition
from Nodes.chatbot_node import ChatbotNode
from langgraph.checkpoint.memory import MemorySaver


class Graphbuilder:
    def __init__(self, model):
        self.llm = model
        self.memory = MemorySaver()
        self.graph_builder = StateGraph(State)

    def get_chatbot_graph(self):
        tools = get_tools()
        tool_node = ToolNode(tools=tools)

        chatbot = ChatbotNode(self.llm).chatbot(tools=tools)

        self.graph_builder.add_node("chatbot", chatbot)
        self.graph_builder.add_node("tools", tool_node)

        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge("chatbot", END)

        return self.graph_builder.compile(checkpointer=self.memory)
