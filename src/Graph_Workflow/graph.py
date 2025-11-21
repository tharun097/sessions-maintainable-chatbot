from langgraph.graph import StateGraph,START,END
from State.state import State
from langgraph.prebuilt import ToolNode,tools_condition
from Tools.tools import get_tools,create_tool_node
from Nodes.chatbot_node import ChatbotNode
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st 

class Graphbuilder:
    def __init__(self,model):
        self.llm = model
        self.memory = MemorySaver()
        self.graph_builder = StateGraph(State)
    
    def get_chatbot_graph(self):
        """Takes input from user as a message and returns the accurate message nothing but acts like a assistant to user basis on workflow defined"""
        tools = get_tools()
        tool_node = create_tool_node(tools)
        self.basic_chatbot_node=ChatbotNode(self.llm)
        chatbot_node = self.basic_chatbot_node.chatbot(tools=tools)
        self.graph_builder.add_node("chatbot",chatbot_node)
        self.graph_builder.add_node("tools",tool_node)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        self.graph_builder.add_edge("tools","chatbot")        
        self.graph_builder.add_edge("chatbot",END)
        # Compile graph WITH MEMORY
        return self.graph_builder.compile(checkpointer=self.memory)

    
