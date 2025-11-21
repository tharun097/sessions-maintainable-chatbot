from Tools.tools import nasa_tool, api_tool, satellite_data_tool
from State.state import State

class ChatbotNode:
    def __init__(self, llm):
        self.llm = llm
    def chatbot(self,tools):
        """Receives graph state and returns the LLM response."""
        # user_message = state["messages"]
        llm_with_tools = self.llm.bind_tools(tools)
        def chatbot_node(state:State)->dict:
            return {"messages":llm_with_tools.invoke(state['messages'])}
        return chatbot_node
