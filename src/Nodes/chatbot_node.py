from State.state import State


class ChatbotNode:
    def __init__(self, llm):
        self.llm = llm

    def chatbot(self, tools):
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            # LLM should ONLY receive messages list
            return {
                "messages": llm_with_tools.invoke(state["messages"])
            }

        return chatbot_node
