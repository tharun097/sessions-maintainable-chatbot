from State.state import State


class ChatbotNode:
    def __init__(self, llm):
        self.llm = llm

    def chatbot(self, tools):
        llm_with_tools = self.llm.bind_tools(tools)

        def node(state: State):
            return {"messages": llm_with_tools.invoke({
                "messages": state["messages"],
                "session_id": state["session_id"]
            })}
        return node
