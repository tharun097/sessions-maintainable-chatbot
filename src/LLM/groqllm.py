from langchain_groq import ChatGroq
class GroqLLM:
    def __init__(self,llm,api_key):
        self.llm = llm
        self.api_key = api_key
    def load_llm(self):
        model = ChatGroq(model=self.llm,api_key=self.api_key)
        return model