from typing_extensions import TypedDict,Annotated,List

from langgraph.graph import add_messages

class State(TypedDict):
    messages:Annotated[List,add_messages]