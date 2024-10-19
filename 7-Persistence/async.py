from dotenv import load_dotenv
load_dotenv()

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]  # list of strings that we are going to append to after each node execution

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State):
        """Making the instance callable by overriding this method"""
        import time
        time.sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_edge(START, "a")
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
# Node A fans out to nodes B and C
builder.add_edge('a', "b")
builder.add_edge('a', "c")
builder.add_edge('b', "d")
builder.add_edge('c', "d")
builder.add_edge('d', END)  # Node D only runs after B
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="async.png")


if __name__ == "__main__":
    print("Hello Async Graph")
    graph.invoke(input={"aggregate": []}, config={"configurable":{"thread_id": "foo"}})
