from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
load_dotenv()

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]  # list of strings that we are going to append to after each node execution
    which: str # which node we want to execute in our conditional edge

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
builder.add_node("e", ReturnNodeValue("I'm E"))


def route_bc_or_cd(state: State) -> list[str]:
    """Conditional edge: Routes execution to 2 downstream nodes"""
    if state['which'] == "cd":
        return ["c", "d"]
    return ["b", "c"]

intermediates = ["b", "c", "d"]
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    path_map=intermediates, # can be a mapping or a list, if list, langgraph uses it to draw the graph (very important)
)

for node in intermediates:
    builder.add_edge(node, end_key="e")
builder.add_edge('e', END)
graph: CompiledStateGraph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="async_v3.png")

if __name__ == "__main__":
    print("Hello Async_v3 Graph")
    print(graph.invoke(input={"aggregate": [], "which": "cd"}, config={"configurable":{"thread_id": "foo"}}))
