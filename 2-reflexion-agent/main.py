from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains_v2 import (
    revisor,
    first_responder,
)  # langchain chains that will run under langgraph nodes
from tool_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = 1
builder = (
    MessageGraph()
)  # builtin graph with the state equal to a bunch of BaseMessages or ToolMessages
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)  # this chain receives the output message of the previous node into the MessagesPlaceholder
builder.add_edge(start_key="draft", end_key="execute_tools")
builder.add_edge(start_key="execute_tools", end_key="revise")


def event_loop(state: List[BaseMessage]):
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END  # END references the ending node
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello Reflexion")
    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )  # injectem el MessagesPlaceholder en el first_responder
    print("Finished Reflexion")
    print("{:-<10}\n".format(""))
    print(res[-1].tool_calls[0]['args']['answer'])
