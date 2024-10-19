import sys
import io

# Set the encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from typing import List, Sequence
from dotenv import find_dotenv, load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage

from chains import generate_chain, reflect_chain
from langgraph.graph import END, MessageGraph

load_dotenv(find_dotenv())
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    print("----GENERATION NODE----")
    for m in state:
        m.pretty_print()  # En la primera iteració li arriba només [HumanMessage] del tweet
    return generate_chain.invoke({"messages": state})

def reflection_node(state: Sequence[BaseMessage]) -> List[HumanMessage]:
    print("----REFLECTION NODE----")
    for m in state:
        m.pretty_print()  # En la primera iteració li arriba HumanMessage del tweet i AI Message de la generació
    res = reflect_chain.invoke({"messages": state})
    return [
        HumanMessage(content=res.content)
    ]  # role of a human, tricking the LLM to think that a human generate this message

builder = MessageGraph()  # Generated messages are appended to the state, and state is plugged into MessagesPlaceholder of each chain. 
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# Conditional edge
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT  # Go to REFLECT node to use an LLM to give us feedback and revise the tweet

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

# Export the graph
# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Visualize the graph
# print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(
        content="""Make this tweet better:"
                          @LangChainAI
                          - newly Tool Calling feature is seriously underrated.
                          
                          After a long wait, it's here- making the implementation of agents across different models with function calling - super easy.
                          
                          Made a video covering their newest blog post
                          
                          """
    )
    response = graph.invoke(inputs)
