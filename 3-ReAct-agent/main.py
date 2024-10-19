from dotenv import find_dotenv, load_dotenv
from langchain_core.agents import AgentFinish
from langgraph.graph import StateGraph, END
from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

load_dotenv(find_dotenv())

AGENT_REASON = "agent_reason"
ACT = "act"

def should_continue(state:AgentState):
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else: # state["agent_outcome"] == AgentAction
        return ACT

flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, execute_tools)

flow.add_conditional_edges(AGENT_REASON,
                           should_continue)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello ReAct with LangGraph")
    res = app.invoke(
        input={
            "input": "what is the weather in Tenerife? Write it and then Triple it "
        }
    )
    print("{:-<10}\n".format(""))
    print(res["agent_outcome"].return_values["output"])
    # res <class 'langgraph.pregel.io.AddableValuesDict'>, té 3 keys:
    # input: prompt inicial
    # agent_outcome: AgentFinish("return_values", "log", "type")
    # intermediate_steps: List[tuple(AgentAction, output)]  # output es el resultat de la tool
