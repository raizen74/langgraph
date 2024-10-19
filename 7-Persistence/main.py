from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.memory import MemorySaver # checkpoint which stores the state after each node execution
from langgraph.checkpoint.sqlite import SqliteSaver # Used to persist in a SQLite DB
import sqlite3

load_dotenv(find_dotenv())

class State(TypedDict):
    input: str
    user_feedback: str

def step_1(state: State) -> None:
    print("---Step 1---")


def human_feedback(state: State) -> None:
    print("---human_feedback---")

    
def step_3(state: State) -> None:
    print("---Step 3---")
    print(state['user_feedback'])

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

builder.add_edge(START,end_key="step_1")
builder.add_edge("step_1",end_key="human_feedback")
builder.add_edge("human_feedback",end_key="step_3")
builder.add_edge("step_3",end_key=END)

# memory = MemorySaver()  # inmemory save
# REMOTE OR LOCAL DATABASE
conn = sqlite3.connect(database="checkpoints.sqlite", check_same_thread=False)  # run a local db in my filesystem, we try to edit the db from different threads so set check_same_thread=False 
memory = SqliteSaver(conn)
# LOCAL DATABASE
# memory = SqliteSaver.from_conn_string("checkpoints.sqlite")

# Responsible of persisting in memory our state upon each graphs execution, we stop graph execution before human_feedback node
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="persistence-graph.png")

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "2"}}  #thread 1 is to differentiate across different runs in our graph
    initial_input = {"input": "hello world"}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print(graph.get_state(thread).next)  # prints the next node that will run in our graph
    user_input = input("Tell me how you want to update the state: ")

    graph.update_state(thread, values={"user_feedback": user_input}, as_node="human_feedback") # human_feedback node will update as this node would run and write {"user_feedback": user_input} into the state (it simulates the execution of this node)
    print(graph.get_state(thread).next)  # step 3

    print("--State after update--")
    print(graph.get_state(thread))

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
