"""This module generates the text from the retrieved documents with the generation chain"""
from typing import Any, Dict
from pprint import pprint 
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE NODE---")
    print("state:")
    pprint(state)    
    question = state["question"]
    documents = state["documents"]

    generation: str = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
