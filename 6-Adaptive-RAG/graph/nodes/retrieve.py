from typing import Any, Dict
from pprint import pprint
from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("----RETRIEVE NODE---")
    print("state:")
    pprint(state)
    question = state["question"]

    documents = retriever.invoke(question)
    # update the field documents of the state, and also the original question for debugging purposes
    return {"documents": documents, "question": question}  # write state, this will be passed to the next node -> grade_documents
