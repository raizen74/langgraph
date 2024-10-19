from typing import Any, Dict
import sys
import os
from pprint import pprint
# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH NODE---")
    print("state:")
    pprint(state)
    question: str = state["question"]
    documents: list | None = state.get("documents")

    tavily_results = web_search_tool.invoke({"query": question}) # returns a list of dicts with url, content keys

    print(f"{tavily_results = }")

    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    ) # Join the 3 content queries into a string

    web_results = Document(page_content=joined_tavily_result)  # create a new doc with the tavily content
    if documents is not None:
        documents.append(web_results) # append it to the list of docs
    else:  # if documents is empty -> no relevant docs are retrieved
        documents = [web_results]
    return {"documents": documents, "question": question} # update the state of our graph execution


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
