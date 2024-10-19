from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from pprint import pprint
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory" # Sample question
    docs = retriever.invoke(question) 
    doc_txt = docs[0].page_content # retrieve the first document that we find (highest score)

    res = retrieval_grader.invoke({"question": question, "document": doc_txt})
    # extract the field from the pydantic model
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory" # Sample question
    docs = retriever.invoke(question) 
    doc_txt = docs[0].page_content # retrieve the first document that we find (highest score)

    res = retrieval_grader.invoke({"question": "how to make pizza", "document": doc_txt})
    # extract the field from the pydantic model
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    print(f"\n{len(docs) = }\n")  # retrieves 4 documents by default
    generation: str = generation_chain.invoke({"context": docs, "question": question})
    print(f"{type(generation) = }\n")
    pprint(generation)
