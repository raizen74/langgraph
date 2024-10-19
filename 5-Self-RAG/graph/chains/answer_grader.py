"""Determines whether the answer answers the question or not"""
# %%
import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence

bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")
llm = ChatBedrockConverse(
    client=bedrock_runtime,
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # max_tokens=200,
    temperature=0,
    top_p=0.75,
    stop_sequences=["\n\nHuman"],
)


class GradeAnswer(BaseModel):
    """Binary score for the answer resolution of the question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | llm.with_structured_output(GradeAnswer)

if __name__ == "__main__":
    # llm.with_structured_output(GradeAnswer).invoke('yes')
    generation = "Agent memory refers to the capability of an AI agent to retain and recall information over time. It typically consists of two types: short-term memory, which involves in-context learning, and long-term memory, which uses external vector stores for extended information retention. This memory system allows agents to make decisions based on past experiences and accumulated knowledge."
    print(answer_grader.invoke(
        {"question": "what is agent memory?", "generation": generation}
    ))

# %%
