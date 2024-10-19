# %%
from datetime import datetime

# ChatPromptTemplate is used to inject the whole sequence of messages and MessagePlaceholder for new messages
import boto3
from dotenv import find_dotenv, load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from schemas import AnswerQuestion

load_dotenv(find_dotenv())
bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")
llm = ChatBedrockConverse(
    client=bedrock_runtime,
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # max_tokens=200,
    temperature=0,
    top_p=0.75,
    stop_sequences=["\n\nHuman"],
)

parser = JsonOutputToolsParser(return_id=True)  # transforms the response of the LLM into a Python dict
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])  # Creates an pydantic AnswerQuestion object from the answer of the LLM

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """,
        ),
        MessagesPlaceholder(
            variable_name="messages"
        ),  # we pass the messages placeholder of all the history before that (previous searches and critiques)
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.now().isoformat(),
)  # populate known variables


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)  # tool_choice forces the LLM to always use the AnswerQuestion tool

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(
            tools=[AnswerQuestion], tool_choice="AnswerQuestion" # tool_choice forces the LLM to use the tool
        )  # res.content[0]["input"] -> dictionary
        | parser_pydantic  # returns List[AnswerQuestion]
    )  # takes the response and parses as a pydantic object

    res = chain.invoke(
        input={"messages": [human_message]}
    )  # we plug the human message into the MessagesPlaceholder
    print(res)
    # res.content[0]["input"]
# %%
