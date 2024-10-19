import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")
llm = ChatBedrockConverse(
    client=bedrock_runtime,
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=200,
    temperature=0,
    top_p=0.75,
    stop_sequences=["\n\nHuman"]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            " Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
