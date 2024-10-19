import boto3
from dotenv import find_dotenv, load_dotenv
from langchain import hub
from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser

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


prompt = hub.pull("rlm/rag-prompt",  api_url="https://api.smith.langchain.com", api_key="langsmith api key")
# "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"
# print(prompt)
generation_chain = prompt | llm | StrOutputParser()