# from langchain_core.output_parsers.json import SimpleJsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_aws import ChatBedrockConverse
# import boto3
# import asyncio

# bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")
# model = ChatBedrockConverse(
#     client=bedrock_runtime,
#     model="anthropic.claude-3-5-sonnet-20240620-v1:0",
#     # max_tokens=200,
#     temperature=0,
#     top_p=0.75,
#     stop_sequences=["\n\nHuman"],
# )

# prompt = PromptTemplate.from_template(
#     'In JSON format, give me a list of {topic} and their '
#     'corresponding names in French, Spanish and in a '
#     'Cat Language.'
# )

# chain = prompt | model

# async def iterate_async_gen():
#     text = ""
#     async for chunk in chain.astream({'topic': 'colors'}):
#         print('-')  # noqa: T201
#         # res = chunk.content
#         if res := chunk.content:
#             try:
#                 text += res[0]['text']
#                 print(text, end='', flush=True)  # noqa: T201
#             except KeyError as ex:
#                 print(ex, end='', flush=True)

# asyncio.run(iterate_async_gen())

import sys
import time

def stream_output(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)  # Write one character at a time
        sys.stdout.flush()      # Force output to be written immediately
        time.sleep(delay)       # Delay between characters for streaming effect

# Example usage
text = "Hello, this is a simulated streaming output.\n"
stream_output(text)