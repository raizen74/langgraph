import boto3
from langchain.agents import create_react_agent
from langchain_aws import ChatBedrockConverse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents.react.agent import create_react_agent

react_prompt = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], input_types={}, partial_variables={}, metadata={'lc_hub_owner': 'hwchase17', 'lc_hub_repo': 'react', 'lc_hub_commit_hash': 'd15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d'}, template='Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}')

@tool
def triple(num: float) -> float:
    """

    :param num: a number to triple
    :return: the number tripled -> multiplied by 3
    """
    return 3 * float(num)

tools = [TavilySearchResults(max_results=1), triple]

bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")
llm = ChatBedrockConverse(
    client=bedrock_runtime,
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # max_tokens=200,
    temperature=0,
    top_p=0.75,
    stop_sequences=["\n\nHuman"],
)

react_agent_runnable = create_react_agent(llm, tools, react_prompt)  # returns a runnable