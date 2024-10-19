import json
from typing import List
from collections import defaultdict
from dotenv import find_dotenv, load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, AnyMessage
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from chains_v2 import parser
from schemas import AnswerQuestion, Reflection

load_dotenv(find_dotenv())

# TavilySearchAPIWrapper().results("AI-powered SOC startups funding")
tavily_wrapper = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(max_results=5) # maximum results that we want from our search engine
tool_executor = ToolExecutor([tavily_tool]) # ToolExecutor has a batch method to run queries in parallel

# ToolMessage represents the result of a tool call
def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    print('Executing tool')
    for mes in state:
        mes.pretty_print()
    print("{:-<10}\n".format(""))
    tool_invocation: AIMessage = state[-1] # We assume that before executing the tools node, we always receive an AI message with the function calling response
    parsed_tool_calls: List[dict] = parser.invoke(tool_invocation)  # transforms the tool output of the LLM from JSON into a Python dict
    
    ids = []
    tool_invocations = []
    outputs = []

    for parsed_call in parsed_tool_calls: # iterate over the tools
        for query in parsed_call['args']['search_queries']: # iterate over the search queries
            # print(query)
            tool_invocations.append(ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query,
            ))
            outputs.append(tavily_wrapper.results(query)) # each search returns a List[dict] with 5 dicts, url and content keys
            ids.append(parsed_call["id"])
            # break
        # break
    #map each tavily search output to its corresponding id and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output # dict[id, dict[tool_input, [output]]]

    # Convert the mapped outputs to ToolMessage objects
    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_))
    
    return tool_messages  # Returns a list of ToolMessage Objects

if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="tooluse_FmpwVHNSSEy5bz36BheO9g"
    )
    
    # we pass a dummy state
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.model_dump(),
                        "id": "tooluse_FmpwVHNSSEy5bz36BheO9g"
                    }
                ]
            )
        ]
    )
    print(raw_res)
