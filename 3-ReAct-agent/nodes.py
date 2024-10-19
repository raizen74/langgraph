from langchain_core.agents import AgentAction
from langgraph.prebuilt.tool_executor import ToolExecutor
from react import react_agent_runnable, tools
from state import AgentState
from pprint import pprint

def run_agent_reasoning_engine(state: AgentState): # reasoning node
    print("AGENT_REASON\n")
    print(f"{state = }\n")
    agent_outcome = react_agent_runnable.invoke(state)
    # print(f"{agent_outcome = }\n")
    pprint(vars(agent_outcome))
    return {"agent_outcome": agent_outcome} # update "agent_outcome" key from state


tool_executor = ToolExecutor(tools)

def execute_tools(state: AgentState):
    print("ACT\n")
    print(f"{state = }\n")
    agent_action: AgentAction | None = state["agent_outcome"] # this is what we returned on the previous node
    tool_name: str = agent_action.tool
    tool_input = agent_action.tool_input
    print(f"{tool_name}.invoke(input={tool_input}\n")

    output: list = tool_executor.invoke(agent_action)
    print(f"{output = }\n")
    # lo que retornes es el que apareix en el rendered output de langsmith
    return {"intermediate_steps": [(agent_action, str(output))]} # update "intermediate_steps" key from state, the tuple will be appended to intermediate_steps
#   "intermediate_steps": [
#     [
#       {
#         "tool": "tavily_search_results_json",
#         "tool_input": "current weather in San Francisco\n",
#         "log": "To answer this question, I'll need to follow these steps:\n1. Find out the current temperature in San Francisco\n2. Triple that temperature\n\nLet's start by searching for the current weather in San Francisco.\n\nAction: tavily_search_results_json\nAction Input: current weather in San Francisco\n",
#         "type": "AgentAction"
#       },
#       "[{'url': 'https://www.weatherapi.com/', 'content': \"{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1729164025, 'localtime': '2024-10-17 04:20'}, 'current': {'last_updated_epoch': 1729163700, 'last_updated': '2024-10-17 04:15', 'temp_c': 13.3, 'temp_f': 55.9, 'is_day': 0, 'condition': {'text': 'Clear', 'icon': '//cdn.weatherapi.com/weather/64x64/night/113.png', 'code': 1000}, 'wind_mph': 6.7, 'wind_kph': 10.8, 'wind_degree': 279, 'wind_dir': 'W', 'pressure_mb': 1013.0, 'pressure_in': 29.9, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 87, 'cloud': 0, 'feelslike_c': 12.5, 'feelslike_f': 54.4, 'windchill_c': 10.7, 'windchill_f': 51.3, 'heatindex_c': 12.0, 'heatindex_f': 53.6, 'dewpoint_c': 11.6, 'dewpoint_f': 52.9, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 9.7, 'gust_kph': 15.6}}\"}]"
#     ],
#     [
#       {
#         "tool": "triple",
#         "tool_input": "55.9\n",
        # log -> Output of the LLM 
#         "log": "Based on the search results, I have found the current temperature in San Francisco. Now, I'll extract the temperature and then triple it.\n\nThe current temperature in San Francisco is 55.9°F (13.3°C).\n\nLet's triple this temperature in Fahrenheit:\n\nAction: triple\nAction Input: 55.9\n",
#         "type": "AgentAction"
#       } -> agent_action,
#       "167.7" -> str(output),
#     ]
#   ]