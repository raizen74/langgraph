import operator
from typing import TypedDict, Annotated, Union

from langchain_core.agents import AgentAction, AgentFinish # AgentAction contains the function we need to run with its input and with the proof of why did we choose it
# AgentFinish contains the final result

class AgentState(TypedDict):
    input: str  # Users query
    agent_outcome: Union[AgentAction, AgentFinish, None] # Initially None
    #intermediate steps is plugged in the agent_scratchpad variable of the react prompt
    # list of tuples, the first element is the AgentAction and the second the output of that action. operator.add continuously adds to this list
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
