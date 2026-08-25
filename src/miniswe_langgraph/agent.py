from miniswe_langgraph.model import llm
from miniswe_langgraph.state import State
from miniswe_langgraph.tools import shell
import logging

logger = logging.getLogger(__name__)

llm_with_tools = llm.bind_tools([shell])

def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    trajectory = state["trajectory"]
    trajectory.add(
        "assistant",
        {
            "content": response.content,
            "tool_calls": response.tool_calls,
        },
    )

    logger.info("=========== AI ===========")
    if response.content:
        logger.info("CONTENT:\n%s", response.content)

    if response.tool_calls:
        for call in response.tool_calls:
            logger.info(
                "TOOL CALL: %s\nARGS: %s",
                call["name"],
                call["args"],
            )

    return {
        "messages": [response],
        "trajectory": trajectory,
    }
