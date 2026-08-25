from langchain_core.tools import tool
from miniswe_langgraph.environment import Environment
import logging

logger = logging.getLogger(__name__)

env = Environment()

@tool
def shell(command: str) -> str:
    """Execute a shell command in the agent environment."""
    logger.info("============ TOOL ===========")
    logger.info(
        "COMMAND: \n%s",
        command,
    )
    result = env.execute(command=command)

    logger.info(
        "RESULT:\n%s",
        result,
    )

    return result
