from langchain_core.tools import tool
from miniswe_langgraph.environment import Environment
import logging

logger = logging.getLogger(__name__)

env = Environment()

# Sentinel used by `track_tools` to extract the returncode from the tool
# message content. Kept here so the parser and the producer stay in sync.
TOOL_OUTPUT_HEADER = "Returncode: {returncode}\nOutput:\n{output}"

# Human-readable description injected into the system prompt. Kept in sync
# with the `@tool` definitions below so the prompt reflects what the model
# actually has access to.
TOOLS_DESCRIPTION = """- shell(command: str) -> str
    Execute a shell command in the agent environment and return its combined
    stdout/stderr along with the exit code. Use this for every action:
    listing files, reading files, editing files, running tests, etc."""


@tool
def shell(command: str) -> str:
    """Execute a shell command in the agent environment."""
    logger.info("============ TOOL ===========")
    logger.info(
        "COMMAND: \n%s",
        command,
    )
    returncode, output = env.execute(command=command)

    logger.info(
        "RESULT (returncode=%s):\n%s",
        returncode,
        output,
    )

    return TOOL_OUTPUT_HEADER.format(returncode=returncode, output=output)
