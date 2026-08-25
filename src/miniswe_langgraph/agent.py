from langchain_core.messages import HumanMessage
from miniswe_langgraph.model import llm
from miniswe_langgraph.prompts import (
    FORMAT_ERROR_MARKER,
    MAX_FORMAT_ERROR_RETRIES,
    render_prompt,
)
from miniswe_langgraph.state import State
from miniswe_langgraph.tools import shell
import logging

logger = logging.getLogger(__name__)

llm_with_tools = llm.bind_tools([shell])


def _looks_like_format_error(response) -> bool:
    """Heuristic: the model produced no tool call but emitted content that
    looks like it tried to invoke one anyway (e.g., a markdown code block or
    a `shell(...)` token). We treat that as a parse failure rather than as
    a final answer."""
    if getattr(response, "tool_calls", None):
        return False
    content = (response.content or "").strip()
    if not content:
        return False
    lowered = content.lower()
    markers = (
        "```",          # code fences
        "shell(",
        "shell ",
        "tool_call",
        "<tool",
        "<function",
    )
    return any(m in lowered for m in markers)


def _count_format_errors(messages) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, HumanMessage)
        and m.content.startswith(FORMAT_ERROR_MARKER)
    )


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

    extra_messages = []
    if _looks_like_format_error(response):
        already = _count_format_errors(state["messages"])
        if already < MAX_FORMAT_ERROR_RETRIES:
            rendered = render_prompt(
                "format_error.j2",
                error=(response.content or "")[:2000],
            )
            extra_messages.append(
                HumanMessage(content=f"{FORMAT_ERROR_MARKER}\n{rendered}")
            )
            logger.warning(
                "Format error detected (attempt %d/%d) — re-prompting.",
                already + 1,
                MAX_FORMAT_ERROR_RETRIES,
            )
        else:
            logger.warning(
                "Format error detected but retry budget exhausted (%d) — ending.",
                MAX_FORMAT_ERROR_RETRIES,
            )

    return {
        "messages": [response] + extra_messages,
        "trajectory": trajectory,
    }