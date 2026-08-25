import re
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage
from miniswe_langgraph.prompts import (
    FORMAT_ERROR_MARKER,
    render_prompt,
)
from miniswe_langgraph.state import State
from miniswe_langgraph.agent import agent
from miniswe_langgraph.tools import shell


def _parse_tool_output(content: str) -> tuple[int, str] | None:
    """Extract `(returncode, output)` from the sentinel format produced by the
    `shell` tool. Returns None if the content does not match — in which case
    we leave the message content untouched."""
    match = re.match(
        r"^Returncode: (-?\d+)\nOutput:\n(.*)$",
        content,
        re.DOTALL,
    )
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def track_tools(state: State):
    trajectory = state["trajectory"]
    tool_node = ToolNode([shell])
    result = tool_node.invoke(state)
    raw_messages = result.get("messages", [])

    wrapped_messages = []
    for message in raw_messages:
        if isinstance(message, ToolMessage):
            parsed = _parse_tool_output(message.content)
            if parsed is not None:
                returncode, output = parsed
                new_content = render_prompt(
                    "observation.j2",
                    returncode=returncode,
                    output=output,
                )
                message = ToolMessage(
                    content=new_content,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )

        wrapped_messages.append(message)

        trajectory.add(
            "tool",
            {
                "name": getattr(message, "name", None),
                "content": getattr(message, "content", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
            },
        )

    return {
        **result,
        "messages": wrapped_messages,
        "trajectory": trajectory,
    }


def route_after_agent(state: State) -> str:
    last_message = state["messages"][-1]

    # The agent injected a format_error reminder — let the model try again.
    if (
        isinstance(last_message, HumanMessage)
        and last_message.content.startswith(FORMAT_ERROR_MARKER)
    ):
        return "agent"

    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", track_tools)
builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "agent": "agent",
        "tools": "tools",
        "__end__": "__end__",
    },
)
graph = builder.compile()