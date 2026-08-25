from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from miniswe_langgraph.state import State
from miniswe_langgraph.agent import agent
from miniswe_langgraph.tools import shell


def track_tools(state: State):
    trajectory = state["trajectory"]
    tool_node = ToolNode([shell])
    result = tool_node.invoke(state)
    messages = result.get("messages", [])
    for message in messages:
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
        "trajectory": trajectory,
    }


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", track_tools)
builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")


def route_after_agent(state: State) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "__end__": "__end__",
    },
)
graph = builder.compile()
