from dotenv import load_dotenv

load_dotenv()

import sys
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from miniswe_langgraph.graph import graph
from miniswe_langgraph.prompts import render_prompt
from miniswe_langgraph.tools import TOOLS_DESCRIPTION
from miniswe_langgraph.trajectory import Trajectory

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    handlers=[
        logging.FileHandler("logs/miniswe.log"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
)

def get_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    return input("What do you want to do?\n")

def main():
    prompt = get_prompt()
    trajectory = Trajectory()

    trajectory.add(
        "user",
        {
            "content": prompt,
        },
    )

    # Inject the system + instance prompts. The system prompt is rendered once
    # (it depends only on the static tools description), the instance prompt is
    # rendered with the current task.
    system_message = SystemMessage(
        content=render_prompt(
            "system.j2",
            tools=TOOLS_DESCRIPTION,
        ),
    )
    instance_message = HumanMessage(
        content=render_prompt(
            "instance.j2",
            task=prompt,
        ),
    )

    input_state = {
        "messages": [system_message, instance_message],
        "trajectory": trajectory,
    }
    for event in graph.stream(
        input_state,
        stream_mode="updates",
    ):
        trajectory.add(
            "graph_update",
            event,
        )

    trajectory.save()
