from dotenv import load_dotenv

load_dotenv()

import sys
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage
from miniswe_langgraph.graph import graph
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

    input_state = {
        "messages": [
            HumanMessage(content=prompt)
        ],
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
