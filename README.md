# miniswe-langgraph

A LangGraph-based reimplementation of [mini-swe-agent](https://github.com/SWE-bench/mini-swe-agent). It is a minimal agent loop that lets an LLM autonomously complete a task by issuing shell commands inside a sandboxed working directory.

> Inspired by [mini-swe-agent](https://github.com/SWE-bench/mini-swe-agent) (built on top of [SWE-bench](https://github.com/SWE-bench/SWE-bench)). This project ports the same minimal agent pattern to [LangGraph](https://github.com/langchain-ai/langgraph).

## Architecture

The agent is modeled as a LangGraph state machine with two nodes that loop until the model stops emitting tool calls:

```
START -> agent -> tools -> agent -> ... -> __end__
```

- **`agent`** (`src/miniswe_langgraph/agent.py:10`)  calls the LLM with the bound `shell` tool and appends its response (content + tool calls) to the trajectory.
- **`tools`** (`src/miniswe_langgraph/graph.py:8`)  executes any tool calls the LLM emitted (currently only `shell`) and records the tool results in the trajectory.

State (`src/miniswe_langgraph/state.py:4`) carries the LangChain message list and a `Trajectory` object used for logging. Routing logic (`src/miniswe_langgraph/graph.py:35`) decides whether to continue into `tools` or terminate at `__end__` based on whether the last assistant message contains tool calls.

## Components

| File | Purpose |
| --- | --- |
| `src/miniswe_langgraph/__init__.py` | CLI entry point; loads `.env`, configures logging, streams graph events, persists trajectory. |
| `src/miniswe_langgraph/agent.py` | LLM node  invokes the model with the `shell` tool bound. |
| `src/miniswe_langgraph/graph.py` | Builds the LangGraph `StateGraph`, defines the `agent` � `tools` loop. |
| `src/miniswe_langgraph/tools.py` | Defines the `shell` tool that delegates to `Environment.execute`. |
| `src/miniswe_langgraph/environment.py` | `subprocess.run` wrapper with `cwd` pinned to `./workspace`. |
| `src/miniswe_langgraph/model.py` | `ChatOpenAI` client configured from `MODEL` and `BASE_URL` env vars. |
| `src/miniswe_langgraph/state.py` | `State` TypedDict: `messages` (with `add_messages` reducer) + `trajectory`. |
| `src/miniswe_langgraph/trajectory.py` | Records every event (user prompt, assistant, tool, graph update) to `logs/last_run_traj.json`. |
| `workspace/` | Default working directory in which `shell` commands are executed. |
| `docker/Dockerfile` | Placeholder for containerized runs. |
| `logs/` | Runtime logs (`miniswe.log`) and the saved trajectory. |

## Requirements

- Python e 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management (a `uv.lock` is checked in)

## Setup

1. Clone the repository and enter the project directory.
2. Create a `.env` file with the following variables (see `.env` for an example):

   ```env
   MODEL=MiniMax-M3
   BASE_URL=https://api.minimaxi.com/v1
   OPENAI_API_KEY=...
   OPENAI_ADMIN_KEY=...
   ```

   `OPENAI_API_KEY` is also read by `langchain-openai`; either set it explicitly or rely on the same value via the environment.
3. Install dependencies and activate the virtual environment:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

## Usage

The CLI accepts the task prompt either from `stdin` or as positional arguments. Both forms feed the prompt into the graph as a `HumanMessage` and stream events until the agent stops calling tools.

```bash
# Interactive  type the task at the prompt
miniswe-langgraph

# Or pass it directly
miniswe-langgraph "List every file in the workspace"
```

You can also run it as a module:

```bash
python -m miniswe_langgraph "Compile hello.c and run it"
```

### What happens at runtime

1. The prompt is wrapped in a `HumanMessage` and pushed onto `state["messages"]`.
2. The `agent` node calls the LLM. If the response contains `tool_calls`, the `tools` node executes the `shell` command inside `./workspace` and returns stdout/stderr.
3. The loop repeats until the model responds without tool calls, at which point the graph transitions to `__end__`.
4. The complete event stream (user prompt, every assistant turn, every tool call and result, every graph update) is serialized to `logs/last_run_traj.json`.
5. A human-readable log of the same run is appended to `logs/miniswe.log`.

## How it differs from `mini-swe-agent`

- The control flow is implemented with [LangGraph](https://github.com/langchain-ai/langgraph) instead of a hand-rolled loop, so it composes naturally with the rest of the LangChain ecosystem.
- Trajectory capture is built into the graph itself (the `track_tools` node and per-event hooks in `Trajectory`), rather than being a separate concern.
- The shell tool is the only tool available by design  the goal is to keep the surface area minimal, mirroring the original mini-swe-agent philosophy.

## Development notes

- The `shell` tool executes with `shell=True` against `./workspace`. Commands are not sandboxed beyond the working directory, so treat the model as a privileged user.
- `Environment.execute` returns combined `stdout + stderr`; the agent must rely on the textual feedback to detect failures (there is no structured exit code in the result).
- `logs/last_run_traj.json` is overwritten on every run. Move it aside if you want to keep a history.

## Acknowledgements / References

This project is a LangGraph port of the ideas and design from the following upstream projects:

- [SWE-bench/mini-swe-agent](https://github.com/SWE-bench/mini-swe-agent) — the original minimal agent whose architecture we reimplemented.
- [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) — the benchmark and ecosystem that mini-swe-agent was built for.
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — the framework used to express the agent loop as a state graph.
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain) — the `ChatOpenAI` client and `ToolNode` infrastructure.

Thanks to the authors and contributors of those projects.

## License

Released under the [MIT License](LICENSE). See `LICENSE` for the full text.

If you re-use code or ideas from [mini-swe-agent](https://github.com/SWE-bench/mini-swe-agent), please also retain attribution to that project (it is MIT-licensed as well).
