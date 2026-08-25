from pathlib import Path
from jinja2 import Environment, FileSystemLoader

PROMPT_DIR = Path(__file__).parent / "prompt_templates"

jinja = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)

def render_prompt(template: str, **kwargs) -> str:
    return jinja.get_template(template).render(**kwargs)


# ---------------------------------------------------------------------------
# Prompt-protocol constants
# ---------------------------------------------------------------------------
# These are shared between the agent (which injects format_error reminders)
# and the graph (which routes a format_error reminder back to the agent).

# How many times we will feed the model a `format_error` reminder before
# giving up and letting the run end.
MAX_FORMAT_ERROR_RETRIES = 3

# Marker prepended to HumanMessages produced by the format_error protocol.
# `route_after_agent` checks for this prefix to distinguish a format_error
# retry turn from a normal user turn.
FORMAT_ERROR_MARKER = "[format_error]"


