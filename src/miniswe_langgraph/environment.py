import subprocess
from pathlib import Path

class Environment:

    def __init__(self, workdir: str = "./workspace") -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    def execute(self, command: str) -> tuple[int, str]:
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr
