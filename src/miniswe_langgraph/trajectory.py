from datetime import datetime
from pathlib import Path
import json

class Trajectory:

    def __init__(self, directory="logs"):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.path = (
            self.directory/f"last_run_traj.json"
        )
        self.events = []

    def add(self, event_type: str, data):
        self.events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "data": data,
            }
        )

    def save(self):
        self.path.write_text(
            json.dumps(
                self.events,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
