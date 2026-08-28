"""
MAITRI — Mission Logger & Flight Data Recorder
Persists mission flight telemetry in JSON and CSV formats for offline review.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
from maitri.config import LOGS_DIR

class MissionLogger:
    def __init__(self, log_dir: Path = LOGS_DIR):
        self.log_dir = log_dir
        self.current_session_file = self.log_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Append telemetry event to session log file."""
        record = {
            "timestamp": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
            "payload": data
        }
        try:
            with open(self.current_session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[MAITRI] Log error: {e}")
