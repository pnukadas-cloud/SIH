"""
MAITRI — Context Memory & Persistent SQLite Database
Stores longitudinal session telemetry, astronaut interaction logs,
intervention efficacy records, and ground station alert histories.
"""

import sqlite3
import json
import time
from typing import Dict, Any, List, Optional
from maitri.config import DB_PATH

class ContextMemory:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_tables()
        self.session_messages = []
        
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def _init_tables(self):
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Telemetry Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    astronaut_id TEXT,
                    dominant_emotion TEXT,
                    confidence REAL,
                    valence REAL,
                    arousal REAL,
                    risk_score REAL,
                    risk_level INTEGER,
                    perclos REAL,
                    pitch_f0 REAL,
                    fused_probabilities TEXT
                )
            """)
            
            # Dialogue Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    astronaut_id TEXT,
                    speaker TEXT,
                    message TEXT,
                    detected_emotion TEXT,
                    intervention_id TEXT,
                    risk_level INTEGER
                )
            """)
            
            # Ground Alerts Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ground_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    timestamp REAL,
                    astronaut_id TEXT,
                    risk_level INTEGER,
                    risk_score REAL,
                    dominant_emotion TEXT,
                    payload_json TEXT,
                    status TEXT
                )
            """)
            
            conn.commit()

    def log_telemetry(self, astronaut_id: str, fused_data: Dict[str, Any], risk_data: Dict[str, Any], vision_data: Dict[str, Any], audio_data: Dict[str, Any]):
        """Record real-time telemetry snapshot."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_logs 
                (timestamp, astronaut_id, dominant_emotion, confidence, valence, arousal, risk_score, risk_level, perclos, pitch_f0, fused_probabilities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                astronaut_id,
                fused_data.get("dominant_emotion", "neutral"),
                fused_data.get("confidence", 0.0),
                fused_data.get("valence", 0.0),
                fused_data.get("arousal", 0.0),
                risk_data.get("risk_score", 0.0),
                risk_data.get("risk_level", 0),
                vision_data.get("perclos", 0.0),
                audio_data.get("pitch_f0_hz", 0.0),
                json.dumps(fused_data.get("fused_probabilities", {}))
            ))
            conn.commit()

    def log_message(self, astronaut_id: str, speaker: str, message: str, detected_emotion: str = "neutral", intervention_id: str = None, risk_level: int = 0):
        """Record conversation message in session context and database."""
        now = time.time()
        entry = {
            "timestamp": now,
            "speaker": speaker,
            "message": message,
            "detected_emotion": detected_emotion,
            "intervention_id": intervention_id,
            "risk_level": risk_level
        }
        self.session_messages.append(entry)
        if len(self.session_messages) > 30:
            self.session_messages.pop(0)
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dialogue_logs 
                (timestamp, astronaut_id, speaker, message, detected_emotion, intervention_id, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, astronaut_id, speaker, message, detected_emotion, intervention_id, risk_level))
            conn.commit()

    def log_alert(self, alert_id: str, astronaut_id: str, risk_level: int, risk_score: float, dominant_emotion: str, payload: Dict[str, Any]):
        """Record ground station alert."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ground_alerts 
                (alert_id, timestamp, astronaut_id, risk_level, risk_score, dominant_emotion, payload_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, now, astronaut_id, risk_level, risk_score, dominant_emotion, json.dumps(payload), "QUEUED_FOR_ORBITAL_PASS"))
            conn.commit()

    def get_recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent conversation messages."""
        return self.session_messages[-limit:]
        
    def get_telemetry_history(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Retrieve historical telemetry records for dashboard charting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, dominant_emotion, valence, arousal, risk_score, risk_level, perclos, pitch_f0
                FROM telemetry_logs
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]
            
    def get_alerts_history(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieve historical ground alerts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT alert_id, timestamp, astronaut_id, risk_level, risk_score, dominant_emotion, status
                FROM ground_alerts
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
