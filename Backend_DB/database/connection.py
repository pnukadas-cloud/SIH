"""
Backend_DB — Database Connection Manager & Repository
SQLite storage for longitudinal telemetry, dialogues, ground alerts, and session logs.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from maitri.config import DB_PATH

class DatabaseManager:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Telemetry logs
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
                    vocal_tension REAL,
                    fused_probabilities TEXT
                )
            """)

            # Dialogue logs
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

            # Ground Alerts
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

            try:
                cursor.execute("ALTER TABLE telemetry_logs ADD COLUMN vocal_tension REAL DEFAULT 0.1")
                conn.commit()
            except sqlite3.OperationalError:
                pass

    def insert_telemetry(self, astronaut_id: str, fused: Dict[str, Any], wellbeing: Dict[str, Any], vision: Dict[str, Any], audio: Dict[str, Any]):
        now = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_logs 
                (timestamp, astronaut_id, dominant_emotion, confidence, valence, arousal, risk_score, risk_level, perclos, pitch_f0, vocal_tension, fused_probabilities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                astronaut_id,
                fused.get("dominant_emotion", "neutral"),
                fused.get("confidence", 0.0),
                fused.get("valence", 0.0),
                fused.get("arousal", 0.0),
                wellbeing.get("wellbeing_score", 12.0),
                wellbeing.get("level", 0),
                vision.get("perclos", 0.0),
                audio.get("pitch_f0_hz", 130.0),
                audio.get("vocal_tension_score", 0.1),
                json.dumps(fused.get("fused_probabilities", {}))
            ))
            conn.commit()

    def get_recent_telemetry(self, astronaut_id: Optional[str] = None, limit: int = 60) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if astronaut_id:
                cursor.execute("""
                    SELECT id, timestamp, astronaut_id, dominant_emotion, confidence, valence, arousal, risk_score, risk_level, perclos, pitch_f0, vocal_tension
                    FROM telemetry_logs
                    WHERE astronaut_id = ?
                    ORDER BY id DESC LIMIT ?
                """, (astronaut_id, limit))
            else:
                cursor.execute("""
                    SELECT id, timestamp, astronaut_id, dominant_emotion, confidence, valence, arousal, risk_score, risk_level, perclos, pitch_f0, vocal_tension
                    FROM telemetry_logs
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]

    def insert_alert(self, alert_id: str, astronaut_id: str, risk_level: int, risk_score: float, dominant_emotion: str, payload: Dict[str, Any]):
        now = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ground_alerts
                (alert_id, timestamp, astronaut_id, risk_level, risk_score, dominant_emotion, payload_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, now, astronaut_id, risk_level, risk_score, dominant_emotion, json.dumps(payload), "QUEUED_S_BAND"))
            conn.commit()

    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT alert_id, timestamp, astronaut_id, risk_level, risk_score, dominant_emotion, status
                FROM ground_alerts
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE ground_alerts SET status = 'ACKNOWLEDGED' WHERE alert_id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
