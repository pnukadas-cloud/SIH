"""
Backend_DB — Database Connection Manager & Repository
SQLite storage for longitudinal telemetry, dialogues, ground alerts, and session logs.
"""

import sqlite3
import json
import time
import hashlib
import numpy as np
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

            # Astronaut Profiles with Face Recognition Data (Identity System Upgrade)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS astronauts (
                    astronaut_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    callsign TEXT,
                    role TEXT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    profile_json TEXT,
                    baseline_json TEXT,
                    face_embedding_json TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)

            # Astronaut Monitoring Sessions (Phase 1 Identity & History)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_sessions (
                    session_id TEXT PRIMARY KEY,
                    astronaut_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    dominant_state TEXT,
                    avg_risk_score REAL,
                    emotion_summary TEXT,
                    risk_observations TEXT,
                    interventions_count INTEGER DEFAULT 0,
                    alerts_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """)
            conn.commit()

            try:
                cursor.execute("ALTER TABLE telemetry_logs ADD COLUMN vocal_tension REAL DEFAULT 0.1")
                conn.commit()
            except sqlite3.OperationalError:
                pass

        # Seed initial astronauts if table is empty
        self.seed_default_astronauts()

    def upsert_session(self, session_id: str, astronaut_id: str, start_time: float, 
                       dominant_state: str = "nominal", avg_risk_score: float = 12.0,
                       emotion_summary: Optional[Dict[str, Any]] = None, 
                       risk_observations: Optional[Dict[str, Any]] = None,
                       interventions_count: int = 0, alerts_count: int = 0,
                       status: str = "ACTIVE", end_time: Optional[float] = None):
        """Create or update an astronaut monitoring session record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO monitoring_sessions
                (session_id, astronaut_id, start_time, end_time, dominant_state, avg_risk_score, 
                 emotion_summary, risk_observations, interventions_count, alerts_count, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    end_time = excluded.end_time,
                    dominant_state = excluded.dominant_state,
                    avg_risk_score = excluded.avg_risk_score,
                    emotion_summary = excluded.emotion_summary,
                    risk_observations = excluded.risk_observations,
                    interventions_count = excluded.interventions_count,
                    alerts_count = excluded.alerts_count,
                    status = excluded.status
            """, (
                session_id, astronaut_id, start_time, end_time, dominant_state, avg_risk_score,
                json.dumps(emotion_summary or {}), json.dumps(risk_observations or {}),
                interventions_count, alerts_count, status
            ))
            conn.commit()

    def get_sessions(self, astronaut_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve sessions strictly partitioned by astronaut ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if astronaut_id:
                cursor.execute("""
                    SELECT session_id, astronaut_id, start_time, end_time, dominant_state, 
                           avg_risk_score, emotion_summary, risk_observations, interventions_count, alerts_count, status
                    FROM monitoring_sessions
                    WHERE astronaut_id = ?
                    ORDER BY start_time DESC LIMIT ?
                """, (astronaut_id, limit))
            else:
                cursor.execute("""
                    SELECT session_id, astronaut_id, start_time, end_time, dominant_state, 
                           avg_risk_score, emotion_summary, risk_observations, interventions_count, alerts_count, status
                    FROM monitoring_sessions
                    ORDER BY start_time DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["emotion_summary"] = json.loads(d.get("emotion_summary") or "{}")
                except Exception:
                    pass
                try:
                    d["risk_observations"] = json.loads(d.get("risk_observations") or "{}")
                except Exception:
                    pass
                result.append(d)
            return result

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

    def get_alerts(self, astronaut_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if astronaut_id:
                cursor.execute("""
                    SELECT alert_id, timestamp, astronaut_id, risk_level, risk_score, dominant_emotion, status
                    FROM ground_alerts
                    WHERE astronaut_id = ?
                    ORDER BY id DESC LIMIT ?
                """, (astronaut_id, limit))
            else:
                cursor.execute("""
                    SELECT alert_id, timestamp, astronaut_id, risk_level, risk_score, dominant_emotion, status
                    FROM ground_alerts
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def insert_dialogue(self, astronaut_id: str, speaker: str, message: str, detected_emotion: str = "neutral", intervention_id: Optional[str] = None, risk_level: int = 0):
        now = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dialogue_logs
                (timestamp, astronaut_id, speaker, message, detected_emotion, intervention_id, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, astronaut_id, speaker, message, detected_emotion, intervention_id, risk_level))
            conn.commit()

    def get_dialogues(self, astronaut_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if astronaut_id:
                cursor.execute("""
                    SELECT id, timestamp, astronaut_id, speaker, message, detected_emotion, intervention_id, risk_level
                    FROM dialogue_logs
                    WHERE astronaut_id = ?
                    ORDER BY id ASC LIMIT ?
                """, (astronaut_id, limit))
            else:
                cursor.execute("""
                    SELECT id, timestamp, astronaut_id, speaker, message, detected_emotion, intervention_id, risk_level
                    FROM dialogue_logs
                    ORDER BY id ASC LIMIT ?
                """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE ground_alerts SET status = 'ACKNOWLEDGED' WHERE alert_id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0

    # -------------------------------------------------------------
    # Database-Backed Astronaut Profile Operations
    # -------------------------------------------------------------
    def upsert_astronaut(
        self,
        astronaut_id: str,
        name: str,
        callsign: str,
        role: str,
        username: str,
        password_hash: str,
        profile: Optional[Dict[str, Any]] = None,
        baseline: Optional[Dict[str, Any]] = None,
        face_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Insert or update an astronaut record with credentials, baselines, and face embedding."""
        now = time.time()
        embedding_json = json.dumps(face_embedding) if face_embedding is not None else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO astronauts 
                (astronaut_id, name, callsign, role, username, password_hash, profile_json, baseline_json, face_embedding_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(astronaut_id) DO UPDATE SET
                    name = excluded.name,
                    callsign = excluded.callsign,
                    role = excluded.role,
                    username = excluded.username,
                    password_hash = COALESCE(excluded.password_hash, astronauts.password_hash),
                    profile_json = COALESCE(excluded.profile_json, astronauts.profile_json),
                    baseline_json = COALESCE(excluded.baseline_json, astronauts.baseline_json),
                    face_embedding_json = COALESCE(excluded.face_embedding_json, astronauts.face_embedding_json),
                    updated_at = excluded.updated_at
            """, (
                astronaut_id, name, callsign, role, username, password_hash,
                json.dumps(profile or {}), json.dumps(baseline or {}),
                embedding_json, now, now
            ))
            conn.commit()
        return self.get_astronaut(astronaut_id)

    def get_astronaut(self, astronaut_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an astronaut profile by unique astronaut ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM astronauts WHERE astronaut_id = ?", (astronaut_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["profile"] = json.loads(d.get("profile_json") or "{}")
            except Exception:
                d["profile"] = {}
            try:
                d["baseline_vitals"] = json.loads(d.get("baseline_json") or "{}")
            except Exception:
                d["baseline_vitals"] = {}
            try:
                d["face_embedding"] = json.loads(d.get("face_embedding_json") or "[]")
            except Exception:
                d["face_embedding"] = []
            return d

    def get_astronaut_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve an astronaut profile by username."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM astronauts WHERE LOWER(username) = LOWER(?)", (username,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["profile"] = json.loads(d.get("profile_json") or "{}")
            except Exception:
                d["profile"] = {}
            try:
                d["baseline_vitals"] = json.loads(d.get("baseline_json") or "{}")
            except Exception:
                d["baseline_vitals"] = {}
            try:
                d["face_embedding"] = json.loads(d.get("face_embedding_json") or "[]")
            except Exception:
                d["face_embedding"] = []
            return d

    def list_astronauts(self) -> List[Dict[str, Any]]:
        """List all enrolled astronauts from the database (excluding sensitive credentials)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT astronaut_id, name, callsign, role, username, profile_json, baseline_json,
                       (face_embedding_json IS NOT NULL AND length(face_embedding_json) > 10) as has_face_enrolled,
                       created_at, updated_at
                FROM astronauts
                ORDER BY astronaut_id ASC
            """)
            rows = cursor.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["profile"] = json.loads(d.get("profile_json") or "{}")
                except Exception:
                    d["profile"] = {}
                try:
                    d["baseline_vitals"] = json.loads(d.get("baseline_json") or "{}")
                except Exception:
                    d["baseline_vitals"] = {}
                result.append(d)
            return result

    def get_all_enrolled_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieve all enrolled astronauts with their face embeddings for matcher."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT astronaut_id, name, callsign, role, face_embedding_json
                FROM astronauts
                WHERE face_embedding_json IS NOT NULL AND length(face_embedding_json) > 10
            """)
            rows = cursor.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["embedding"] = json.loads(d["face_embedding_json"])
                    if isinstance(d["embedding"], list) and len(d["embedding"]) > 0:
                        result.append(d)
                except Exception:
                    pass
            return result

    def seed_default_astronauts(self):
        """Seed initial astronaut profiles with baselines and synthetic face embeddings if table is empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM astronauts")
            if cursor.fetchone()[0] > 0:
                return

        now = time.time()

        # Deterministic 128-dimensional seed embeddings with distinctive orthogonal seeds
        def make_seed_vector(seed_val: int) -> List[float]:
            rng = np.random.RandomState(seed_val)
            v = rng.randn(128).astype(np.float32)
            v = v / max(1e-6, float(np.linalg.norm(v)))
            return [round(float(x), 5) for x in v]

        default_crew = [
            {
                "astronaut_id": "AST-001",
                "name": "Aryan",
                "callsign": "GARUDA-1",
                "role": "Mission Commander",
                "username": "aryan",
                "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
                "profile": {
                    "mission_assignment": "Gaganyaan-BAS Expedition 1",
                    "clearance_level": "Level-2 (Mission Commander)",
                    "coping_preferences": ["Tactical box breathing", "Direct operational summaries", "Short cognitive reframing"]
                },
                "baseline": {
                    "resting_heart_rate_bpm": 64,
                    "blink_rate_bpm": 16.5,
                    "resting_ear": 0.32,
                    "resting_mar": 0.18,
                    "mean_f0_pitch_hz": 128.0
                },
                "embedding": make_seed_vector(101)
            },
            {
                "astronaut_id": "AST-002",
                "name": "Riya",
                "callsign": "TEJAS-2",
                "role": "Flight Engineer",
                "username": "riya",
                "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
                "profile": {
                    "mission_assignment": "Gaganyaan-BAS Expedition 1",
                    "clearance_level": "Level-2 (Flight Engineer)",
                    "coping_preferences": ["Objective checklist review", "Sensory grounding", "Micro-rest power nap"]
                },
                "baseline": {
                    "resting_heart_rate_bpm": 68,
                    "blink_rate_bpm": 18.0,
                    "resting_ear": 0.34,
                    "resting_mar": 0.19,
                    "mean_f0_pitch_hz": 215.0
                },
                "embedding": make_seed_vector(202)
            },
            {
                "astronaut_id": "AST-003",
                "name": "Karan",
                "callsign": "VIKRAM-3",
                "role": "Mission Specialist",
                "username": "karan",
                "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
                "profile": {
                    "mission_assignment": "Gaganyaan-BAS Expedition 1",
                    "clearance_level": "Level-2 (Mission Specialist)",
                    "coping_preferences": ["Classical ambient music", "Family audio logs", "Guided body scan"]
                },
                "baseline": {
                    "resting_heart_rate_bpm": 62,
                    "blink_rate_bpm": 15.0,
                    "resting_ear": 0.31,
                    "resting_mar": 0.17,
                    "mean_f0_pitch_hz": 134.0
                },
                "embedding": make_seed_vector(303)
            },
            {
                "astronaut_id": "CREW-BAS-01",
                "name": "Wing Cmdr. Prashanth Nair",
                "callsign": "Vyom-Leader",
                "role": "Mission Commander",
                "username": "prashanth",
                "password_hash": hashlib.sha256("astronaut123".encode()).hexdigest(),
                "profile": {
                    "mission_assignment": "Gaganyaan-BAS Expedition 1",
                    "clearance_level": "Level-2 (Mission Commander)",
                    "coping_preferences": ["Tactical breathing", "Direct operational summaries"]
                },
                "baseline": {
                    "resting_heart_rate_bpm": 64,
                    "blink_rate_bpm": 16.5,
                    "resting_ear": 0.32,
                    "resting_mar": 0.18,
                    "mean_f0_pitch_hz": 128.0
                },
                "embedding": make_seed_vector(404)
            },
            {
                "astronaut_id": "ADMIN-MED-01",
                "name": "Dr. Sunita Sharma",
                "callsign": "GROUND-SURGEON",
                "role": "Chief Flight Surgeon",
                "username": "surgeon_sharma",
                "password_hash": hashlib.sha256("isro_surgeon2025".encode()).hexdigest(),
                "profile": {
                    "mission_assignment": "ISRO Telemetry & Biomedical Mission Control",
                    "clearance_level": "Level-4 (Chief Medical Officer)",
                    "coping_preferences": ["Clinical triage", "Ground intervention override"]
                },
                "baseline": {
                    "resting_heart_rate_bpm": 70,
                    "blink_rate_bpm": 17.0,
                    "resting_ear": 0.33,
                    "resting_mar": 0.18,
                    "mean_f0_pitch_hz": 195.0
                },
                "embedding": make_seed_vector(505)
            }
        ]

        for crew in default_crew:
            self.upsert_astronaut(
                astronaut_id=crew["astronaut_id"],
                name=crew["name"],
                callsign=crew["callsign"],
                role=crew["role"],
                username=crew["username"],
                password_hash=crew["password_hash"],
                profile=crew["profile"],
                baseline=crew["baseline"],
                face_embedding=crew["embedding"]
            )
