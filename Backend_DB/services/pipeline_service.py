"""
Backend_DB — Master Pipeline Orchestrator Service
Integrates AIML domain modules with Database persistence, Ground alerts, and Crew profiles.
"""

import time
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from AIML.facial_emotion.fer_module import FacialEmotionModule
from AIML.speech_acoustics.ser_module import SpeechAcousticsModule
from AIML.emotional_valence.fusion_module import EmotionalValenceFusionModule
from AIML.wellbeing.wellbeing_evaluator import WellBeingEvaluator
from AIML.maitri.companion_ai import MaitriCompanionAI
from Backend_DB.database.connection import DatabaseManager
from maitri.analysis.text_sentiment_engine import TextSentimentEngine
from maitri.telemetry.ground_station import GroundStationDispatcher
from maitri.response.interventions import InterventionManager
from maitri.config import DATA_DIR
import json

class MasterPipelineService:
    def __init__(self):
        # 1. AIML Modules
        self.fer = FacialEmotionModule()
        self.ser = SpeechAcousticsModule()
        self.sentiment = TextSentimentEngine()
        self.fusion = EmotionalValenceFusionModule()
        self.wellbeing = WellBeingEvaluator()
        self.companion = MaitriCompanionAI()

        # 2. Database & Dispatcher
        self.db = DatabaseManager()
        self.ground = GroundStationDispatcher()
        self.interventions = InterventionManager()

        # 3. Crew Profiles
        self.crew_profiles = self._load_crew_profiles()
        self.active_astronaut = self.crew_profiles[0] if self.crew_profiles else {
            "astronaut_id": "CREW-BAS-01",
            "name": "Captain Vikram Rathore",
            "callsign": "SURYA-1",
            "role": "Mission Commander"
        }
        self.last_telemetry: Optional[Dict[str, Any]] = None
        self.last_alert_time = 0.0

    def _load_crew_profiles(self) -> List[Dict[str, Any]]:
        path = DATA_DIR / "astronaut_baselines.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("crew_profiles", [])
        return []

    def set_active_astronaut(self, astronaut_id: str) -> Dict[str, Any]:
        for p in self.crew_profiles:
            if p["astronaut_id"] == astronaut_id:
                self.active_astronaut = p
                return p
        return self.active_astronaut

    def process_frame_and_audio(
        self,
        frame: Optional[np.ndarray] = None,
        audio_chunk: Optional[np.ndarray] = None,
        text_transcript: str = ""
    ) -> Dict[str, Any]:
        """Execute end-to-end multimodal perception and assessment."""
        now = time.time()

        # 1. Preprocessing & Analysis
        fer_res = self.fer.extract_features(frame)
        ser_res = self.ser.extract_prosody(audio_chunk)
        text_res = self.sentiment.analyze(text_transcript)

        # 2. Attention Fusion & Valence
        fused_res = self.fusion.fuse(fer_res, ser_res, text_res)

        # 3. Well-Being Evaluation
        physical_features = {
            "perclos_percentage": fer_res.get("perclos", 0.04) * 100.0,
            "yawns_per_min": fer_res.get("yawns_per_min", 0),
            "blinks_per_min": fer_res.get("blinks_per_min", 16.0)
        }
        wellbeing_res = self.wellbeing.evaluate_wellbeing(
            fused_res,
            physical_features,
            ser_res,
            self.active_astronaut.get("baseline_vitals")
        )

        # 4. Clinical Intervention Trigger
        selected_intervention = self.interventions.select_intervention(
            fused_res["dominant_emotion"],
            wellbeing_res["level"],
            physical_features
        )

        # 5. Ground Alert Check (Level >= 2 and throttle 30s)
        alert_packet = None
        if wellbeing_res["level"] >= 2 and (now - self.last_alert_time > 30.0):
            alert_packet = self.ground.create_alert_packet(
                self.active_astronaut,
                fused_res,
                physical_features,
                {"risk_level": wellbeing_res["level"], "risk_score": wellbeing_res["wellbeing_score"], "tier_name": wellbeing_res["tier_name"]},
                {"current_emotion": fused_res["dominant_emotion"], "trajectory_trend": "elevating"},
                selected_intervention
            )
            self.last_alert_time = now
            self.db.insert_alert(
                alert_packet["alert_id"],
                self.active_astronaut["astronaut_id"],
                wellbeing_res["level"],
                wellbeing_res["wellbeing_score"],
                fused_res["dominant_emotion"],
                alert_packet
            )

        # 6. Database Logging
        self.db.insert_telemetry(
            self.active_astronaut["astronaut_id"],
            fused_res,
            wellbeing_res,
            fer_res,
            ser_res
        )

        # 7. Construct Unified Telemetry Response
        telemetry = {
            "timestamp": now,
            "astronaut": self.active_astronaut,
            "vision": {
                "face_detected": fer_res["face_detected"],
                "face_count": fer_res.get("face_count", 0),
                "multiple_faces": fer_res.get("multiple_faces", False),
                "lighting": fer_res.get("lighting", {}),
                "face_bounding_box": fer_res.get("face_bounding_box"),
                "eye_aspect_ratio": fer_res.get("eye_aspect_ratio", 0.28),
                "mouth_aspect_ratio": fer_res.get("mouth_aspect_ratio", 0.20),
                "blinks_per_min": fer_res.get("blinks_per_min", 0.0),
                "yawns_per_min": fer_res.get("yawns_per_min", 0),
                "perclos": fer_res.get("perclos", 0.0),
                "action_units": fer_res.get("action_units", {})
            },
            "audio": {
                "is_speech_active": ser_res.get("is_speech_active", False),
                "pitch_f0_hz": ser_res.get("pitch_f0_hz", 130.0),
                "rms_energy": ser_res.get("rms_energy", 0.0),
                "db_level": ser_res.get("db_level", -60.0),
                "vocal_tension_score": ser_res.get("vocal_tension_score", 0.0),
                "vocal_jitter": ser_res.get("vocal_jitter", 0.01),
                "spectral_centroid_hz": ser_res.get("spectral_centroid_hz", 1000.0)
            },
            "fer": fer_res,
            "ser": ser_res,
            "text_sentiment": text_res,
            "fusion": fused_res,
            "wellbeing": wellbeing_res,
            "risk_assessment": {
                "risk_score": wellbeing_res["wellbeing_score"],
                "risk_level": wellbeing_res["level"],
                "tier_name": wellbeing_res["tier_name"],
                "status_color": wellbeing_res["status_color"]
            },
            "physical_distress": {
                "perclos_percentage": physical_features["perclos_percentage"],
                "blink_rate_bpm": physical_features["blinks_per_min"],
                "yawns_per_min": physical_features["yawns_per_min"],
                "fatigue_level": "Severe" if physical_features["perclos_percentage"] > 12.0 else ("Moderate" if physical_features["perclos_percentage"] > 8.0 else "Nominal / Rested"),
                "status_color": wellbeing_res["status_color"]
            },
            "recommended_intervention": selected_intervention,
            "alert_dispatched": alert_packet
        }

        self.last_telemetry = telemetry
        return telemetry

    def interact(self, astronaut_message: str) -> Dict[str, Any]:
        """Generate conversational response using MAITRI companion AI."""
        fused = self.last_telemetry.get("fusion", {}) if self.last_telemetry else {
            "dominant_emotion": "neutral", "valence": 0.0, "arousal": 0.2
        }
        wellbeing = self.last_telemetry.get("wellbeing", {}) if self.last_telemetry else {
            "wellbeing_score": 14.0, "level": 0, "tier_name": "Level 0: Nominal / Rested"
        }
        physical = self.last_telemetry.get("physical_distress", {}) if self.last_telemetry else {}

        response = self.companion.generate_response(
            astronaut_message,
            self.active_astronaut,
            fused,
            physical,
            wellbeing
        )
        return {
            "ai_response": response["response_text"],
            "model_source": response["model_source"],
            "detected_state": response["detected_state"],
            "intervention": self.interventions.get_intervention(response.get("intervention_id")) if response.get("intervention_id") else None,
            "latency_ms": response["latency_ms"]
        }

    def simulate_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Preset scenarios for instant 1-click hackathon demonstration."""
        scenarios = {
            "nominal": {
                "transcript": "Station systems nominal, completing orbit 14 Earth observation logs.",
                "emotion": "happy", "valence": 0.55, "arousal": 0.35, "pitch": 132.0, "perclos": 0.03, "tension": 0.08
            },
            "docking_stress": {
                "transcript": "RCS thruster alignment error on port 2! Thruster firing intermittent during final 10-meter docking approach!",
                "emotion": "stressed", "valence": -0.65, "arousal": 0.85, "pitch": 242.0, "perclos": 0.04, "tension": 0.82
            },
            "isolation_sadness": {
                "transcript": "Quiet cycle in the module. Missing family back home during Diwali tonight.",
                "emotion": "sad", "valence": -0.58, "arousal": 0.22, "pitch": 108.0, "perclos": 0.06, "tension": 0.20
            },
            "severe_fatigue": {
                "transcript": "Second consecutive sleepless circadian cycle... eyes burning, reaction latency slipping.",
                "emotion": "fatigued", "valence": -0.45, "arousal": 0.18, "pitch": 102.0, "perclos": 0.19, "tension": 0.35
            },
            "masked_stress": {
                "transcript": "Everything is fine here, nothing to report, completely nominal.",
                "emotion": "stressed", "valence": -0.30, "arousal": 0.72, "pitch": 215.0, "perclos": 0.05, "tension": 0.75
            }
        }
        spec = scenarios.get(scenario_name, scenarios["nominal"])

        # Construct synthetic biometrics matching scenario
        fer_res = {
            "face_detected": True, "face_count": 1, "multiple_faces": False,
            "lighting": {"status": "OPTIMAL", "brightness": 120, "sharpness": 85},
            "eye_aspect_ratio": 0.16 if spec["emotion"] == "fatigued" else 0.29,
            "mouth_aspect_ratio": 0.58 if spec["emotion"] == "fatigued" else 0.22,
            "blinks_per_min": 24.0 if spec["emotion"] == "stressed" else 12.0,
            "yawns_per_min": 3 if spec["emotion"] == "fatigued" else 0,
            "perclos": spec["perclos"],
            "action_units": {"AU04_brow_furrow": 0.8 if spec["emotion"] in ["stressed", "frustrated"] else 0.1, "AU12_lip_corner_puller": 0.7 if spec["emotion"] == "happy" else 0.0},
            "probabilities": {e: 0.70 if e == spec["emotion"] else 0.05 for e in ['neutral', 'happy', 'stressed', 'fatigued', 'anxious', 'sad', 'frustrated']},
            "dominant_emotion": spec["emotion"], "confidence": 0.88,
            "valence": spec["valence"], "arousal": spec["arousal"], "modality_active": True
        }

        ser_res = {
            "is_speech_active": True, "pitch_f0_hz": spec["pitch"], "rms_energy": 0.08 if spec["emotion"] == "stressed" else 0.04,
            "db_level": -18.0 if spec["emotion"] == "stressed" else -32.0, "vocal_tension_score": spec["tension"],
            "vocal_jitter": 0.06 if spec["emotion"] == "stressed" else 0.015, "spectral_centroid_hz": 2400.0 if spec["emotion"] == "stressed" else 1350.0,
            "probabilities": {e: 0.75 if e == spec["emotion"] else 0.04 for e in ['neutral', 'happy', 'stressed', 'fatigued', 'anxious', 'sad', 'frustrated']},
            "dominant_emotion": spec["emotion"], "confidence": 0.85,
            "valence": spec["valence"], "arousal": spec["arousal"], "modality_active": True
        }

        text_res = self.sentiment.analyze(spec["transcript"])
        fused_res = self.fusion.fuse(fer_res, ser_res, text_res)
        
        # In masked stress, explicitly trigger discordance
        if scenario_name == "masked_stress":
            fused_res["cross_modal_discordance"] = True
            fused_res["discordance_reason"] = "Masked Stress: Verbal statement claims nominal state, but vocal tension and pitch shift indicate severe autonomic stress."

        physical_features = {"perclos_percentage": spec["perclos"] * 100.0, "yawns_per_min": fer_res["yawns_per_min"], "blinks_per_min": fer_res["blinks_per_min"]}
        wellbeing_res = self.wellbeing.evaluate_wellbeing(fused_res, physical_features, ser_res, self.active_astronaut.get("baseline_vitals"))
        selected_intervention = self.interventions.select_intervention(fused_res["dominant_emotion"], wellbeing_res["level"], physical_features)

        # Alert if level >= 2
        alert_packet = None
        if wellbeing_res["level"] >= 2:
            alert_packet = self.ground.create_alert_packet(
                self.active_astronaut, fused_res, physical_features,
                {"risk_level": wellbeing_res["level"], "risk_score": wellbeing_res["wellbeing_score"], "tier_name": wellbeing_res["tier_name"]},
                {"current_emotion": fused_res["dominant_emotion"], "trajectory_trend": "elevating"},
                selected_intervention
            )
            self.db.insert_alert(alert_packet["alert_id"], self.active_astronaut["astronaut_id"], wellbeing_res["level"], wellbeing_res["wellbeing_score"], fused_res["dominant_emotion"], alert_packet)

        # Log
        self.db.insert_telemetry(self.active_astronaut["astronaut_id"], fused_res, wellbeing_res, fer_res, ser_res)

        telemetry = {
            "timestamp": time.time(),
            "astronaut": self.active_astronaut,
            "transcript": spec["transcript"],
            "vision": fer_res,
            "audio": ser_res,
            "fer": fer_res,
            "ser": ser_res,
            "text_sentiment": text_res,
            "fusion": fused_res,
            "wellbeing": wellbeing_res,
            "risk_assessment": {
                "risk_score": wellbeing_res["wellbeing_score"],
                "risk_level": wellbeing_res["level"],
                "tier_name": wellbeing_res["tier_name"],
                "status_color": wellbeing_res["status_color"]
            },
            "physical_distress": {
                "perclos_percentage": physical_features["perclos_percentage"],
                "blink_rate_bpm": physical_features["blinks_per_min"],
                "yawns_per_min": physical_features["yawns_per_min"],
                "fatigue_level": "Severe" if physical_features["perclos_percentage"] > 12.0 else ("Moderate" if physical_features["perclos_percentage"] > 8.0 else "Nominal / Rested"),
                "status_color": wellbeing_res["status_color"]
            },
            "recommended_intervention": selected_intervention,
            "alert_dispatched": alert_packet
        }
        self.last_telemetry = telemetry
        return telemetry
