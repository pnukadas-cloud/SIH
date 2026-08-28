"""
MAITRI — Master Pipeline Orchestrator
Coordinates end-to-end Multimodal AI analysis, risk escalation,
evidence-based interventions, ground telemetry, and conversational AI.
"""

import time
import json
import base64
import numpy as np
import cv2
from typing import Dict, Any, Optional

from maitri.config import DATA_DIR
from maitri.preprocessing.vision_processor import VisionProcessor
from maitri.preprocessing.audio_processor import AudioProcessor
from maitri.analysis.fer_engine import FacialEmotionEngine
from maitri.analysis.ser_engine import SpeechEmotionEngine
from maitri.analysis.text_sentiment_engine import TextSentimentEngine
from maitri.analysis.physical_distress_engine import PhysicalDistressEngine
from maitri.analysis.multimodal_fusion import MultimodalFusionEngine
from maitri.decision.state_tracker import EmotionStateTracker
from maitri.decision.risk_scorer import RiskScorer
from maitri.decision.context_memory import ContextMemory
from maitri.response.interventions import InterventionManager
from maitri.response.conversational_agent import ConversationalAgent
from maitri.response.tts_engine import TTSEngine
from maitri.telemetry.ground_station import GroundStationDispatcher
from maitri.telemetry.mission_logger import MissionLogger

class MaitriPipeline:
    def __init__(self):
        # 1. Preprocessing
        self.vision = VisionProcessor()
        self.audio = AudioProcessor()
        
        # 2. Multimodal Analysis
        self.fer = FacialEmotionEngine()
        self.ser = SpeechEmotionEngine()
        self.sentiment = TextSentimentEngine()
        self.physical = PhysicalDistressEngine()
        self.fusion = MultimodalFusionEngine()
        
        # 3. Decision & State Engine
        self.state_tracker = EmotionStateTracker()
        self.risk_scorer = RiskScorer()
        self.memory = ContextMemory()
        
        # 4. Response Engine
        self.interventions = InterventionManager()
        self.agent = ConversationalAgent()
        self.tts = TTSEngine()
        
        # 5. Telemetry & Ground Dispatch
        self.ground = GroundStationDispatcher()
        self.logger = MissionLogger()
        
        # Load Crew Profiles
        self.crew_profiles = self._load_crew_profiles()
        self.active_astronaut = self.crew_profiles[0] if self.crew_profiles else {
            "astronaut_id": "CREW-BAS-01",
            "callsign": "Vyom-Leader",
            "name": "Wing Cmdr. Prashanth Nair",
            "role": "Mission Commander"
        }
        
        # State caches
        self.last_telemetry = None
        self.last_alert_time = 0.0
        
    def _load_crew_profiles(self):
        try:
            profile_path = DATA_DIR / "astronaut_baselines.json"
            if profile_path.exists():
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("crew_profiles", [])
        except Exception as e:
            print(f"[MAITRI] Profile load error: {e}")
        return []

    def set_active_astronaut(self, astronaut_id: str):
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
        """
        Execute full multimodal pipeline on sensor inputs.
        """
        now = time.time()
        
        # Step 1: Preprocessing
        vision_feat = self.vision.process_frame(frame) if frame is not None else self.vision._default_empty_response(now)
        audio_feat = self.audio.process_audio_chunk(audio_chunk) if audio_chunk is not None else self.audio._default_empty_response()
        
        # Step 2: Feature Analysis
        fer_res = self.fer.predict(vision_feat)
        ser_res = self.ser.predict(audio_feat)
        text_res = self.sentiment.analyze(text_transcript)
        physical_res = self.physical.evaluate(vision_feat, audio_feat, text_res)
        
        # Step 3: Multimodal Fusion
        fused_res = self.fusion.fuse(fer_res, ser_res, text_res)
        
        # Step 4: Decision & State Tracking
        state_res = self.state_tracker.update(fused_res, physical_res)
        risk_res = self.risk_scorer.calculate_risk(fused_res, physical_res, state_res)
        
        # Step 5: Clinical Intervention Selection
        selected_intervention = self.interventions.select_intervention(
            fused_res["dominant_emotion"],
            risk_res["risk_level"],
            physical_res
        )
        
        # Step 6: Ground Station Alert Escalation Check
        # Trigger Ground alert if Risk Level >= 2 and at least 30s elapsed since last alert
        alert_packet = None
        if risk_res["risk_level"] >= 2 and (now - self.last_alert_time > 30.0):
            alert_packet = self.ground.create_alert_packet(
                self.active_astronaut,
                fused_res,
                physical_res,
                risk_res,
                state_res,
                selected_intervention
            )
            self.last_alert_time = now
            self.memory.log_alert(
                alert_packet["alert_id"],
                self.active_astronaut["astronaut_id"],
                risk_res["risk_level"],
                risk_res["risk_score"],
                fused_res["dominant_emotion"],
                alert_packet
            )
            
        # Step 7: Persistence Logging
        self.memory.log_telemetry(
            self.active_astronaut["astronaut_id"],
            fused_res,
            risk_res,
            vision_feat,
            audio_feat
        )
        
        # Encode HUD Frame to JPEG Base64 for web streaming if available
        hud_jpeg_b64 = None
        if vision_feat.get("hud_frame") is not None:
            _, buffer = cv2.imencode('.jpg', vision_feat["hud_frame"], [cv2.IMWRITE_JPEG_QUALITY, 75])
            hud_jpeg_b64 = base64.b64encode(buffer).decode('utf-8')
            
        telemetry = {
            "timestamp": now,
            "astronaut": self.active_astronaut,
            "vision": {
                "face_detected": vision_feat["face_detected"],
                "eye_aspect_ratio": vision_feat["eye_aspect_ratio"],
                "mouth_aspect_ratio": vision_feat["mouth_aspect_ratio"],
                "blinks_per_min": vision_feat["blinks_per_min"],
                "yawns_per_min": vision_feat["yawns_per_min"],
                "perclos": vision_feat["perclos"],
                "action_units": vision_feat["action_units"],
                "smile_detected": vision_feat["smile_detected"]
            },
            "audio": {
                "is_speech_active": audio_feat["is_speech_active"],
                "pitch_f0_hz": audio_feat["pitch_f0_hz"],
                "rms_energy": audio_feat["rms_energy"],
                "db_level": audio_feat["db_level"],
                "vocal_tension_score": audio_feat["vocal_tension_score"],
                "vocal_jitter": audio_feat["vocal_jitter"],
                "spectral_centroid_hz": audio_feat["spectral_centroid_hz"]
            },
            "fer": fer_res,
            "ser": ser_res,
            "text_sentiment": text_res,
            "physical_distress": physical_res,
            "fusion": fused_res,
            "state_tracking": state_res,
            "risk_assessment": risk_res,
            "recommended_intervention": selected_intervention,
            "alert_dispatched": alert_packet,
            "hud_frame_base64": hud_jpeg_b64
        }
        
        self.last_telemetry = telemetry
        return telemetry

    def interact(self, astronaut_message: str) -> Dict[str, Any]:
        """
        Process dialogue from astronaut, generate empathetic guidance, and speak.
        """
        now = time.time()
        
        # Analyze linguistic sentiment
        text_res = self.sentiment.analyze(astronaut_message)
        
        # Build composite features from recent telemetry or defaults
        fused_res = self.last_telemetry.get("fusion") if self.last_telemetry else self.fusion.fuse(
            self.fer.predict({"face_detected": False}),
            self.ser.predict({"is_speech_active": False}),
            text_res
        )
        
        physical_res = self.last_telemetry.get("physical_distress") if self.last_telemetry else {
            "composite_physical_distress_score": 10.0,
            "fatigue_level": "Nominal",
            "status_color": "green"
        }
        
        state_res = self.last_telemetry.get("state_tracking") if self.last_telemetry else {
            "current_emotion": text_res["dominant_emotion"],
            "duration_in_state_seconds": 30.0,
            "trajectory_trend": "stable"
        }
        
        risk_res = self.risk_scorer.calculate_risk(fused_res, physical_res, state_res)
        
        # Select intervention
        intervention = self.interventions.select_intervention(
            fused_res["dominant_emotion"],
            risk_res["risk_level"],
            physical_res
        )
        
        # Generate conversational response
        response = self.agent.generate_response(
            astronaut_message,
            self.active_astronaut,
            fused_res,
            physical_res,
            risk_res,
            intervention
        )
        
        # Log to context memory
        self.memory.log_message(
            self.active_astronaut["astronaut_id"],
            speaker="ASTRONAUT",
            message=astronaut_message,
            detected_emotion=text_res["dominant_emotion"],
            risk_level=risk_res["risk_level"]
        )
        
        self.memory.log_message(
            self.active_astronaut["astronaut_id"],
            speaker="MAITRI",
            message=response["response_text"],
            detected_emotion=response["detected_state"],
            intervention_id=response.get("intervention_id"),
            risk_level=risk_res["risk_level"]
        )
        
        # Trigger offline speech synthesis
        self.tts.speak_async(response["response_text"])
        
        return {
            "user_message": astronaut_message,
            "ai_response": response["response_text"],
            "detected_state": response["detected_state"],
            "risk_level": response["risk_level"],
            "intervention": intervention,
            "verbal_guidance": response.get("verbal_guidance"),
            "timestamp": now
        }
        
    def simulate_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Generate synthetic high-fidelity multimodal telemetry for live SIH demonstration.
        """
        now = time.time()
        
        if scenario_name == "nominal":
            # Scenario 1: Nominal flight operations
            fer_sim = {"face_detected": True, "eye_aspect_ratio": 0.32, "mouth_aspect_ratio": 0.19, "blinks_per_min": 16.0, "yawns_per_min": 0, "perclos": 0.04, "action_units": {"AU04_brow_furrow": 0.05, "AU12_lip_corner_puller": 0.25, "AU20_lip_stretcher": 0.05, "AU43_eye_closure": 0.0}, "smile_detected": True, "modality_active": True}
            audio_sim = {"is_speech_active": True, "rms_energy": 0.06, "db_level": -24.0, "pitch_f0_hz": 132.0, "vocal_tension_score": 0.08, "vocal_jitter": 0.015, "spectral_centroid_hz": 1400.0}
            text_sim = "Telemetry checks are complete. Life support and navigation are nominal."
            
        elif scenario_name == "docking_stress":
            # Scenario 2: High-consequence docking stress / Level 2
            fer_sim = {"face_detected": True, "eye_aspect_ratio": 0.22, "mouth_aspect_ratio": 0.28, "blinks_per_min": 32.0, "yawns_per_min": 0, "perclos": 0.08, "action_units": {"AU04_brow_furrow": 0.85, "AU12_lip_corner_puller": 0.0, "AU20_lip_stretcher": 0.75, "AU43_eye_closure": 0.0}, "smile_detected": False, "modality_active": True}
            audio_sim = {"is_speech_active": True, "rms_energy": 0.12, "db_level": -18.0, "pitch_f0_hz": 238.0, "vocal_tension_score": 0.78, "vocal_jitter": 0.045, "spectral_centroid_hz": 2450.0}
            text_sim = "Thruster variance detected during final approach! The alignment angle is slipping, pressure rising!"
            
        elif scenario_name == "isolation_sadness":
            # Scenario 3: Prolonged space isolation / Loneliness
            fer_sim = {"face_detected": True, "eye_aspect_ratio": 0.26, "mouth_aspect_ratio": 0.15, "blinks_per_min": 11.0, "yawns_per_min": 0, "perclos": 0.14, "action_units": {"AU04_brow_furrow": 0.55, "AU12_lip_corner_puller": 0.0, "AU20_lip_stretcher": 0.10, "AU43_eye_closure": 0.1}, "smile_detected": False, "modality_active": True}
            audio_sim = {"is_speech_active": True, "rms_energy": 0.025, "db_level": -32.0, "pitch_f0_hz": 105.0, "vocal_tension_score": 0.15, "vocal_jitter": 0.022, "spectral_centroid_hz": 980.0}
            text_sim = "It's so quiet here. Another 90-day stretch without seeing my family. Earth looks so far away."
            
        elif scenario_name == "severe_fatigue":
            # Scenario 4: Critical sleep disruption & microsleep risk / Level 3
            fer_sim = {"face_detected": True, "eye_aspect_ratio": 0.15, "mouth_aspect_ratio": 0.65, "blinks_per_min": 6.0, "yawns_per_min": 3, "perclos": 0.38, "action_units": {"AU04_brow_furrow": 0.35, "AU12_lip_corner_puller": 0.0, "AU20_lip_stretcher": 0.40, "AU43_eye_closure": 0.8}, "smile_detected": False, "modality_active": True}
            audio_sim = {"is_speech_active": True, "rms_energy": 0.018, "db_level": -36.0, "pitch_f0_hz": 95.0, "vocal_tension_score": 0.45, "vocal_jitter": 0.038, "spectral_centroid_hz": 820.0}
            text_sim = "I cannot keep my eyes open. 4th consecutive day on circadian shift... headache is intense."
            
        elif scenario_name == "masked_stress":
            # Scenario 5: Masked distress (Astronaut smiling facially to hide panic)
            fer_sim = {"face_detected": True, "eye_aspect_ratio": 0.30, "mouth_aspect_ratio": 0.30, "blinks_per_min": 30.0, "yawns_per_min": 0, "perclos": 0.05, "action_units": {"AU04_brow_furrow": 0.20, "AU12_lip_corner_puller": 0.85, "AU20_lip_stretcher": 0.35, "AU43_eye_closure": 0.0}, "smile_detected": True, "modality_active": True}
            audio_sim = {"is_speech_active": True, "rms_energy": 0.11, "db_level": -19.0, "pitch_f0_hz": 245.0, "vocal_tension_score": 0.82, "vocal_jitter": 0.052, "spectral_centroid_hz": 2600.0}
            text_sim = "Everything is fine, nothing to worry about here! Haha... just a minor warning light."
        else:
            return self.simulate_scenario("nominal")
            
        # Run pipeline with simulated features
        fer_res = self.fer.predict(fer_sim)
        ser_res = self.ser.predict(audio_sim)
        text_res = self.sentiment.analyze(text_sim)
        physical_res = self.physical.evaluate(fer_sim, audio_sim, text_res)
        fused_res = self.fusion.fuse(fer_res, ser_res, text_res)
        state_res = self.state_tracker.update(fused_res, physical_res)
        risk_res = self.risk_scorer.calculate_risk(fused_res, physical_res, state_res)
        
        intervention = self.interventions.select_intervention(
            fused_res["dominant_emotion"],
            risk_res["risk_level"],
            physical_res
        )
        
        # Ground Station Alert if needed
        alert_packet = None
        if risk_res["risk_level"] >= 2:
            alert_packet = self.ground.create_alert_packet(
                self.active_astronaut,
                fused_res,
                physical_res,
                risk_res,
                state_res,
                intervention
            )
            self.memory.log_alert(
                alert_packet["alert_id"],
                self.active_astronaut["astronaut_id"],
                risk_res["risk_level"],
                risk_res["risk_score"],
                fused_res["dominant_emotion"],
                alert_packet
            )
            
        telemetry = {
            "timestamp": now,
            "scenario": scenario_name,
            "astronaut": self.active_astronaut,
            "vision": fer_sim,
            "audio": audio_sim,
            "transcript": text_sim,
            "fer": fer_res,
            "ser": ser_res,
            "text_sentiment": text_res,
            "physical_distress": physical_res,
            "fusion": fused_res,
            "state_tracking": state_res,
            "risk_assessment": risk_res,
            "recommended_intervention": intervention,
            "alert_dispatched": alert_packet,
            "hud_frame_base64": None
        }
        
        self.last_telemetry = telemetry
        return telemetry
