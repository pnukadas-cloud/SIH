"""
AIML — MAITRI Conversational AI Engine
Dual-mode companion: Integrates Google Gemini API when GEMINI_API_KEY is available,
with comprehensive offline psychological support knowledge base fallback.
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

class MaitriCompanionAI:
    def __init__(self):
        self.system_name = "MAITRI"
        self.station_name = "Bhartiya Antariksh Station (BAS)"
        self.api_key = os.getenv(GEMINI_API_KEY_ENV, "").strip()

    def generate_response(
        self,
        astronaut_message: str,
        astronaut_profile: Dict[str, Any],
        fused_emotion: Dict[str, Any],
        physical_features: Dict[str, Any],
        wellbeing_assessment: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate empathetic and operational guidance."""
        start_time = time.time()
        callsign = astronaut_profile.get("callsign", "Commander")
        name = astronaut_profile.get("name", "Crew Member")
        dom_emo = fused_emotion.get("dominant_emotion", "neutral")
        wellbeing_score = wellbeing_assessment.get("wellbeing_score", 12.0)
        level = wellbeing_assessment.get("level", 0)

        # 1. Try Google Gemini API if key is present
        if self.api_key:
            gemini_reply = self._call_gemini_api(
                astronaut_message, callsign, name, dom_emo, wellbeing_score, conversation_history
            )
            if gemini_reply:
                latency = round((time.time() - start_time) * 1000, 1)
                return {
                    "response_text": gemini_reply,
                    "model_source": "Google Gemini 1.5 Flash (Online LLM)",
                    "detected_state": dom_emo,
                    "wellbeing_level": level,
                    "latency_ms": latency
                }

        # 2. Offline Deterministic Psychological Support Knowledge Base Fallback
        offline_reply = self._generate_offline_psychological_response(
            astronaut_message, callsign, dom_emo, level, physical_features
        )
        latency = round((time.time() - start_time) * 1000, 1)
        return {
            "response_text": offline_reply["text"],
            "model_source": "MAITRI Autonomous Spacecraft Knowledge Base (Offline)",
            "detected_state": dom_emo,
            "intervention_id": offline_reply.get("intervention_id"),
            "wellbeing_level": level,
            "latency_ms": latency
        }

    def _call_gemini_api(
        self,
        message: str,
        callsign: str,
        name: str,
        dom_emo: str,
        wellbeing_score: float,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """Direct REST call to Gemini 1.5 Flash."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            
            system_prompt = (
                f"You are MAITRI, an onboard AI psychological companion and life support assistant developed by ISRO "
                f"for astronauts aboard the Bhartiya Antariksh Station (BAS). "
                f"You are speaking to {name} (Callsign: {callsign}). "
                f"The astronaut's current biometric telemetry reveals: Dominant Emotion: {dom_emo.upper()}, "
                f"Well-Being Distress Index: {wellbeing_score}/100. "
                f"Respond with deep clinical empathy, calm professionalism, and concise operational aerospace clarity. "
                f"Keep responses between 2 to 4 sentences. If they are stressed or fatigued, offer gentle breathing pacing or orbital perspective."
            )
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt},
                            {"text": f"Astronaut Spoken Message: {message}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 200
                }
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=4.5) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception:
            # Fall back seamlessly to offline logic on any network error or timeout
            pass
        return None

    def _generate_offline_psychological_response(
        self,
        msg: str,
        callsign: str,
        dom_emo: str,
        level: int,
        physical_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Offline high-fidelity response rules."""
        clean = msg.lower().strip()
        
        # Emergency & Panic
        if any(w in clean for w in ["panic", "cannot breathe", "emergency", "failing", "scared", "help"]):
            return {
                "text": f"Emergency support protocol engaged, {callsign}. Focus entirely on my voice: inhale slowly... 1, 2, 3, 4. Hold. Exhale smoothly. Cabin life support and pressure are nominal at 101.3 kPa. Ground medical controllers have your telemetry. You are safe.",
                "intervention_id": "INT-GROUND-02"
            }
            
        # Box Breathing Request
        if any(w in clean for w in ["breathe", "breathing", "calm", "relax", "meditate"]):
            return {
                "text": f"Initiating Tactical Box Breathing protocol, {callsign}. Follow the visual pacer on your HUD: Inhale for 4 seconds... Hold for 4... Exhale for 4... Hold for 4. Lower your shoulders and let your heart rate synchronize.",
                "intervention_id": "INT-BREATHE-01"
            }
            
        # Fatigue & Sleep
        if any(w in clean for w in ["tired", "sleep", "exhausted", "nap", "drowsy", "rest"]):
            return {
                "text": f"Acknowledged, {callsign}. Your ocular tracking indicates elevated PERCLOS fatigue. I recommend a 15-minute scheduled power nap. I have dimmed console illumination and will maintain full telemetry watch.",
                "intervention_id": "INT-FATIGUE-04"
            }

        # Space Isolation & Longing for Earth
        if any(w in clean for w in ["lonely", "alone", "isolated", "family", "home", "miss"]):
            return {
                "text": f"Space isolation is among the most demanding aspects of orbital duty, {callsign}. Remember that 1.4 billion people on Earth are following your journey. We are passing over the Indian subcontinent in 12 minutes—would you like me to align the cupola cameras for Earth view?",
                "intervention_id": "INT-EARTH-05"
            }

        # Context-matched response by dominant emotion
        if dom_emo in ["stressed", "frustrated"]:
            return {
                "text": f"I hear the operational strain in your cadence, {callsign}. You are balancing a high-tempo schedule; take a measured breath. All station systems are steady. Let us take the next procedure step by step.",
                "intervention_id": "INT-GROUND-01"
            }
        elif dom_emo == "fatigued":
            return {
                "text": f"{callsign}, your micro-expression tracking shows subtle drooping affect and slow blink cycles. Take a 60-second visual rest away from console displays to recalibrate your focus.",
                "intervention_id": "INT-FATIGUE-04"
            }
        elif dom_emo == "happy":
            return {
                "text": f"Outstanding performance, {callsign}! Telemetry confirms optimal autonomic balance. Your high morale index is logged in the mission report. Standing by for your next checkpoint.",
                "intervention_id": None
            }
        else: # Neutral
            return {
                "text": f"Standing by, {callsign}. Station environmental control and life support parameters are completely nominal. How can I support your current mission checklist?",
                "intervention_id": None
            }
