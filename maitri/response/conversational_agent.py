"""
MAITRI — Empathetic Spacecraft Conversational AI Engine
Offline conversational intelligence providing psychological companionship,
operational de-escalation, and situation-aware astronaut support.
"""

import time
import re
from typing import Dict, Any, List, Optional
from maitri.config import SYSTEM_NAME, SPACE_STATION

class ConversationalAgent:
    def __init__(self):
        self.system_name = SYSTEM_NAME
        self.station_name = SPACE_STATION
        
        # Dialogue templates categorized by situation & state
        self.response_library = {
            "stressed": [
                "I hear the tension in your voice, {callsign}. You are managing complex orbital operations; take a slow breath. What is the immediate checkpoint in your procedure?",
                "Telemetry indicates elevated cognitive load, {callsign}. Let us break this down into singular steps. I have verified life support telemetry—the station is stable.",
                "Let us pause for 30 seconds, {callsign}. High-tempo drills are demanding, but your baseline capability is exceptional. Inhale deeply, reset, and proceed on your mark."
            ],
            "anxious": [
                "{callsign}, I detect rapid vocal jitter and elevated arousal. You are safe aboard {station}. Let us focus on what is right in front of you. Name your next switch.",
                "I am right here with you, {callsign}. The telemetry is completely green. Let us synchronize a short 4-second box breathing cycle to calm your autonomic nervous system.",
                "Acknowledged, {callsign}. Feeling apprehension in microgravity is a natural biological reflex. Ground control has optical tracking and I am monitoring all hull sensors."
            ],
            "fatigued": [
                "Your eye closure telemetry indicates severe fatigue, {callsign}. Reaction latency is climbing. Can we schedule a 15-minute micro-nap before the next airlock ingress?",
                "I notice consecutive yawns and drooping eyelids, {callsign}. Sustained circadian shifts take a heavy toll. Dimming your console HUD to amber mode now.",
                "Commander, operational safety requires rested cognition. Let me take over non-critical telemetry monitoring while you perform a 5-minute eye relaxation exercise."
            ],
            "sad": [
                "Space isolation is one of the toughest challenges of long-duration missions, {callsign}. Remember that your mission is inspiring millions back home on Earth. Would you like me to play family audio uplinks?",
                "You have been working tirelessly in this module, {callsign}. We are passing over the Himalayas in 8 minutes—the view from the cupola will be breathtaking. Would you like to look?",
                "I am always here to listen, {callsign}. You don't have to carry the mission pressure alone. Tell me what is on your mind."
            ],
            "frustrated": [
                "I understand how exasperating equipment glitches can be in zero-G, {callsign}. Let us step back from the console for 20 seconds and review the schematic objectively.",
                "Deep breath, {callsign}. Frustration narrows operational bandwidth. What specific sub-system is failing to respond as expected?",
                "Recognizing the difficulty, {callsign}. Let us recalibrate the sensor baseline together. We will solve this step-by-step."
            ],
            "happy": [
                "Outstanding work, {callsign}! Telemetry confirms flawless execution. Your focus and morale are reflecting clearly in the mission logs.",
                "Copy that, {callsign}! Excellent progress on today's flight schedule. Morale index is optimal.",
                "Wonderful to see you in high spirits, {callsign}. It makes every orbital rotation smoother. Standing by for next checklist items."
            ],
            "neutral": [
                "Standing by, {callsign}. All station systems aboard {station} are operating within nominal parameters. How can I assist your current workflow?",
                "MAITRI online and listening, {callsign}. Telemetry is green across all sectors.",
                "Ready when you are, {callsign}. Life support, orbital path, and communication links are nominal."
            ]
        }
        
    def generate_response(
        self,
        astronaut_message: str,
        astronaut_profile: Dict[str, Any],
        fused_emotion: Dict[str, Any],
        physical_distress: Dict[str, Any],
        risk_data: Dict[str, Any],
        selected_intervention: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate contextual, empathetic response matching astronaut voice and emotional state.
        """
        callsign = astronaut_profile.get("callsign", "Astronaut")
        name = astronaut_profile.get("name", "Crew Member")
        dom_emo = fused_emotion.get("dominant_emotion", "neutral")
        risk_level = risk_data.get("risk_level", 0)
        
        clean_input = astronaut_message.lower().strip()
        
        # 1. Check for Direct Emergency / Distress Queries
        if any(w in clean_input for w in ["help", "cannot breathe", "panic", "emergency", "failing", "scared"]):
            text = (
                f"Emergency support protocol engaged, {callsign}. Focus entirely on my voice: inhale slowly... 1, 2, 3, 4. "
                f"Hold. Exhale smoothly. The life support and cabin pressure are completely steady at 101.3 kPa. "
                f"I am alerting the ground surgeon right now. You are safe."
            )
            intervention_trigger = "INT-GROUND-02"
            
        # 2. Check for Specific Breathing / Intervention Requests
        elif any(w in clean_input for w in ["breathe", "breathing", "calm down", "relax", "exercise"]):
            text = (
                f"Initiating Box Breathing protocol with you now, {callsign}. "
                f"Follow the visual pacer on your HUD: Inhale for 4 seconds... Hold for 4... Exhale for 4... Hold for 4. "
                f"Keep your shoulders dropped and body relaxed against the restraint."
            )
            intervention_trigger = "INT-BREATHE-01"
            
        # 3. Check for Sleep / Fatigue Queries
        elif any(w in clean_input for w in ["tired", "sleep", "exhausted", "nap", "rest"]):
            text = (
                f"Acknowledged, {callsign}. Your PERCLOS ocular tracking confirms fatigue buildup. "
                f"I recommend a 15-minute scheduled power nap. I have adjusted ambient cabin lighting to 480nm amber "
                f"and will log full telemetry while you rest."
            )
            intervention_trigger = "INT-FATIGUE-04"
            
        # 4. Check for Station Status / Operational Check
        elif any(w in clean_input for w in ["status", "station", "orbit", "systems", "report", "telemetry"]):
            text = (
                f"All systems aboard {self.station_name} are nominal, {callsign}. Cabin oxygen is 21.2%, "
                f"CO2 scrubber is nominal, and orbital trajectory is tracking at 410 km altitude. "
                f"Your physiological telemetry indicates current state is {dom_emo.upper()} with risk level {risk_level}."
            )
            intervention_trigger = None
            
        # Delegate to intelligent MaitriCompanionAI with semantic space QA dataset
        from AIML.maitri.companion_ai import MaitriCompanionAI
        companion = MaitriCompanionAI()
        res = companion.generate_response(
            astronaut_message=astronaut_message,
            astronaut_profile=astronaut_profile,
            fused_emotion=fused_emotion,
            physical_features=physical_distress or {},
            wellbeing_assessment={"wellbeing_score": risk_data.get("risk_score", 12.0), "level": risk_level}
        )
        return {
            "response_text": res["response_text"],
            "detected_state": dom_emo,
            "risk_level": risk_level,
            "intervention_id": res.get("intervention_id"),
            "verbal_guidance": selected_intervention.get("verbal_guidance") if selected_intervention and risk_level >= 2 else None,
            "timestamp": time.time()
        }
