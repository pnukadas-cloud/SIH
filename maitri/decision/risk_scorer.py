"""
MAITRI — 4-Tier Risk Escalation Engine & Scorer
Computes composite astronaut well-being risk score (0-100) and maps to
actionable intervention and ground station escalation tiers.
"""

import numpy as np
from typing import Dict, Any

class RiskScorer:
    def __init__(self):
        # Base severity penalties per emotion (0.0 - 100.0)
        self.emotion_severities = {
            "neutral": 5.0,
            "happy": 0.0,
            "fatigued": 48.0,
            "stressed": 58.0,
            "anxious": 68.0,
            "sad": 52.0,
            "frustrated": 64.0
        }
        
    def calculate_risk(self, fused_data: Dict[str, Any], physical_data: Dict[str, Any], state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute holistic risk score (0.0 to 100.0) and assign risk level tier.
        """
        dom_emo = fused_data.get("dominant_emotion", "neutral")
        confidence = fused_data.get("confidence", 0.5)
        is_discordant = fused_data.get("cross_modal_discordance", False)
        
        duration_secs = state_data.get("duration_in_state_seconds", 0.0)
        volatility = state_data.get("volatility_index", 0.1)
        trend = state_data.get("trajectory_trend", "stable")
        
        physical_score = physical_data.get("composite_physical_distress_score", 0.0)
        microsleep_risk = physical_data.get("microsleep_risk_score", 0.0)
        
        # 1. Base Emotion Severity Component (0 - 100)
        base_emo_score = self.emotion_severities.get(dom_emo, 20.0) * confidence
        
        # 2. State Duration Amplifier (Prolonged negative state increases risk)
        # e.g., in 'anxious' for > 10 minutes (600s) raises risk significantly
        if dom_emo in ["stressed", "anxious", "frustrated", "sad", "fatigued"]:
            duration_multiplier = min(1.6, 1.0 + (duration_secs / 600.0) * 0.5)
        else:
            duration_multiplier = 1.0
            
        # 3. Trend Adjustment
        trend_modifier = 0.0
        if trend == "worsening":
            trend_modifier = 12.0
        elif trend == "improving":
            trend_modifier = -10.0
            
        # 4. Volatility Penalty (Erratic emotional swings in space indicate decompensation)
        volatility_penalty = volatility * 18.0
        
        # 5. Discordance / Masked Distress Penalty
        discordance_penalty = 15.0 if is_discordant else 0.0
        
        # 6. Physical Distress Integration
        physical_component = physical_score * 0.45
        
        # Composite Risk Formula
        raw_risk = (
            (base_emo_score * duration_multiplier * 0.40) +
            physical_component +
            trend_modifier +
            volatility_penalty +
            discordance_penalty +
            (microsleep_risk * 0.20)
        )
        
        final_risk_score = float(np.clip(raw_risk, 0.0, 100.0))
        
        # Tier Classification
        if final_risk_score >= 71.0:
            risk_level = 3
            tier_name = "LEVEL 3: CRITICAL ALARM"
            status_badge = "CRITICAL"
            action_protocol = "CRISIS INTERVENTION & IMMEDIATE GROUND CONTROL DISPATCH"
            recommendation = "Deploy emergency sensory grounding protocol. Signal Flight Surgeon console at Ground Station immediately."
            color = "#ef4444"
        elif final_risk_score >= 51.0:
            risk_level = 2
            tier_name = "LEVEL 2: MODERATE RISK"
            status_badge = "MODERATE"
            action_protocol = "ACTIVE INTERVENTION & GROUND STATION QUEUE"
            recommendation = "Initiate guided tactical breathing/CBT protocol. Log telemetry alert to mission ground telemetry buffer."
            color = "#f97316"
        elif final_risk_score >= 31.0:
            risk_level = 1
            tier_name = "LEVEL 1: MILD CONCERN"
            status_badge = "MILD"
            action_protocol = "PROACTIVE COMPANION CHECK-IN"
            recommendation = "Engage astronaut with short conversational check-in and situational support."
            color = "#f59e0b"
        else:
            risk_level = 0
            tier_name = "LEVEL 0: NOMINAL"
            status_badge = "NOMINAL"
            action_protocol = "PASSIVE TELEMETRY MONITORING"
            recommendation = "Astronaut emotional and physical biomarkers within nominal baseline tolerance."
            color = "#10b981"
            
        return {
            "risk_score": round(final_risk_score, 1),
            "risk_level": risk_level,
            "tier_name": tier_name,
            "status_badge": status_badge,
            "action_protocol": action_protocol,
            "recommendation": recommendation,
            "color_hex": color,
            "components": {
                "base_emotion_score": round(base_emo_score, 1),
                "physical_distress_contribution": round(physical_component, 1),
                "duration_amplifier": round(duration_multiplier, 2),
                "trend_adjustment": round(trend_modifier, 1),
                "volatility_penalty": round(volatility_penalty, 1),
                "discordance_penalty": round(discordance_penalty, 1)
            }
        }
