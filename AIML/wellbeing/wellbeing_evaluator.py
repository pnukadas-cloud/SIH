"""
AIML — Current Well-Being & Risk Evaluator Module
Calculates the authentic astronaut well-being index (0-100 scale) and 4-tier ISRO
flight status from multi-modal sensor inputs with transparent mathematical breakdown.
"""

from typing import Dict, Any

class WellBeingEvaluator:
    def __init__(self):
        self.tiers = {
            0: {"name": "Level 0: Nominal / Rested", "color": "emerald", "range": (0, 30)},
            1: {"name": "Level 1: Mild Cognitive Load", "color": "amber", "range": (31, 50)},
            2: {"name": "Level 2: Moderate Distress", "color": "orange", "range": (51, 70)},
            3: {"name": "Level 3: Acute Critical Distress", "range": (71, 100), "color": "red"}
        }

    def evaluate_wellbeing(
        self,
        fused_emotion: Dict[str, Any],
        physical_features: Dict[str, Any],
        acoustic_features: Dict[str, Any],
        baseline_vitals: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Compute deterministic well-being score (0 to 100 scale)
        where 0 = Peak Operational Calm, 100 = Critical Physiological/Psychological Distress.
        """
        valence = fused_emotion.get("valence", 0.0)      # -1.0 to +1.0
        arousal = fused_emotion.get("arousal", 0.2)      #  0.0 to  1.0
        dominant = fused_emotion.get("dominant_emotion", "neutral")
        
        perclos = physical_features.get("perclos_percentage", 4.0) # e.g. 4.0%
        vocal_tension = acoustic_features.get("vocal_tension_score", 0.1) # 0.0 to 1.0
        pitch_f0 = acoustic_features.get("pitch_f0_hz", 135.0)

        # Baseline comparison
        base_f0 = 135.0
        if baseline_vitals and "mean_f0_pitch_hz" in baseline_vitals:
            base_f0 = float(baseline_vitals["mean_f0_pitch_hz"])

        # 1. Negative Affect Penalty (0 to 35 pts)
        # Valence from +1.0 to -1.0 -> maps to 0 to 35
        affect_penalty = max(0.0, (-valence) * 25.0) + (10.0 if dominant in ["stressed", "frustrated", "anxious"] else 0.0)
        affect_penalty = min(35.0, affect_penalty)

        # 2. Ocular & Circadian Fatigue Contribution (0 to 30 pts)
        # PERCLOS > 12% is clinical threshold for microsleep in aerospace medicine
        fatigue_score = min(30.0, (perclos / 15.0) * 20.0 + (10.0 if dominant == "fatigued" else 0.0))

        # 3. Autonomic Tension Contribution (0 to 25 pts)
        pitch_shift = max(0.0, (pitch_f0 - base_f0) / max(10.0, base_f0))
        tension_score = min(25.0, (vocal_tension * 15.0) + (pitch_shift * 10.0))

        # 4. Cross-Modal Discordance Multiplier (10 pts if detected)
        discordance_penalty = 10.0 if fused_emotion.get("cross_modal_discordance", False) else 0.0

        # Raw Composite Score (0 to 100)
        raw_score = affect_penalty + fatigue_score + tension_score + discordance_penalty
        wellbeing_score = round(float(min(100.0, max(5.0, raw_score))), 1)

        # Determine Tier
        if wellbeing_score <= 30.0:
            level = 0
            tier_name = "Level 0: Nominal / Rested"
            status_color = "emerald"
            recommendation = "All biometrics nominal. Continue routine mission timeline."
        elif wellbeing_score <= 50.0:
            level = 1
            tier_name = "Level 1: Mild Cognitive Load"
            status_color = "amber"
            recommendation = "Elevated cognitive tension detected. Tactical box breathing advised."
        elif wellbeing_score <= 70.0:
            level = 2
            tier_name = "Level 2: Moderate Distress / High Fatigue"
            status_color = "orange"
            recommendation = "Significant psychological strain detected. Initiate clinical intervention and notify flight surgeon."
        else:
            level = 3
            tier_name = "Level 3: Acute Emergency Distress"
            status_color = "red"
            recommendation = "Emergency stress protocol engaged. Priority S-Band medical alert transmitted to ground station."

        return {
            "wellbeing_score": wellbeing_score,
            "level": level,
            "tier_name": tier_name,
            "status_color": status_color,
            "recommendation": recommendation,
            "components": {
                "negative_affect_penalty": round(affect_penalty, 1),
                "ocular_fatigue_score": round(fatigue_score, 1),
                "autonomic_tension_score": round(tension_score, 1),
                "cross_modal_discordance_penalty": round(discordance_penalty, 1)
            },
            "formula_explanation": "Score = Negative_Affect(35%) + Ocular_Fatigue(30%) + Autonomic_Tension(25%) + Discordance(10%)"
        }
