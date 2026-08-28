"""
MAITRI — Physical Distress & Fatigue Detection Engine
Monitors physiological fatigue, PERCLOS eye closure, yawning frequency,
blink rate anomalies, pain grimaces, and vocal exhaustion biomarkers.
"""

import numpy as np
from typing import Dict, Any, List

class PhysicalDistressEngine:
    def __init__(self):
        pass
        
    def evaluate(self, vision_features: Dict[str, Any], audio_features: Dict[str, Any], text_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate overall physical distress and fatigue from multimodal sensors.
        """
        perclos = vision_features.get("perclos", 0.0)
        ear = vision_features.get("eye_aspect_ratio", 0.28)
        mar = vision_features.get("mouth_aspect_ratio", 0.20)
        blinks_per_min = vision_features.get("blinks_per_min", 16.0)
        yawns_per_min = vision_features.get("yawns_per_min", 0)
        aus = vision_features.get("action_units", {})
        
        vocal_tension = audio_features.get("vocal_tension_score", 0.0)
        vocal_jitter = audio_features.get("vocal_jitter", 0.0)
        audio_energy = audio_features.get("rms_energy", 0.05)
        
        physical_complaints = text_features.get("physical_complaints", [])
        
        # 1. PERCLOS & Microsleep Risk (0.0 to 100.0)
        # PERCLOS > 0.20 begins indicating drowsiness, > 0.35 indicates severe microsleep risk
        microsleep_risk = float(np.clip(perclos * 220.0, 0.0, 100.0))
        
        # 2. Yawn & Ocular Fatigue Score
        ocular_fatigue = float(np.clip((yawns_per_min * 22.0) + (microsleep_risk * 0.6) + (max(0.0, 0.22 - ear) * 300.0), 0.0, 100.0))
        
        # 3. Blink Rate Anomaly Score
        # Normal blink rate is 12-20 bpm. > 28 is acute ocular strain/stress. < 8 is fixation/severe drowsiness.
        if blinks_per_min > 26.0:
            blink_anomaly_score = min(100.0, (blinks_per_min - 26.0) * 5.0)
            blink_status = "Hyper-blinking (Cognitive/Ocular Strain)"
        elif blinks_per_min < 8.0 and vision_features.get("face_detected", False):
            blink_anomaly_score = (8.0 - blinks_per_min) * 10.0
            blink_status = "Hypo-blinking / Fixed Stare (Exhaustion Risk)"
        else:
            blink_anomaly_score = 0.0
            blink_status = "Nominal"
            
        # 4. Facial Pain & Grimacing Score (AU4 brow furrow + AU7 lid tight + AU20 lip stretch)
        au4 = aus.get("AU04_brow_furrow", 0.0)
        au20 = aus.get("AU20_lip_stretcher", 0.0)
        pain_grimace_score = float(np.clip((au4 * 40.0) + (au20 * 35.0) + (len(physical_complaints) * 25.0), 0.0, 100.0))
        
        # 5. Vocal Fatigue & Tremor Score
        vocal_fatigue_score = float(np.clip((vocal_jitter * 800.0) + (vocal_tension * 40.0) + (30.0 if audio_energy < 0.02 else 0.0), 0.0, 100.0))
        
        # Composite Physical Distress Score (0 - 100)
        weights = [0.30, 0.25, 0.15, 0.15, 0.15]
        composite_score = float(
            weights[0] * microsleep_risk +
            weights[1] * ocular_fatigue +
            weights[2] * blink_anomaly_score +
            weights[3] * pain_grimace_score +
            weights[4] * vocal_fatigue_score
        )
        composite_score = float(np.clip(composite_score, 0.0, 100.0))
        
        # Fatigue Classification
        if composite_score > 70.0 or perclos > 0.32:
            fatigue_level = "Severe Exhaustion (Critical Rest Required)"
            status_color = "red"
        elif composite_score > 45.0 or yawns_per_min >= 2:
            fatigue_level = "Moderate Fatigue (Micro-Rest Recommended)"
            status_color = "orange"
        elif composite_score > 25.0:
            fatigue_level = "Mild Drowsiness (Monitor Trend)"
            status_color = "yellow"
        else:
            fatigue_level = "Nominal / Rested"
            status_color = "green"
            
        return {
            "composite_physical_distress_score": round(composite_score, 1),
            "fatigue_level": fatigue_level,
            "status_color": status_color,
            "perclos_percentage": round(perclos * 100.0, 1),
            "microsleep_risk_score": round(microsleep_risk, 1),
            "ocular_fatigue_score": round(ocular_fatigue, 1),
            "blink_rate_bpm": round(blinks_per_min, 1),
            "blink_status": blink_status,
            "pain_grimace_score": round(pain_grimace_score, 1),
            "vocal_fatigue_score": round(vocal_fatigue_score, 1),
            "physical_complaints_reported": physical_complaints
        }
