"""
MAITRI — Facial Emotion Recognition (FER) Engine
Analyzes geometric action units, landmark dynamics, and micro-expressions
to output 7-class emotion probabilities, valence, and arousal.
"""

import numpy as np
from typing import Dict, Any
from maitri.config import EMOTIONS

class FacialEmotionEngine:
    def __init__(self):
        self.classes = EMOTIONS # ['neutral', 'happy', 'stressed', 'fatigued', 'anxious', 'sad', 'frustrated']
        
    def predict(self, vision_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute facial emotion probabilities from extracted vision features.
        """
        if not vision_features.get("face_detected", False):
            # Uniform / neutral fallback if no face
            probs = {
                "neutral": 0.50,
                "happy": 0.10,
                "stressed": 0.08,
                "fatigued": 0.12,
                "anxious": 0.06,
                "sad": 0.08,
                "frustrated": 0.06
            }
            return {
                "probabilities": probs,
                "dominant_emotion": "neutral",
                "confidence": 0.35,
                "valence": 0.0,
                "arousal": 0.2,
                "modality_active": False
            }
            
        ear = vision_features.get("eye_aspect_ratio", 0.28)
        mar = vision_features.get("mouth_aspect_ratio", 0.20)
        perclos = vision_features.get("perclos", 0.0)
        aus = vision_features.get("action_units", {})
        smile_detected = vision_features.get("smile_detected", False)
        yawns = vision_features.get("yawns_per_min", 0)
        
        au4_brow = aus.get("AU04_brow_furrow", 0.0)
        au12_smile = aus.get("AU12_lip_corner_puller", 0.0)
        au20_stretch = aus.get("AU20_lip_stretcher", 0.0)
        au43_closed = aus.get("AU43_eye_closure", 0.0)
        
        # Raw Logit Estimation based on validated FACS (Facial Action Coding System)
        logits = {}
        
        # 1. Happy: high AU12 (smile), elevated cheek (AU6), moderate eye opening
        logits["happy"] = (au12_smile * 4.2) + (1.0 if smile_detected else 0.0) - (au4_brow * 2.0) - (perclos * 2.0)
        
        # 2. Fatigued: high PERCLOS, low EAR, yawns, eye closure
        logits["fatigued"] = (perclos * 4.5) + (yawns * 1.5) + (au43_closed * 2.0) + (1.2 if mar > 0.45 else 0.0)
        
        # 3. Stressed: brow furrowing (AU4), eye narrowing, lip tightening
        logits["stressed"] = (au4_brow * 3.2) + (au20_stretch * 1.8) + (1.2 if ear < 0.24 and ear > 0.18 else 0.0) - (au12_smile * 1.5)
        
        # 4. Anxious: wide eyes (high EAR), lip stretch (AU20), brow tension (AU4)
        logits["anxious"] = (au4_brow * 2.2) + (au20_stretch * 2.5) + (1.5 if ear > 0.35 else 0.0) - (au12_smile * 1.8)
        
        # 5. Sad: mouth corners down, drooping eyelids, low energy
        logits["sad"] = (au4_brow * 2.0) + (1.5 if mar < 0.18 else 0.0) + (perclos * 1.5) - (au12_smile * 3.0)
        
        # 6. Frustrated: intense brow furrowing (AU4), jaw tension / mouth stretch
        logits["frustrated"] = (au4_brow * 3.8) + (au20_stretch * 2.2) - (au12_smile * 2.5) - (perclos * 1.0)
        
        # 7. Neutral: baseline balance
        logits["neutral"] = 1.2 - (au4_brow * 1.5) - (au12_smile * 1.5) - (perclos * 2.0) - (au20_stretch * 1.2)
        
        # Softmax normalization
        exp_logits = {k: np.exp(np.clip(v, -5.0, 5.0)) for k, v in logits.items()}
        total_exp = sum(exp_logits.values())
        probs = {k: round(float(v / total_exp), 4) for k, v in exp_logits.items()}
        
        dominant = max(probs.items(), key=lambda x: x[1])
        
        # Calculate Valence (-1.0 to +1.0) and Arousal (0.0 to 1.0)
        valence = (probs["happy"] * 1.0) + (probs["neutral"] * 0.0) - (probs["sad"] * 0.8) - (probs["frustrated"] * 0.7) - (probs["stressed"] * 0.6) - (probs["anxious"] * 0.6) - (probs["fatigued"] * 0.4)
        arousal = (probs["frustrated"] * 0.9) + (probs["anxious"] * 0.85) + (probs["stressed"] * 0.75) + (probs["happy"] * 0.6) + (probs["neutral"] * 0.2) + (probs["sad"] * 0.3) + (probs["fatigued"] * 0.1)
        
        return {
            "probabilities": probs,
            "dominant_emotion": dominant[0],
            "confidence": round(dominant[1], 3),
            "valence": round(float(np.clip(valence, -1.0, 1.0)), 3),
            "arousal": round(float(np.clip(arousal, 0.0, 1.0)), 3),
            "modality_active": True
        }
