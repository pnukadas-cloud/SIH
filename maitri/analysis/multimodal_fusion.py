"""
MAITRI — Attention-Weighted Multimodal Late Fusion Module
Integrates Facial Emotion, Speech Emotion, and Linguistic Sentiment streams
with dynamic confidence weighting, modality dropout tolerance, and discordance detection.
"""

import numpy as np
from typing import Dict, Any
from maitri.config import EMOTIONS, DEFAULT_FUSION_WEIGHTS

class MultimodalFusionEngine:
    def __init__(self):
        self.default_weights = DEFAULT_FUSION_WEIGHTS
        self.emotions = EMOTIONS
        
    def fuse(self, fer_result: Dict[str, Any], ser_result: Dict[str, Any], text_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform attention-weighted late fusion across all 3 emotion modalities.
        """
        # Determine modality activity and raw confidence
        w_face = self.default_weights["facial"] if fer_result.get("modality_active", False) else 0.0
        w_voice = self.default_weights["speech"] if ser_result.get("modality_active", False) else 0.0
        w_text = self.default_weights["linguistic"] if text_result.get("modality_active", False) else 0.0
        
        # Factor in prediction confidence
        w_face *= fer_result.get("confidence", 0.5)
        w_voice *= ser_result.get("confidence", 0.5)
        w_text *= text_result.get("confidence", 0.5)
        
        total_weight = w_face + w_voice + w_text
        if total_weight > 1e-4:
            alpha = w_face / total_weight
            beta = w_voice / total_weight
            gamma = w_text / total_weight
        else:
            # All modalities idle — fallback uniform weights
            alpha, beta, gamma = 0.4, 0.35, 0.25
            
        p_face = fer_result.get("probabilities", {})
        p_voice = ser_result.get("probabilities", {})
        p_text = text_result.get("probabilities", {})
        
        # Unified Softmax Probabilities
        fused_probs = {}
        for emo in self.emotions:
            val = (alpha * p_face.get(emo, 0.14) +
                   beta * p_voice.get(emo, 0.14) +
                   gamma * p_text.get(emo, 0.14))
            fused_probs[emo] = round(float(val), 4)
            
        # Normalize
        total_p = sum(fused_probs.values())
        if total_p > 0:
            fused_probs = {k: round(v / total_p, 4) for k, v in fused_probs.items()}
            
        dominant = max(fused_probs.items(), key=lambda x: x[1])
        
        # Fused Valence & Arousal
        fused_valence = round(float(
            (alpha * fer_result.get("valence", 0.0)) +
            (beta * ser_result.get("valence", 0.0)) +
            (gamma * text_result.get("valence", 0.0))
        ), 3)
        
        fused_arousal = round(float(
            (alpha * fer_result.get("arousal", 0.2)) +
            (beta * ser_result.get("arousal", 0.2)) +
            (gamma * text_result.get("arousal", 0.2))
        ), 3)
        
        # Cross-Modal Discordance / Masking Analysis
        # Example: Smiling face (high happy prob) vs tense voice (high stressed/anxious prob)
        is_discordant = False
        discordance_reason = None
        
        if fer_result.get("modality_active", False) and ser_result.get("modality_active", False):
            face_dom = fer_result.get("dominant_emotion", "neutral")
            voice_dom = ser_result.get("dominant_emotion", "neutral")
            
            if face_dom == "happy" and voice_dom in ["stressed", "anxious", "frustrated"]:
                is_discordant = True
                discordance_reason = "Masked Stress: Astronaut is smiling facially, but vocal biomarkers exhibit acute autonomic tension."
            elif face_dom in ["sad", "fatigued"] and text_result.get("dominant_emotion") == "happy":
                is_discordant = True
                discordance_reason = "Verbal Minimization: Verbal statements indicate nominal state, but facial cues demonstrate exhaustion/flat affect."

        return {
            "fused_probabilities": fused_probs,
            "dominant_emotion": dominant[0],
            "confidence": round(dominant[1], 3),
            "valence": fused_valence,
            "arousal": fused_arousal,
            "attention_weights": {
                "facial_alpha": round(alpha, 3),
                "speech_beta": round(beta, 3),
                "linguistic_gamma": round(gamma, 3)
            },
            "cross_modal_discordance": is_discordant,
            "discordance_reason": discordance_reason
        }
