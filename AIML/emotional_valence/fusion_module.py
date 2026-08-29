"""
AIML — Attention-Weighted Multimodal Late Fusion & Valence Module
Combines facial, acoustic, and linguistic emotion streams to compute unified
Valence (-1.0 to +1.0) and Arousal (0.0 to 1.0) with cross-modal discordance detection.
"""

import numpy as np
from typing import Dict, Any

class EmotionalValenceFusionModule:
    def __init__(self):
        self.emotions = ['neutral', 'happy', 'stressed', 'fatigued', 'anxious', 'sad', 'frustrated']
        self.default_weights = {"facial": 0.45, "speech": 0.35, "linguistic": 0.20}

    def fuse(self, fer_result: Dict[str, Any], ser_result: Dict[str, Any], text_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic confidence-weighted late fusion across all active modalities."""
        # Calculate dynamic attention weights based on activity & confidence
        w_face = self.default_weights["facial"] * fer_result.get("confidence", 0.5) if fer_result.get("modality_active", False) else 0.0
        w_voice = self.default_weights["speech"] * ser_result.get("confidence", 0.5) if ser_result.get("modality_active", False) else 0.0
        w_text = self.default_weights["linguistic"] * text_result.get("confidence", 0.5) if text_result.get("modality_active", False) else 0.0

        total_weight = w_face + w_voice + w_text
        if total_weight > 1e-4:
            alpha = w_face / total_weight
            beta = w_voice / total_weight
            gamma = w_text / total_weight
        else:
            # All modalities idle or silent
            alpha, beta, gamma = 0.40, 0.35, 0.25

        p_face = fer_result.get("probabilities", {})
        p_voice = ser_result.get("probabilities", {})
        p_text = text_result.get("probabilities", {})

        # Compute fused softmax probability distribution
        fused_probs = {}
        for emo in self.emotions:
            val = (alpha * p_face.get(emo, 0.14) +
                   beta * p_voice.get(emo, 0.14) +
                   gamma * p_text.get(emo, 0.14))
            fused_probs[emo] = round(float(val), 4)

        tot_p = sum(fused_probs.values())
        if tot_p > 0:
            fused_probs = {k: round(v / tot_p, 4) for k, v in fused_probs.items()}

        dominant = max(fused_probs.items(), key=lambda x: x[1])

        # Fused Valence (-1.0 to +1.0)
        # Negative: Sad, Frustrated, Stressed, Anxious, Fatigued
        # Neutral: Neutral
        # Positive: Happy / Calm
        val_face = fer_result.get("valence", 0.0)
        val_voice = ser_result.get("valence", 0.0)
        val_text = text_result.get("valence", 0.0)
        fused_valence = round(float(np.clip((alpha * val_face) + (beta * val_voice) + (gamma * val_text), -1.0, 1.0)), 3)

        # Fused Arousal (0.0 to 1.0)
        aro_face = fer_result.get("arousal", 0.2)
        aro_voice = ser_result.get("arousal", 0.2)
        aro_text = text_result.get("arousal", 0.2)
        fused_arousal = round(float(np.clip((alpha * aro_face) + (beta * aro_voice) + (gamma * aro_text), 0.0, 1.0)), 3)

        # Cross-Modal Discordance / Masking Analysis
        is_discordant = False
        discordance_reason = None
        
        if fer_result.get("modality_active", False) and ser_result.get("modality_active", False):
            f_dom = fer_result.get("dominant_emotion", "neutral")
            v_dom = ser_result.get("dominant_emotion", "neutral")
            if f_dom == "happy" and v_dom in ["stressed", "anxious", "frustrated"]:
                is_discordant = True
                discordance_reason = "Masked Stress: Astronaut is smiling facially, but vocal biomarkers exhibit acute autonomic tension."
            elif f_dom in ["sad", "fatigued"] and text_result.get("dominant_emotion") == "happy":
                is_discordant = True
                discordance_reason = "Verbal Minimization: Verbal transcript indicates nominal state, but facial cues exhibit drooping affect/exhaustion."

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
