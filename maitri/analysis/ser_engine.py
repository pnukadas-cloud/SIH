"""
MAITRI — Speech Emotion Recognition (SER) Engine
Classifies emotion probabilities, acoustic valence, and arousal
from vocal prosody, pitch dynamics, spectral centroids, and jitter.
"""

import numpy as np
from typing import Dict, Any
from maitri.config import EMOTIONS

class SpeechEmotionEngine:
    def __init__(self):
        self.classes = EMOTIONS
        
    def predict(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute speech emotion probabilities from acoustic features.
        """
        is_speech = audio_features.get("is_speech_active", False)
        
        if not is_speech:
            probs = {
                "neutral": 0.60,
                "happy": 0.08,
                "stressed": 0.08,
                "fatigued": 0.10,
                "anxious": 0.05,
                "sad": 0.05,
                "frustrated": 0.04
            }
            return {
                "probabilities": probs,
                "dominant_emotion": "neutral",
                "confidence": 0.40,
                "valence": 0.0,
                "arousal": 0.15,
                "modality_active": False
            }
            
        pitch = audio_features.get("pitch_f0_hz", 140.0)
        energy = audio_features.get("rms_energy", 0.05)
        vocal_tension = audio_features.get("vocal_tension_score", 0.1)
        jitter = audio_features.get("vocal_jitter", 0.02)
        spectral_centroid = audio_features.get("spectral_centroid_hz", 1200.0)
        
        logits = {}
        
        # 1. Stressed: elevated pitch, high vocal tension, elevated energy
        logits["stressed"] = (vocal_tension * 3.5) + (1.5 if pitch > 210 else 0.0) + (1.0 if energy > 0.08 else 0.0)
        
        # 2. Anxious: pitch instability / jitter, high spectral centroid, rapid shallow energy
        logits["anxious"] = (jitter * 25.0) + (vocal_tension * 2.8) + (1.5 if spectral_centroid > 2200 else 0.0)
        
        # 3. Frustrated: harsh high energy, high pitch peaks, high spectral centroid
        logits["frustrated"] = (energy * 15.0) + (vocal_tension * 2.2) + (1.8 if pitch > 230 else 0.0)
        
        # 4. Fatigued: low energy, monotonous low pitch, low spectral centroid
        logits["fatigued"] = (2.5 if energy < 0.03 else 0.0) + (1.8 if pitch < 120 and pitch > 60 else 0.0) + (1.5 if spectral_centroid < 1000 else 0.0)
        
        # 5. Sad: soft low energy, downward pitch contour, low tension
        logits["sad"] = (2.0 if energy < 0.035 else 0.0) + (1.5 if pitch < 130 and pitch > 60 else 0.0) - (vocal_tension * 1.5)
        
        # 6. Happy: dynamic pitch range, bright spectral centroid, moderate-high energy
        logits["happy"] = (1.8 if pitch > 160 and pitch < 240 else 0.0) + (1.5 if spectral_centroid > 1600 and spectral_centroid < 2600 else 0.0) + (1.0 if energy > 0.05 else 0.0) - (vocal_tension * 2.0)
        
        # 7. Neutral: balanced pitch, moderate energy, low tension
        logits["neutral"] = 1.5 - (vocal_tension * 2.5) - (1.0 if energy > 0.12 or energy < 0.02 else 0.0)
        
        # Softmax
        exp_logits = {k: np.exp(np.clip(v, -5.0, 5.0)) for k, v in logits.items()}
        total_exp = sum(exp_logits.values())
        probs = {k: round(float(v / total_exp), 4) for k, v in exp_logits.items()}
        
        dominant = max(probs.items(), key=lambda x: x[1])
        
        valence = (probs["happy"] * 1.0) + (probs["neutral"] * 0.0) - (probs["sad"] * 0.8) - (probs["frustrated"] * 0.7) - (probs["stressed"] * 0.6) - (probs["anxious"] * 0.6) - (probs["fatigued"] * 0.4)
        arousal = (probs["frustrated"] * 0.95) + (probs["anxious"] * 0.88) + (probs["stressed"] * 0.80) + (probs["happy"] * 0.65) + (probs["neutral"] * 0.20) + (probs["sad"] * 0.25) + (probs["fatigued"] * 0.10)
        
        return {
            "probabilities": probs,
            "dominant_emotion": dominant[0],
            "confidence": round(dominant[1], 3),
            "valence": round(float(np.clip(valence, -1.0, 1.0)), 3),
            "arousal": round(float(np.clip(arousal, 0.0, 1.0)), 3),
            "modality_active": True
        }
