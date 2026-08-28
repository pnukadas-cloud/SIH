"""
MAITRI — Linguistic Sentiment & Cognitive State Engine
Analyzes astronaut conversational transcripts for sentiment valence,
arousal, cognitive overload markers, isolation signals, and emotion probabilities.
"""

import re
from typing import Dict, Any, List
from maitri.config import EMOTIONS

class TextSentimentEngine:
    def __init__(self):
        # Curated affective & spaceflight psychological lexicons
        self.emotion_lexicons = {
            "stressed": [
                "pressure", "stressed", "overwhelmed", "deadline", "alarms", "failing",
                "urgent", "rushing", "heavy", "too much", "trouble", "cannot manage",
                "overloaded", "tight schedule", "behind schedule", "drill", "malfunction"
            ],
            "anxious": [
                "scared", "worried", "panic", "heart racing", "fear", "uncertain",
                "dread", "what if", "danger", "unstable", "anxious", "nervous", "shaking",
                "decompression", "radiation", "airlock", "thruster failure"
            ],
            "frustrated": [
                "annoyed", "frustrated", "angry", "broken", "stupid", "hate", "useless",
                "why is this", "irritated", "glitch", "stuck", "fed up", "refuse",
                "not working", "lagging", "waste of time", "impossible"
            ],
            "fatigued": [
                "tired", "exhausted", "sleepy", "drowsy", "drained", "no energy",
                "yawning", "cannot focus", "heavy eyes", "fatigue", "burnt out",
                "headache", "sleep deprived", "insomnia", "need rest", "worn out"
            ],
            "sad": [
                "lonely", "isolated", "miss my family", "miss home", "sad", "hopeless",
                "alone", "empty", "depressed", "nobody", "far away", "earth", "dark",
                "homesick", "crying", "disconnected"
            ],
            "happy": [
                "great", "awesome", "success", "happy", "docked", "green", "nominal",
                "wonderful", "beautiful", "excited", "proud", "good job", "ready",
                "flawless", "mission accomplished", "energized", "glad", "celebrate"
            ]
        }
        
        self.cognitive_overload_markers = [
            "cannot keep up", "too many alarms", "confused", "lost focus",
            "brain fog", "overloaded", "which switch", "disoriented"
        ]
        
        self.physical_complaint_markers = [
            "nausea", "headache", "back pain", "zero-g sickness", "dizziness",
            "muscle spasm", "eye strain", "stomach cramp", "cannot breathe"
        ]
        
        self.isolation_markers = [
            "miss my family", "miss home", "so far from earth", "feel alone",
            "endless darkness", "nobody understands", "trapped"
        ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze verbal transcript and compute affective and cognitive metrics.
        """
        if not text or not text.strip():
            probs = {
                "neutral": 0.70,
                "happy": 0.05,
                "stressed": 0.05,
                "fatigued": 0.05,
                "anxious": 0.05,
                "sad": 0.05,
                "frustrated": 0.05
            }
            return {
                "probabilities": probs,
                "dominant_emotion": "neutral",
                "confidence": 0.50,
                "valence": 0.0,
                "arousal": 0.1,
                "cognitive_overload": False,
                "physical_complaints": [],
                "isolation_detected": False,
                "modality_active": False
            }
            
        clean_text = text.lower()
        
        # Keyword scoring
        scores = {emo: 0.1 for emo in EMOTIONS}
        scores["neutral"] = 0.5
        
        for emo, words in self.emotion_lexicons.items():
            for word in words:
                if word in clean_text:
                    scores[emo] += 1.8
                    scores["neutral"] -= 0.3
                    
        # Exclamation & Punctuation markers (High Arousal)
        if "!" in text:
            scores["frustrated"] += 0.5
            scores["anxious"] += 0.5
            scores["happy"] += 0.5
            
        if "?" in text:
            scores["anxious"] += 0.4
            
        # Specific Clinical / Operational Flag Detections
        has_overload = any(m in clean_text for m in self.cognitive_overload_markers)
        detected_physical = [p for p in self.physical_complaint_markers if p in clean_text]
        has_isolation = any(i in clean_text for i in self.isolation_markers)
        
        if has_overload:
            scores["stressed"] += 2.5
            scores["anxious"] += 1.5
            
        if detected_physical:
            scores["fatigued"] += 2.0
            scores["stressed"] += 1.0
            
        if has_isolation:
            scores["sad"] += 3.0
            
        # Softmax normalization
        total = sum(max(0.01, v) for v in scores.values())
        probs = {k: round(float(max(0.01, v) / total), 4) for k, v in scores.items()}
        
        dominant = max(probs.items(), key=lambda x: x[1])
        
        # Calculate Valence and Arousal
        valence = (probs["happy"] * 1.0) + (probs["neutral"] * 0.0) - (probs["sad"] * 0.8) - (probs["frustrated"] * 0.7) - (probs["stressed"] * 0.6) - (probs["anxious"] * 0.6) - (probs["fatigued"] * 0.4)
        arousal = (probs["frustrated"] * 0.9) + (probs["anxious"] * 0.85) + (probs["stressed"] * 0.75) + (probs["happy"] * 0.6) + (probs["neutral"] * 0.2) + (probs["sad"] * 0.3) + (probs["fatigued"] * 0.1)
        
        return {
            "probabilities": probs,
            "dominant_emotion": dominant[0],
            "confidence": round(dominant[1], 3),
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3),
            "cognitive_overload": has_overload,
            "physical_complaints": detected_physical,
            "isolation_detected": has_isolation,
            "modality_active": True
        }
