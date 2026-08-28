"""
MAITRI — Temporal Emotion State Tracker
Maintains rolling temporal state windows, tracks state durations,
computes emotional volatility, and estimates stability trends.
"""

import time
import numpy as np
from typing import Dict, Any, List

class EmotionStateTracker:
    def __init__(self, max_history_seconds: float = 300.0):
        self.max_history_seconds = max_history_seconds
        self.state_history = [] # List of tuples (timestamp, fused_data, physical_data)
        self.current_state = "neutral"
        self.state_start_time = time.time()
        
    def update(self, fused_data: Dict[str, Any], physical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest new multimodal frame, update sliding window, and compute temporal metrics.
        """
        now = time.time()
        dom_emo = fused_data.get("dominant_emotion", "neutral")
        
        # State duration tracking
        if dom_emo != self.current_state:
            self.current_state = dom_emo
            self.state_start_time = now
            
        state_duration_seconds = now - self.state_start_time
        
        # Add to rolling history
        self.state_history.append((now, fused_data, physical_data))
        
        # Prune old samples
        cutoff = now - self.max_history_seconds
        self.state_history = [item for item in self.state_history if item[0] >= cutoff]
        
        # Compute Emotional Volatility Index (frequency of emotion shifts in rolling window)
        if len(self.state_history) > 10:
            emotions_in_window = [item[1].get("dominant_emotion") for item in self.state_history]
            transitions = sum(1 for i in range(len(emotions_in_window)-1) if emotions_in_window[i] != emotions_in_window[i+1])
            volatility_index = float(min(1.0, transitions / (len(emotions_in_window) * 0.4)))
        else:
            volatility_index = 0.1
            
        # Compute Emotional Trajectory (Valence trend over window)
        if len(self.state_history) >= 8:
            valences = [item[1].get("valence", 0.0) for item in self.state_history]
            x = np.arange(len(valences))
            slope, _ = np.polyfit(x, valences, 1)
            
            if slope > 0.005:
                trend = "improving"
                trend_desc = "Valence trending upward toward positive/stable baseline."
            elif slope < -0.005:
                trend = "worsening"
                trend_desc = "Valence trending downward into distress/fatigue."
            else:
                trend = "stable"
                trend_desc = "Emotional state remains steady."
        else:
            trend = "stable"
            trend_desc = "Calibrating baseline window..."
            
        # Rolling averages
        avg_valence = float(np.mean([item[1].get("valence", 0.0) for item in self.state_history])) if self.state_history else 0.0
        avg_arousal = float(np.mean([item[1].get("arousal", 0.0) for item in self.state_history])) if self.state_history else 0.0
        avg_physical_distress = float(np.mean([item[2].get("composite_physical_distress_score", 0.0) for item in self.state_history])) if self.state_history else 0.0

        return {
            "current_emotion": self.current_state,
            "duration_in_state_seconds": round(state_duration_seconds, 1),
            "duration_in_state_formatted": self._format_duration(state_duration_seconds),
            "volatility_index": round(volatility_index, 3),
            "trajectory_trend": trend,
            "trend_description": trend_desc,
            "rolling_avg_valence": round(avg_valence, 3),
            "rolling_avg_arousal": round(avg_arousal, 3),
            "rolling_avg_physical_distress": round(avg_physical_distress, 1),
            "samples_in_window": len(self.state_history)
        }
        
    def _format_duration(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
