"""
MAITRI — Evidence-Based Intervention Selector & Protocol Manager
Selects and guides astronauts through tactical breathing, sensory grounding,
CBT cognitive reframing, and circadian micro-rest protocols.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from maitri.config import DATA_DIR

class InterventionManager:
    def __init__(self, db_path: Path = DATA_DIR / "interventions_db.json"):
        self.db_path = db_path
        self.interventions = self._load_interventions()
        
    def _load_interventions(self) -> List[Dict[str, Any]]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("interventions", [])
        except Exception as e:
            print(f"[MAITRI] Warning: Error loading interventions DB: {e}")
            return []

    def select_intervention(self, dominant_emotion: str, risk_level: int, physical_distress: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Select the most clinically appropriate intervention based on multimodal findings.
        """
        # Prioritize physical fatigue / microsleep
        if physical_distress.get("fatigue_level", "").startswith("Severe") or dominant_emotion == "fatigued":
            for item in self.interventions:
                if item["id"] == "INT-FATIGUE-04":
                    return item
                    
        # Critical panic / anxiety -> 5-4-3-2-1 Sensory Grounding
        if risk_level >= 3 and dominant_emotion in ["anxious", "stressed"]:
            for item in self.interventions:
                if item["id"] == "INT-GROUND-02":
                    return item
                    
        # Acute autonomic arousal / frustration / moderate stress -> Box Breathing
        if dominant_emotion in ["stressed", "anxious", "frustrated"]:
            for item in self.interventions:
                if item["id"] == "INT-BREATHE-01":
                    return item
                    
        # Sadness / Isolation -> Companionship
        if dominant_emotion in ["sad", "neutral"]:
            for item in self.interventions:
                if item["id"] == "INT-COMPANION-05":
                    return item
                    
        # Default fallback to first matching intervention
        for item in self.interventions:
            if dominant_emotion in item.get("target_emotions", []):
                return item
                
        return self.interventions[0] if self.interventions else None

    def get_all_interventions(self) -> List[Dict[str, Any]]:
        return self.interventions
