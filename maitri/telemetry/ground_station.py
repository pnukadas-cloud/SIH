"""
MAITRI — ISRO Ground Station Telemetry & Emergency Alert Dispatcher
Formats structured telemetry packets and queues orbital alerts for
transmission to Mission Control (ISTRAC / IDRSS) during communication windows.
"""

import time
import uuid
from typing import Dict, Any, List
from maitri.config import SPACE_STATION, AGENCY

class GroundStationDispatcher:
    def __init__(self):
        self.alert_queue = []
        self.dispatched_history = []
        self.station_id = SPACE_STATION
        self.agency = AGENCY
        
    def create_alert_packet(
        self,
        astronaut_profile: Dict[str, Any],
        fused_emotion: Dict[str, Any],
        physical_distress: Dict[str, Any],
        risk_data: Dict[str, Any],
        state_data: Dict[str, Any],
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construct official ISRO-standard ground control alert packet.
        """
        now = time.time()
        date_str = time.strftime("%Y-%m-%d", time.gmtime(now))
        time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        
        alert_id = f"MAITRI-ALERT-{date_str}-{uuid.uuid4().hex[:6].upper()}"
        astronaut_id = astronaut_profile.get("astronaut_id", "CREW-BAS-01")
        
        packet = {
            "alert_id": alert_id,
            "timestamp": time_iso,
            "spacecraft": self.station_id,
            "agency": self.agency,
            "astronaut_id": astronaut_id,
            "callsign": astronaut_profile.get("callsign", "Vyom-Leader"),
            "risk_level": risk_data.get("risk_level", 0),
            "risk_score": risk_data.get("risk_score", 0.0),
            "tier_name": risk_data.get("tier_name"),
            "emotional_state": {
                "primary": fused_emotion.get("dominant_emotion", "neutral"),
                "confidence": fused_emotion.get("confidence", 0.0),
                "valence": fused_emotion.get("valence", 0.0),
                "arousal": fused_emotion.get("arousal", 0.0),
                "duration_minutes": round(state_data.get("duration_in_state_seconds", 0.0) / 60.0, 1),
                "trend": state_data.get("trajectory_trend", "stable")
            },
            "physical_state": {
                "fatigue_level": physical_distress.get("fatigue_level", "Nominal"),
                "perclos_percentage": physical_distress.get("perclos_percentage", 0.0),
                "microsleep_risk": physical_distress.get("microsleep_risk_score", 0.0),
                "pain_grimace_score": physical_distress.get("pain_grimace_score", 0.0),
                "vocal_fatigue_score": physical_distress.get("vocal_fatigue_score", 0.0),
                "physical_complaints": physical_distress.get("physical_complaints_reported", [])
            },
            "cross_modal_discordance": {
                "detected": fused_emotion.get("cross_modal_discordance", False),
                "reason": fused_emotion.get("discordance_reason")
            },
            "intervention_deployed": {
                "id": intervention.get("id") if intervention else None,
                "name": intervention.get("name") if intervention else "None",
                "category": intervention.get("category") if intervention else "Passive Monitoring"
            },
            "recommended_ground_action": risk_data.get("recommendation"),
            "status": "QUEUED_FOR_ORBITAL_PASS"
        }
        
        self.alert_queue.append(packet)
        self.dispatched_history.insert(0, packet)
        if len(self.dispatched_history) > 50:
            self.dispatched_history.pop()
            
        return packet
        
    def get_pending_alerts(self) -> List[Dict[str, Any]]:
        return self.alert_queue
        
    def get_recent_alerts(self) -> List[Dict[str, Any]]:
        return self.dispatched_history
        
    def acknowledge_alert(self, alert_id: str):
        for alert in self.alert_queue:
            if alert["alert_id"] == alert_id:
                alert["status"] = "TRANSMITTED_TO_ISTRAC_GROUND"
        self.alert_queue = [a for a in self.alert_queue if a["alert_id"] != alert_id]
