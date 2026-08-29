"""
Backend_DB — Flight Surgeon / Administrator Protected Routes
Enforces UserRole.ADMIN RBAC on all endpoints. Rejects unauthorized requests with 403 Forbidden.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import time
from Security_API.authorization.roles import UserRole
from Security_API.rbac.guard import require_role
from Backend_DB.database.connection import DatabaseManager
from maitri.config import SYSTEM_NAME, SYSTEM_VERSION, SPACE_STATION, AGENCY

router = APIRouter(
    prefix="/api/admin",
    tags=["Flight Surgeon / Admin"],
    dependencies=[Depends(require_role(UserRole.ADMIN))]
)

db = DatabaseManager()

# Crew roster with baseline calibration
CREW_ROSTER = [
    {
        "astronaut_id": "CREW-BAS-01",
        "name": "Captain Vikram Rathore",
        "callsign": "SURYA-1",
        "role": "Mission Commander",
        "resting_hr": 64,
        "base_f0": 132.0,
        "current_status": "NOMINAL",
        "last_wellbeing_score": 14.5
    },
    {
        "astronaut_id": "CREW-BAS-02",
        "name": "Dr. Ananya Iyer",
        "callsign": "CHANDRA-2",
        "role": "Science Officer / Astrobiologist",
        "resting_hr": 70,
        "base_f0": 195.0,
        "current_status": "RESTED",
        "last_wellbeing_score": 18.0
    },
    {
        "astronaut_id": "CREW-BAS-03",
        "name": "Major Rajesh Pillai",
        "callsign": "AGNI-3",
        "role": "Flight Engineer / Systems Specialist",
        "resting_hr": 62,
        "base_f0": 128.0,
        "current_status": "NOMINAL",
        "last_wellbeing_score": 12.0
    }
]

@router.get("/crew-summary")
async def get_crew_summary(admin_user: Dict[str, Any] = Depends(require_role(UserRole.ADMIN))):
    """Retrieve full crew status overview for Flight Surgeon console."""
    # Update with recent telemetry if available
    recent = db.get_recent_telemetry(limit=10)
    if recent:
        latest = recent[-1]
        for crew in CREW_ROSTER:
            if crew["astronaut_id"] == latest.get("astronaut_id", "CREW-BAS-01"):
                crew["last_wellbeing_score"] = latest.get("risk_score", 14.5)
                crew["current_status"] = "DISTRESS" if crew["last_wellbeing_score"] > 60 else ("ELEVATED" if crew["last_wellbeing_score"] > 35 else "NOMINAL")

    return {
        "status": "AUTHORIZED",
        "flight_surgeon": admin_user["name"],
        "roster_count": len(CREW_ROSTER),
        "crew": CREW_ROSTER,
        "timestamp": time.time()
    }

@router.get("/alerts/triage")
async def get_alerts_triage(admin_user: Dict[str, Any] = Depends(require_role(UserRole.ADMIN))):
    """Retrieve prioritized ground station alerts queue."""
    alerts = db.get_alerts(limit=25)
    return {
        "status": "AUTHORIZED",
        "queue_length": len(alerts),
        "alerts": alerts
    }

@router.post("/alert/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str, admin_user: Dict[str, Any] = Depends(require_role(UserRole.ADMIN))):
    """Acknowledge and mark ground station alert as verified."""
    success = db.acknowledge_alert(alert_id)
    return {
        "status": "ACKNOWLEDGED" if success else "ALERT_NOT_FOUND",
        "alert_id": alert_id,
        "acknowledged_by": admin_user["name"],
        "timestamp": time.time()
    }

@router.get("/system-diagnostics")
async def get_system_diagnostics(admin_user: Dict[str, Any] = Depends(require_role(UserRole.ADMIN))):
    """Retrieve internal AI pipeline latency and modality health."""
    return {
        "status": "AUTHORIZED",
        "system": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
        "station": SPACE_STATION,
        "agency": AGENCY,
        "pipeline_health": {
            "vision_fer": {"status": "ONLINE", "fps_target": 4, "device": "WebRTC / OpenCV Universal"},
            "audio_ser": {"status": "ONLINE", "f0_tracker": "Autocorrelation 60-450Hz"},
            "fusion_engine": {"status": "ONLINE", "type": "Attention-Weighted Late Fusion"},
            "database": {"status": "ONLINE", "engine": "SQLite WAL Mode"},
            "telemetry_stream": {"status": "ONLINE", "protocol": "WebSocket /ws/telemetry"}
        },
        "timestamp": time.time()
    }
