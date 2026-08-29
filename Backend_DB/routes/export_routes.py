"""
Backend_DB — Export & Download Endpoints
Provides real file downloads in JSON, vector PDF (via ReportLab), and visual JPG formats.
"""

from fastapi import APIRouter, Response, Query, Request, HTTPException
from typing import Optional
import time
from Backend_DB.services.export_service import ExportService
from Backend_DB.database.connection import DatabaseManager
from Security_API.authentication.auth_manager import USERS_DB

router = APIRouter(prefix="/api/export", tags=["Exports"])
db = DatabaseManager()

def _get_astronaut_profile(astronaut_id: str):
    return USERS_DB.get(astronaut_id, {
        "astronaut_id": astronaut_id,
        "name": "Captain Vikram Rathore",
        "callsign": "SURYA-1",
        "role": "Mission Commander"
    })

@router.get("/json")
async def export_json(astronaut_id: str = Query("CREW-BAS-01")):
    """Download comprehensive raw mission telemetry in JSON."""
    astronaut = _get_astronaut_profile(astronaut_id)
    telemetry = db.get_recent_telemetry(astronaut_id=astronaut_id, limit=100)
    alerts = db.get_alerts(limit=50)
    
    json_bytes = ExportService.generate_json_export(astronaut, telemetry, alerts)
    filename = f"MAITRI_Telemetry_{astronaut_id}_{int(time.time())}.json"
    
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@router.get("/pdf")
async def export_pdf(astronaut_id: str = Query("CREW-BAS-01")):
    """Download official Flight Surgeon medical evaluation report in vector PDF."""
    astronaut = _get_astronaut_profile(astronaut_id)
    telemetry = db.get_recent_telemetry(astronaut_id=astronaut_id, limit=60)
    alerts = db.get_alerts(limit=20)
    
    pdf_bytes = ExportService.generate_pdf_report(astronaut, telemetry, alerts)
    filename = f"MAITRI_Medical_Report_{astronaut_id}_{int(time.time())}.pdf"
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@router.get("/jpg")
async def export_jpg(astronaut_id: str = Query("CREW-BAS-01")):
    """Download visual health passport card in JPEG format."""
    astronaut = _get_astronaut_profile(astronaut_id)
    telemetry = db.get_recent_telemetry(astronaut_id=astronaut_id, limit=1)
    latest = telemetry[-1] if telemetry else {"risk_score": 14.2, "dominant_emotion": "Neutral", "valence": 0.15}
    
    jpg_bytes = ExportService.generate_jpg_passport(astronaut, latest)
    filename = f"MAITRI_Health_Passport_{astronaut_id}_{int(time.time())}.jpg"
    
    return Response(
        content=jpg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
