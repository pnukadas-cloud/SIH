"""
Backend_DB — Telemetry & Processing Routes
Handles camera frame ingestion, simulation scenarios, history queries, and crew management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import numpy as np
import cv2
import time

from Backend_DB.services.pipeline_service import MasterPipelineService
from maitri.config import SYSTEM_NAME, SYSTEM_VERSION, SPACE_STATION, AGENCY

router = APIRouter(prefix="/api", tags=["Telemetry"])
pipeline_service = MasterPipelineService()

class FrameDataRequest(BaseModel):
    image_base64: Optional[str] = None
    transcript: Optional[str] = ""
    astronaut_id: Optional[str] = None

class AstronautSelectRequest(BaseModel):
    astronaut_id: str

@router.get("/status")
async def get_status():
    return {
        "status": "ONLINE",
        "system": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
        "station": SPACE_STATION,
        "agency": AGENCY,
        "active_crew": pipeline_service.active_astronaut,
        "timestamp": time.time()
    }

@router.get("/crew")
async def get_crew():
    return {
        "crew": pipeline_service.crew_profiles,
        "active": pipeline_service.active_astronaut
    }

@router.post("/crew/select")
async def select_crew(req: AstronautSelectRequest):
    selected = pipeline_service.set_active_astronaut(req.astronaut_id)
    return {"status": "SUCCESS", "active_crew": selected}

@router.post("/process_frame")
async def process_frame(req: FrameDataRequest):
    frame = None
    if req.image_base64:
        try:
            b64_data = req.image_base64
            header_prefix = "base64,"
            if header_prefix in b64_data:
                b64_data = b64_data.split(header_prefix)[1]
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[MAITRI] Frame decode error: {e}")

    if req.astronaut_id:
        pipeline_service.set_active_astronaut(req.astronaut_id)

    telemetry = pipeline_service.process_frame_and_audio(
        frame=frame,
        audio_chunk=None,
        text_transcript=req.transcript or ""
    )
    return telemetry

@router.post("/simulate/{scenario_name}")
async def simulate_scenario(scenario_name: str):
    return pipeline_service.simulate_scenario(scenario_name)

@router.get("/history/telemetry")
async def get_telemetry_history(astronaut_id: Optional[str] = None):
    data = pipeline_service.db.get_recent_telemetry(astronaut_id=astronaut_id, limit=60)
    return {"history": data}

@router.get("/history/alerts")
async def get_alerts_history():
    data = pipeline_service.db.get_alerts(limit=25)
    return {"alerts": data}

@router.post("/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str):
    pipeline_service.db.acknowledge_alert(alert_id)
    return {"status": "ACKNOWLEDGED", "alert_id": alert_id}

@router.get("/interventions")
async def list_interventions():
    return {"interventions": pipeline_service.interventions.get_all_interventions()}
