"""
MAITRI — FastAPI Application & WebSocket Server
Multimodal AI Assistant for Psychological & Physical Well-Being of Astronauts
SIH 2025 · Problem ID 25175 · ISRO / Department of Space
"""

import os
import json
import base64
import time
import asyncio
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from maitri.config import (
    SYSTEM_NAME, SYSTEM_VERSION, SPACE_STATION, AGENCY,
    SERVER_HOST, SERVER_PORT, BASE_DIR
)
from maitri.pipeline import MaitriPipeline

app = FastAPI(
    title=f"{SYSTEM_NAME} — Orbital Well-Being Assistant",
    description="Multimodal AI for Astronaut Psychological & Physical Health",
    version=SYSTEM_VERSION
)

# Setup Templates and Static directories
WEB_DIR = BASE_DIR / "maitri" / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize Singleton Pipeline
pipeline = MaitriPipeline()

# Active WebSocket connections
connected_websockets = set()

# Pydantic Schemas
class MessageRequest(BaseModel):
    message: str
    astronaut_id: Optional[str] = None

class FrameDataRequest(BaseModel):
    image_base64: Optional[str] = None
    audio_energy: Optional[float] = None
    transcript: Optional[str] = ""
    astronaut_id: Optional[str] = None

class AstronautSelectRequest(BaseModel):
    astronaut_id: str

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/analysis", response_class=HTMLResponse)
@app.get("/interventions", response_class=HTMLResponse)
@app.get("/alerts", response_class=HTMLResponse)
@app.get("/sessions", response_class=HTMLResponse)
@app.get("/profile", response_class=HTMLResponse)
@app.get("/architecture", response_class=HTMLResponse)
async def index_page(request: Request):
    """Render main Spacecraft HUD & Mission Control Dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "system_name": SYSTEM_NAME,
            "system_version": SYSTEM_VERSION,
            "space_station": SPACE_STATION,
            "agency": AGENCY,
            "crew_profiles": pipeline.crew_profiles,
            "active_astronaut": pipeline.active_astronaut,
            "interventions": pipeline.interventions.get_all_interventions()
        }
    )

@app.get("/api/status")
async def get_status():
    """Return system health and orbital metadata."""
    return {
        "status": "ONLINE",
        "system": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
        "station": SPACE_STATION,
        "agency": AGENCY,
        "active_crew": pipeline.active_astronaut,
        "timestamp": time.time()
    }

@app.get("/api/crew")
async def get_crew():
    """Return all registered crew profiles."""
    return {"crew": pipeline.crew_profiles, "active": pipeline.active_astronaut}

@app.post("/api/crew/select")
async def select_crew(req: AstronautSelectRequest):
    """Switch active crew profile for personalized baseline."""
    selected = pipeline.set_active_astronaut(req.astronaut_id)
    return {"status": "SUCCESS", "active_crew": selected}

@app.post("/api/process_frame")
async def process_frame(req: FrameDataRequest):
    """Process a webcam frame (Base64 JPEG) and optional audio data."""
    frame = None
    if req.image_base64:
        try:
            # Strip data URI header if present
            header_prefix = "base64,"
            b64_data = req.image_base64
            if header_prefix in b64_data:
                b64_data = b64_data.split(header_prefix)[1]
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[MAITRI] Frame decode error: {e}")

    # Process in pipeline
    telemetry = pipeline.process_frame_and_audio(
        frame=frame,
        audio_chunk=None,
        text_transcript=req.transcript or ""
    )
    return telemetry

@app.post("/api/interact")
async def chat_interact(req: MessageRequest):
    """Astronaut conversational chat endpoint with speech response."""
    if req.astronaut_id:
        pipeline.set_active_astronaut(req.astronaut_id)
    response = pipeline.interact(req.message)
    return response

@app.post("/api/simulate/{scenario_name}")
async def simulate_scenario(scenario_name: str):
    """Trigger high-fidelity flight simulation scenario for demonstration."""
    result = pipeline.simulate_scenario(scenario_name)
    return result

@app.get("/api/interventions")
async def list_interventions():
    """Retrieve all evidence-based psychological and physical protocols."""
    return {"interventions": pipeline.interventions.get_all_interventions()}

@app.get("/api/history/telemetry")
async def get_telemetry_history():
    """Retrieve historical telemetry for charting."""
    data = pipeline.memory.get_telemetry_history(limit=60)
    return {"history": data}

@app.get("/api/history/alerts")
async def get_alerts_history():
    """Retrieve historical ground station alerts."""
    data = pipeline.ground.get_recent_alerts()
    return {"alerts": data}

@app.post("/api/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str):
    """Acknowledge ground station alert."""
    pipeline.ground.acknowledge_alert(alert_id)
    return {"status": "ACKNOWLEDGED", "alert_id": alert_id}

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Real-time WebSocket stream for telemetry HUD updates."""
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            # Receive client ping or frame
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                action = parsed.get("action")
                
                if action == "frame":
                    b64_img = parsed.get("image_base64")
                    frame = None
                    if b64_img:
                        header_prefix = "base64,"
                        if header_prefix in b64_img:
                            b64_img = b64_img.split(header_prefix)[1]
                        img_bytes = base64.b64decode(b64_img)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                    telemetry = pipeline.process_frame_and_audio(
                        frame=frame,
                        text_transcript=parsed.get("transcript", "")
                    )
                    await websocket.send_json({"type": "telemetry", "payload": telemetry})
                    
                elif action == "chat":
                    msg = parsed.get("message", "")
                    chat_res = pipeline.interact(msg)
                    await websocket.send_json({"type": "chat_response", "payload": chat_res})
                    
                elif action == "simulate":
                    scenario = parsed.get("scenario", "nominal")
                    sim_res = pipeline.simulate_scenario(scenario)
                    await websocket.send_json({"type": "telemetry", "payload": sim_res})
            except Exception as inner_e:
                print(f"[WS Error]: {inner_e}")
                
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
    except Exception as e:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
