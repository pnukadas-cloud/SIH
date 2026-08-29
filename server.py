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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from maitri.config import (
    SYSTEM_NAME, SYSTEM_VERSION, SPACE_STATION, AGENCY,
    BASE_DIR
)
from Security_API.security_middleware.shield import SecurityShieldMiddleware
from Security_API.rbac.guard import get_current_user_optional
from Security_API.authentication.auth_manager import AuthManager
from Backend_DB.api.router import master_api_router
from Backend_DB.routes.telemetry_routes import pipeline_service

app = FastAPI(
    title=f"{SYSTEM_NAME} — Orbital Well-Being Assistant",
    description="Multimodal AI for Astronaut Psychological & Physical Health",
    version=SYSTEM_VERSION
)

# Attach Security Shield Middleware
app.add_middleware(SecurityShieldMiddleware)

# Mount Domain Master API Router
app.include_router(master_api_router)

# Setup Templates and Static directories
WEB_DIR = BASE_DIR / "maitri" / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Active WebSocket connections
connected_websockets = set()

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
@app.get("/analysis", response_class=HTMLResponse)
@app.get("/interventions", response_class=HTMLResponse)
@app.get("/alerts", response_class=HTMLResponse)
@app.get("/sessions", response_class=HTMLResponse)
@app.get("/profile", response_class=HTMLResponse)
@app.get("/architecture", response_class=HTMLResponse)
async def index_page(request: Request):
    """Render main Spacecraft HUD, Astronaut Portal & Flight Surgeon Console."""
    user = await get_current_user_optional(request)
    if not user:
        user = AuthManager.get_default_astronaut()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "system_name": SYSTEM_NAME,
            "system_version": SYSTEM_VERSION,
            "space_station": SPACE_STATION,
            "agency": AGENCY,
            "crew_profiles": pipeline_service.crew_profiles,
            "active_astronaut": pipeline_service.active_astronaut,
            "interventions": pipeline_service.interventions.get_all_interventions(),
            "current_user": user
        }
    )

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Real-time WebSocket stream for telemetry HUD updates and sub-50ms latency."""
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
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
                        
                    telemetry = pipeline_service.process_frame_and_audio(
                        frame=frame,
                        text_transcript=parsed.get("transcript", "")
                    )
                    await websocket.send_json({"type": "telemetry", "payload": telemetry})
                    
                elif action == "chat":
                    msg = parsed.get("message", "")
                    chat_res = pipeline_service.interact(msg)
                    await websocket.send_json({"type": "chat_response", "payload": chat_res})
                    
                elif action == "simulate":
                    scenario = parsed.get("scenario", "nominal")
                    sim_res = pipeline_service.simulate_scenario(scenario)
                    await websocket.send_json({"type": "telemetry", "payload": sim_res})
            except Exception as inner_e:
                print(f"[MAITRI WS Error]: {inner_e}")
                
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
    except Exception as e:
        connected_websockets.discard(websocket)
