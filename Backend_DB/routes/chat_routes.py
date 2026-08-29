"""
Backend_DB — Conversational Companion Chat Routes
Handles astronaut natural language interactions with MAITRI AI.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from Backend_DB.routes.telemetry_routes import pipeline_service

router = APIRouter(prefix="/api", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    astronaut_id: Optional[str] = None

@router.post("/interact")
async def chat_interact(req: ChatRequest):
    """Communicate with MAITRI Conversational AI companion."""
    if req.astronaut_id:
        pipeline_service.set_active_astronaut(req.astronaut_id)
    response = pipeline_service.interact(req.message)
    return response
