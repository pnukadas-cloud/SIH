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

class AIKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=5, max_length=200)
    provider: Optional[str] = "gemini"

@router.post("/settings/ai-key")
async def set_ai_key(req: AIKeyRequest):
    """Dynamically set and persist Gemini / LLM API key."""
    pipeline_service.companion.set_api_key(req.api_key, req.provider or "gemini")
    return {
        "status": "SUCCESS",
        "message": f"Successfully activated {req.provider or 'gemini'} generative brain!",
        "details": pipeline_service.companion.get_status()
    }

@router.get("/settings/ai-status")
async def get_ai_status():
    """Check whether online Gemini LLM is active or running on offline cognitive engine."""
    return pipeline_service.companion.get_status()
