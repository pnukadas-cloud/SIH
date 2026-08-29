"""
Backend_DB — Main API Router Aggregator
Combines auth, admin, telemetry, chat, and export routers into a single master router.
"""

from fastapi import APIRouter
from Backend_DB.routes.auth_routes import router as auth_router
from Backend_DB.routes.admin_routes import router as admin_router
from Backend_DB.routes.telemetry_routes import router as telemetry_router
from Backend_DB.routes.chat_routes import router as chat_router
from Backend_DB.routes.export_routes import router as export_router

master_api_router = APIRouter()
master_api_router.include_router(auth_router)
master_api_router.include_router(admin_router)
master_api_router.include_router(telemetry_router)
master_api_router.include_router(chat_router)
master_api_router.include_router(export_router)
