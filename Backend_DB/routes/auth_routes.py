"""
Backend_DB — Authentication & RBAC Routes
Provides login, logout, current session inspection, and demo role switching.
"""

from fastapi import APIRouter, Response, Request, HTTPException, status
from typing import Dict, Any
from Security_API.authentication.auth_manager import AuthManager, USERS_DB
from Security_API.authorization.roles import UserRole
from Security_API.rbac.guard import get_current_user_optional
from Security_API.validation.schemas import LoginRequest, SwitchRoleRequest

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/login")
async def login(req: LoginRequest, response: Response):
    user = AuthManager.verify_credentials(req.username_or_id, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid crew ID or passcode", "code": "INVALID_CREDENTIALS"}
        )
    token = AuthManager.create_session(user)
    response.set_cookie(
        key="maitri_session",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax"
    )
    return {
        "status": "SUCCESS",
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "callsign": user["callsign"],
            "role": user["role"].value,
            "clearance_level": user["clearance_level"]
        }
    }

@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.headers.get("X-MAITRI-Auth") or request.cookies.get("maitri_session")
    if token:
        AuthManager.revoke_session(token)
    response.delete_cookie("maitri_session")
    return {"status": "SUCCESS", "message": "Logged out successfully"}

@router.get("/me")
async def get_me(request: Request):
    user = await get_current_user_optional(request)
    if user:
        return {"authenticated": True, "user": user}
    # Default demo astronaut session if not logged in
    default_astronaut = AuthManager.get_default_astronaut()
    return {
        "authenticated": False,
        "default_mode": "DEMO_ASTRONAUT",
        "user": {
            "user_id": default_astronaut["user_id"],
            "name": default_astronaut["name"],
            "callsign": default_astronaut["callsign"],
            "role": default_astronaut["role"].value,
            "clearance_level": default_astronaut["clearance_level"]
        }
    }

@router.post("/switch-role")
async def switch_demo_role(req: SwitchRoleRequest, response: Response):
    """Convenience endpoint for hackathon judges to toggle between Astronaut and Admin views."""
    if req.role == UserRole.ADMIN:
        user = AuthManager.get_default_admin()
    else:
        user = AuthManager.get_default_astronaut()
        
    token = AuthManager.create_session(user)
    response.set_cookie(
        key="maitri_session",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax"
    )
    return {
        "status": "SUCCESS",
        "switched_to_role": user["role"].value,
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "callsign": user["callsign"],
            "role": user["role"].value,
            "clearance_level": user["clearance_level"]
        }
    }
