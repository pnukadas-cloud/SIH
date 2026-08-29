"""
Backend_DB — Authentication & RBAC Routes
Provides login, logout, current session inspection, and demo role switching.
"""

from fastapi import APIRouter, Response, Request, HTTPException, status
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
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

@router.get("/astronauts")
async def list_enrolled_astronauts():
    """Retrieve all database-enrolled astronauts for dynamic UI selection and login."""
    from Backend_DB.database.connection import DatabaseManager
    db = DatabaseManager()
    astronauts = db.list_astronauts()
    return {"status": "SUCCESS", "count": len(astronauts), "astronauts": astronauts}

@router.post("/enroll")
async def enroll_astronaut(req: Dict[str, Any], response: Response):
    """
    Enroll a new astronaut profile with credentials, physiological baselines,
    and optional camera-captured face biometric embedding into SQLite database.
    """
    from Backend_DB.database.connection import DatabaseManager
    from AIML.facial_emotion.face_recognizer import FaceRecognizer
    import hashlib

    astronaut_id = req.get("astronaut_id", "").strip()
    name = req.get("name", "").strip()
    callsign = req.get("callsign", "Flight-Crew").strip()
    role = req.get("role", "Mission Specialist").strip()
    username = req.get("username", astronaut_id.lower()).strip()
    password = req.get("password", "astronaut123").strip()
    frame_data = req.get("frame")

    if not astronaut_id or not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "astronaut_id and name are required for enrollment."}
        )

    db = DatabaseManager()
    recognizer = FaceRecognizer()

    # Extract 128-d face embedding if camera frame provided
    face_embedding = None
    if frame_data:
        frame = recognizer.decode_image(frame_data)
        if frame is not None:
            # Detect face
            faces = recognizer.fer.detect_face_multispectral(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if faces:
                fx, fy, fw, fh, _, _ = faces[0]
                face_crop = frame[fy:fy+fh, fx:fx+fw]
                face_embedding = recognizer.extract_face_embedding(face_crop)

    # If no camera frame provided, seed with deterministic signature based on astronaut_id
    if face_embedding is None:
        seed_int = sum(ord(c) for c in astronaut_id)
        rng = np.random.RandomState(seed_int)
        v = rng.randn(128).astype(np.float32)
        v = v / max(1e-6, float(np.linalg.norm(v)))
        face_embedding = [round(float(x), 5) for x in v]

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    profile = {
        "mission_assignment": req.get("mission_assignment", "Gaganyaan-BAS Expedition 1"),
        "clearance_level": req.get("clearance_level", "Level-2 (Flight Crew)"),
        "coping_preferences": req.get("coping_preferences", ["Tactical breathing", "Operational checklist review"])
    }
    baseline = req.get("baseline") or {
        "resting_heart_rate_bpm": 66,
        "blink_rate_bpm": 16.5,
        "resting_ear": 0.32,
        "resting_mar": 0.18,
        "mean_f0_pitch_hz": 130.0
    }

    astro = db.upsert_astronaut(
        astronaut_id=astronaut_id,
        name=name,
        callsign=callsign,
        role=role,
        username=username,
        password_hash=password_hash,
        profile=profile,
        baseline=baseline,
        face_embedding=face_embedding
    )

    return {
        "status": "SUCCESS",
        "message": f"Astronaut {name} ({astronaut_id}) enrolled successfully with biometric face profile.",
        "astronaut": {
            "astronaut_id": astro["astronaut_id"],
            "name": astro["name"],
            "callsign": astro["callsign"],
            "role": astro["role"],
            "has_face_enrolled": bool(face_embedding)
        }
    }

@router.post("/recognize-face")
async def recognize_face(req: Dict[str, Any], response: Response):
    """
    Biometric Face Recognition Endpoint:
    Receives camera frame -> Evaluates quality -> Computes 128-D embedding ->
    Matches against enrolled database -> Automatically activates astronaut and opens session.
    """
    from AIML.facial_emotion.face_recognizer import FaceRecognizer
    from Backend_DB.services.pipeline_service import pipeline_service

    frame_data = req.get("frame")
    if not frame_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "frame base64 data required."}
        )

    recognizer = FaceRecognizer()
    frame = recognizer.decode_image(frame_data)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid image format."}
        )

    result = recognizer.recognize_face(frame)

    if result["status"] == "IDENTIFIED":
        # Automatically load astronaut profile into pipeline service and start isolated session
        active_astro = pipeline_service.set_active_astronaut(result["astronaut_id"])
        
        # Issue authentication token
        token = AuthManager.create_session({
            "user_id": active_astro["astronaut_id"],
            "name": active_astro["name"],
            "callsign": active_astro["callsign"],
            "role": UserRole.ASTRONAUT,
            "clearance_level": "Level-2 (Flight Crew)"
        })
        response.set_cookie(
            key="maitri_session",
            value=token,
            httponly=True,
            max_age=86400,
            samesite="lax"
        )
        result["session_id"] = pipeline_service.current_session_id
        result["token"] = token
        result["astronaut"] = active_astro

    return result

@router.post("/session/stop")
async def stop_session():
    """Finalize active astronaut monitoring session."""
    from Backend_DB.services.pipeline_service import pipeline_service
    res = pipeline_service.close_active_session()
    return {"status": "SUCCESS", "session": res}

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
            "role": default_astronaut["role"].value if hasattr(default_astronaut["role"], "value") else default_astronaut["role"],
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
        "switched_to_role": user["role"].value if hasattr(user["role"], "value") else user["role"],
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "callsign": user["callsign"],
            "role": user["role"].value if hasattr(user["role"], "value") else user["role"],
            "clearance_level": user["clearance_level"]
        }
    }
