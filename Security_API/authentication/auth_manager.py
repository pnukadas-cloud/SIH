"""
Security_API — Authentication & Session Management
Handles credentials verification, token generation, and session lifecycle.
"""

import os
import time
import secrets
import hashlib
from typing import Dict, Any, Optional
from Security_API.authorization.roles import UserRole

# In-Memory & Database-backed User Directory
USERS_DB: Dict[str, Dict[str, Any]] = {
    # Astronaut 1: Wing Cmdr. Prashanth Nair (Mission Commander)
    "CREW-BAS-01": {
        "user_id": "CREW-BAS-01",
        "username": "prashanth",
        "name": "Wing Cmdr. Prashanth Nair",
        "callsign": "Vyom-Leader",
        "role": UserRole.ASTRONAUT,
        "password_hash": hashlib.sha256("astronaut123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan-BAS Expedition 1",
        "clearance_level": "Level-2 (Mission Commander)"
    },
    # Astronaut 2: Dr. Sunidhi Sharma (Flight Surgeon / Payload Specialist)
    "CREW-BAS-02": {
        "user_id": "CREW-BAS-02",
        "username": "sunidhi",
        "name": "Dr. Sunidhi Sharma",
        "callsign": "Gagan-Doc",
        "role": UserRole.ASTRONAUT,
        "password_hash": hashlib.sha256("astronaut123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan-BAS Expedition 1",
        "clearance_level": "Level-2 (Payload Specialist)"
    },
    # Astronaut 3: Group Capt. Ajit Krishnan (Flight Engineer)
    "CREW-BAS-03": {
        "user_id": "CREW-BAS-03",
        "username": "ajit",
        "name": "Group Capt. Ajit Krishnan",
        "callsign": "Shakti-Pilot",
        "role": UserRole.ASTRONAUT,
        "password_hash": hashlib.sha256("astronaut123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan-BAS Expedition 1",
        "clearance_level": "Level-2 (Flight Engineer)"
    },
    # Demo Astronaut 1: Aryan (Commander)
    "AST-001": {
        "user_id": "AST-001",
        "username": "aryan",
        "name": "Aryan",
        "callsign": "GARUDA-1",
        "role": UserRole.ASTRONAUT,
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan-BAS Expedition 1",
        "clearance_level": "Level-2 (Mission Commander)"
    },
    # Demo Astronaut 2: Riya (Flight Engineer)
    "AST-002": {
        "user_id": "AST-002",
        "username": "riya",
        "name": "Riya",
        "callsign": "TEJAS-2",
        "role": UserRole.ASTRONAUT,
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan-BAS Expedition 1",
        "clearance_level": "Level-2 (Flight Engineer)"
    },
    # Demo Astronaut 3: Karan (Mission Specialist)
    "AST-003": {
        "user_id": "AST-003",
        "username": "karan",
        "name": "Karan",
        "callsign": "VIKRAM-3",
        "role": UserRole.ASTRONAUT,
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan-BAS Expedition 1",
        "clearance_level": "Level-2 (Mission Specialist)"
    },
    # Flight Surgeon / Admin: Dr. Sunita Sharma (Ground Station)
    "ADMIN-MED-01": {
        "user_id": "ADMIN-MED-01",
        "username": "surgeon_sharma",
        "name": "Dr. Sunita Sharma",
        "callsign": "GROUND-SURGEON",
        "role": UserRole.ADMIN,
        "password_hash": hashlib.sha256("isro_surgeon2025".encode()).hexdigest(),
        "mission_assignment": "ISRO Telemetry & Biomedical Mission Control",
        "clearance_level": "Level-4 (Chief Medical Officer)"
    }
}

# Active Session Cache: token -> user payload
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}

SESSION_EXPIRY_SECONDS = 86400  # 24 hours

class AuthManager:
    @staticmethod
    def verify_credentials(user_id_or_name: str, password_plain: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials against database first, with fallback to in-memory store."""
        input_hash = hashlib.sha256(password_plain.encode()).hexdigest()
        
        # 1. Database-backed verification (Primary)
        try:
            from Backend_DB.database.connection import DatabaseManager
            db = DatabaseManager()
            db_user = db.get_astronaut(user_id_or_name) or db.get_astronaut_by_username(user_id_or_name)
            if db_user:
                if db_user.get("password_hash") == input_hash:
                    role_enum = UserRole.ADMIN if ("admin" in (db_user.get("role") or "").lower() or "surgeon" in (db_user.get("role") or "").lower()) else UserRole.ASTRONAUT
                    return {
                        "user_id": db_user["astronaut_id"],
                        "username": db_user.get("username", ""),
                        "name": db_user["name"],
                        "callsign": db_user.get("callsign", ""),
                        "role": role_enum,
                        "mission_assignment": (db_user.get("profile") or {}).get("mission_assignment", "Gaganyaan-BAS Expedition 1"),
                        "clearance_level": (db_user.get("profile") or {}).get("clearance_level", "Level-2 (Flight Crew)")
                    }
        except Exception:
            pass

        # 2. In-memory fallback
        for user in USERS_DB.values():
            if (user["user_id"].lower() == user_id_or_name.lower() or 
                user["username"].lower() == user_id_or_name.lower()):
                if user["password_hash"] == input_hash:
                    return user
        return None

    @staticmethod
    def create_session(user: Dict[str, Any]) -> str:
        """Create a secure session token."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        ACTIVE_SESSIONS[token] = {
            "user_id": user["user_id"],
            "name": user["name"],
            "callsign": user["callsign"],
            "role": user["role"],
            "clearance_level": user["clearance_level"],
            "created_at": now,
            "expires_at": now + SESSION_EXPIRY_SECONDS
        }
        return token

    @staticmethod
    def get_session(token: str) -> Optional[Dict[str, Any]]:
        """Retrieve active session if not expired."""
        if not token or token not in ACTIVE_SESSIONS:
            return None
        session = ACTIVE_SESSIONS[token]
        if time.time() > session["expires_at"]:
            del ACTIVE_SESSIONS[token]
            return None
        return session

    @staticmethod
    def revoke_session(token: str) -> bool:
        """Revoke a session on logout."""
        if token in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[token]
            return True
        return False

    @staticmethod
    def get_default_astronaut() -> Dict[str, Any]:
        """Convenience accessor for default astronaut account."""
        return USERS_DB["CREW-BAS-01"]

    @staticmethod
    def get_default_admin() -> Dict[str, Any]:
        """Convenience accessor for default admin account."""
        return USERS_DB["ADMIN-MED-01"]
