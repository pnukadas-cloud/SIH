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
    # Default Astronaut: Captain Vikram Rathore
    "CREW-BAS-01": {
        "user_id": "CREW-BAS-01",
        "username": "vikram",
        "name": "Captain Vikram Rathore",
        "callsign": "SURYA-1",
        "role": UserRole.ASTRONAUT,
        # SHA-256 for demo password 'astronaut123'
        "password_hash": hashlib.sha256("astronaut123".encode()).hexdigest(),
        "mission_assignment": "Gaganyaan Orbital Expedition",
        "clearance_level": "Level-2 (Orbital Crew)"
    },
    # Default Flight Surgeon / Admin: Dr. Sunita Sharma
    "ADMIN-MED-01": {
        "user_id": "ADMIN-MED-01",
        "username": "surgeon_sharma",
        "name": "Dr. Sunita Sharma",
        "callsign": "GROUND-SURGEON",
        "role": UserRole.ADMIN,
        # SHA-256 for demo password 'isro_surgeon2025'
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
        """Verify user credentials and return user record if valid."""
        input_hash = hashlib.sha256(password_plain.encode()).hexdigest()
        
        # Search by user_id or username
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
