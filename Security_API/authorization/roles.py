"""
Security_API — Role Definitions & Permissions
Defines access levels for Astronauts and Flight Surgeons (Admins).
"""

from enum import Enum
from typing import List, Dict, Set

class UserRole(str, Enum):
    ASTRONAUT = "astronaut"
    ADMIN = "admin"  # Flight Surgeon / Mission Director

# Role Permission Mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.ASTRONAUT: {
        "telemetry:read_self",
        "telemetry:write_self",
        "chat:interact",
        "interventions:execute",
        "export:self_data",
        "camera:stream",
        "microphone:stream"
    },
    UserRole.ADMIN: {
        "telemetry:read_self",
        "telemetry:write_self",
        "telemetry:read_all",
        "chat:interact",
        "interventions:execute",
        "interventions:manage",
        "alerts:read",
        "alerts:acknowledge",
        "alerts:escalate",
        "diagnostics:read",
        "crew:manage",
        "export:all_data",
        "export:self_data",
        "camera:stream",
        "microphone:stream"
    }
}

def has_permission(role: UserRole, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())
