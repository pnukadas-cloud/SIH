"""
Security_API — Input Validation Schemas
Validates authentication and user management payloads.
"""

from pydantic import BaseModel, Field
from typing import Optional
from Security_API.authorization.roles import UserRole

class LoginRequest(BaseModel):
    username_or_id: str = Field(..., min_length=2, max_length=64, description="User ID or callsign")
    password: str = Field(..., min_length=4, max_length=128, description="Passcode")

class LoginResponse(BaseModel):
    status: str
    token: str
    user_id: str
    name: str
    callsign: str
    role: UserRole
    clearance_level: str

class SwitchRoleRequest(BaseModel):
    role: UserRole
