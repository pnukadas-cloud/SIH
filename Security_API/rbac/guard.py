"""
Security_API — RBAC Route Guards & Authentication Dependencies
Enforces role-based access control across all API routes.
"""

from fastapi import Request, HTTPException, status, Depends
from typing import Optional, Dict, Any, Callable
from Security_API.authorization.roles import UserRole
from Security_API.authentication.auth_manager import AuthManager

async def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Extract authenticated user session from header, cookie, or query param."""
    token = None
    
    # 1. Custom Header
    auth_header = request.headers.get("X-MAITRI-Auth")
    if auth_header:
        token = auth_header.strip()
        
    # 2. Bearer Authorization Header
    if not token:
        bearer = request.headers.get("Authorization")
        if bearer and bearer.startswith("Bearer "):
            token = bearer.split("Bearer ")[1].strip()
            
    # 3. Cookie
    if not token:
        token = request.cookies.get("maitri_session")
        
    if not token:
        return None
        
    return AuthManager.get_session(token)

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Strict dependency requiring authenticated user; raises 401 if missing."""
    user = await get_current_user_optional(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Authentication required", "code": "AUTH_REQUIRED"}
        )
    return user

def require_role(required_role: UserRole) -> Callable:
    """Dependency factory checking that current user has required role."""
    async def role_checker(request: Request) -> Dict[str, Any]:
        user = await get_current_user_optional(request)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "Authentication required", "code": "AUTH_REQUIRED"}
            )
        user_role = user.get("role")
        # Admin can access everything
        if user_role == UserRole.ADMIN:
            return user
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": f"Access forbidden. Requires '{required_role.value}' role.",
                    "code": "FORBIDDEN_ROLE",
                    "current_role": user_role
                }
            )
        return user
    return role_checker
