"""
Security_API — Security Middleware & Exception Shield
Attaches hardening headers and masks internal server errors.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

logger = logging.getLogger("MAITRI.Security")

class SecurityShieldMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response: Response = await call_next(request)
            # Attach Security Hardening Headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "camera=*, microphone=*"
            return response
        except Exception as exc:
            logger.error(f"[SECURITY SHIELD EXCEPTION] {request.url.path}: {exc}", exc_info=True)
            # Mask internal details from normal users
            return JSONResponse(
                status_code=500,
                content={
                    "status": "ERROR",
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "A secure mission server error occurred. Telemetry recorded to ground logs."
                }
            )
