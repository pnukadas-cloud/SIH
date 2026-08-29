"""
Security_API Domain Package
Houses authentication, authorization, role-based access control, and security hardening.
"""

from Security_API.authorization.roles import UserRole, has_permission
from Security_API.authentication.auth_manager import AuthManager
from Security_API.rbac.guard import get_current_user, get_current_user_optional, require_role
from Security_API.security_middleware.shield import SecurityShieldMiddleware
