# Security_API Domain

## Overview
This domain is responsible for all identity, credential, authorization, and endpoint protection mechanisms within MAITRI.

## Architecture
- **`authentication/`**: Manages user directory, cryptographic password matching (SHA-256), and session token lifecycle.
- **`authorization/`**: Defines system roles (`UserRole.ASTRONAUT`, `UserRole.ADMIN`) and granular permission sets.
- **`rbac/`**: FastAPI dependency injection guards (`require_role()`, `get_current_user()`). Blocks unauthorized requests directly at the routing layer with HTTP 401/403.
- **`validation/`**: Pydantic input sanitizers and schema constraints.
- **`security_middleware/`**: Shield middleware preventing stack trace leakage, configuring CORS, and adding HTTP security headers (`nosniff`, `SAMEORIGIN`).

## Default Demo Credentials
| Role | User ID | Username | Password | Callsign |
|---|---|---|---|---|
| **Astronaut** | `CREW-BAS-01` | `vikram` | `astronaut123` | SURYA-1 |
| **Flight Surgeon (Admin)** | `ADMIN-MED-01` | `surgeon_sharma` | `isro_surgeon2025` | GROUND-SURGEON |

## RBAC Enforcement Rules
- Any endpoint under `/api/admin/*` is protected by `require_role(UserRole.ADMIN)`.
- Astronaut accounts attempting to hit `/api/admin/*` receive `403 Forbidden` with a standardized security response payload.
- Telemetry writing accepts self-writes for current crew or administrative overrides.
