# Backend_DB Domain

## Overview
This domain manages the core application server, REST APIs, database queries, and reporting exports.

## Folder Structure
- **`api/`**: Main master router aggregating all modular endpoints.
- **`routes/`**:
  - `auth_routes.py`: Session management, credentials verification, demo role switcher (`/api/auth/*`).
  - `admin_routes.py`: Flight Surgeon triage console, crew overview, system diagnostics (`/api/admin/*`, guarded by `require_role(UserRole.ADMIN)`).
  - `telemetry_routes.py`: Live frame processing, simulation scenario presets, and history records.
  - `chat_routes.py`: Spoken / typed interactions with MAITRI AI companion.
  - `export_routes.py`: High-fidelity multi-format download endpoints (`/api/export/*`).
- **`services/`**:
  - `pipeline_service.py`: Central coordinator linking AIML perception modules to SQLite persistence and S-Band alerting.
  - `export_service.py`: Vector PDF generator (via `reportlab`), structured JSON generator, and visual JPEG health passport generator (via `matplotlib`).
- **`database/`**:
  - `connection.py`: Thread-safe SQLite database manager (`telemetry_logs`, `dialogue_logs`, `ground_alerts`).

## Supported Export Formats
1. **JSON (`/api/export/json`)**: Serializes full biometric history, mission metadata, and ground alerts.
2. **PDF (`/api/export/pdf`)**: Formats an official ISRO / BAS Flight Surgeon Health Record with vector tables, telemetry metrics, and doctor sign-off blocks.
3. **JPG (`/api/export/jpg`)**: Generates a visual health passport card with crew metadata and vitals gauges.
