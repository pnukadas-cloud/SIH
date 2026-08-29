# Frontend Domain

## Architecture & Layout Overview
The MAITRI user interface is designed as an aerospace mission-control product for ISRO / Bhartiya Antariksh Station (BAS).
It features:
- A responsive 60px Top Navigation Bar with live MET clock, active callsign, and 1-click RBAC role switcher (`Astronaut` <-> `Flight Surgeon`).
- Hardware Permission Lifecycle Banner with 4 explicit UI states: Ready to Request, Granted/Locked, Denied with step-by-step browser unblock instructions, and Device Unavailable.
- Dedicated Astronaut HUD with real-time FACS Action Units, EAR, MAR, PERCLOS, and vocal F0 pitch extraction.
- Dedicated Flight Surgeon / Admin Console with multi-crew monitoring, ground alert triage, and system diagnostics.
- Accurate Emotional Valence Timeline with real timestamps and colored bands for Positive (+0.2 to +1.0), Neutral (-0.2 to +0.2), and Negative (-1.0 to -0.2).
- Multi-format Export Center for Vector PDF (ReportLab), Structured JSON, and Visual Health Passports (JPG).

## Directory Structure
- **`components/`**: Modular interface elements (HUD reticle, audio waveform, breathing pacer, companion chat).
- **`dashboards/`**: Astronaut HUD portal and Flight Surgeon / Admin Console.
- **`charts/`**: Time-series SVG valence curves with colored affect zones.
- **`exports/`**: Export modal and asynchronous download handlers.
- **`services/`**: WebSocket telemetry stream and REST API client.
