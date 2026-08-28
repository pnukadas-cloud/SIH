"""
MAITRI — Run Script & Launcher
Multimodal AI Assistant for Psychological & Physical Well-Being of Astronauts
ISRO / Bhartiya Antariksh Station (BAS) · SIH 2025
"""

import sys
import os
import webbrowser
import time
import uvicorn
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from maitri.config import SERVER_HOST, SERVER_PORT, SYSTEM_NAME, SYSTEM_VERSION, SPACE_STATION

def print_banner():
    print("=" * 70)
    print(f"  🛰️  {SYSTEM_NAME} v{SYSTEM_VERSION}")
    print(f"  Multimodal AI Assistant for Astronaut Psychological & Physical Health")
    print(f"  Target Spacecraft: {SPACE_STATION}")
    print(f"  Smart India Hackathon (SIH 2025) | Problem ID: 25175 | ISRO")
    print("=" * 70)
    print(f"  [+] Local Standalone Edge URL: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  [+] Offline Multimodal Pipeline: ACTIVE")
    print(f"  [+] Press Ctrl+C to terminate orbital session.")
    print("=" * 70)

def main():
    print_banner()
    
    # Auto-open browser after 1.5 seconds
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://{SERVER_HOST}:{SERVER_PORT}")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI Uvicorn Server
    uvicorn.run(
        "server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()
