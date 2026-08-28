"""
MAITRI — Configuration & Constants
Multimodal AI Assistant for Psychological & Physical Well-Being of Astronauts
Bhartiya Antariksh Station (BAS) / Gaganyaan Mission Ecosystem
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "maitri" / "data"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "maitri_session.db"

# Ensure runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# System Metadata
SYSTEM_NAME = "MAITRI"
SYSTEM_VERSION = "2.5.0-ORBITAL"
SPACE_STATION = "Bhartiya Antariksh Station (BAS)"
AGENCY = "ISRO / Department of Space"

# Emotion Classes (7 Standardized Dimensions)
EMOTION_CLASSES = [
    "Calm / Neutral",
    "Happy / Uplifted",
    "Stressed / Overloaded",
    "Fatigued / Drowsy",
    "Anxious / Tense",
    "Sad / Isolated",
    "Frustrated / Irritated"
]

# Primary Emotion Keys for Internal Computation
EMOTIONS = ["neutral", "happy", "stressed", "fatigued", "anxious", "sad", "frustrated"]

# Modality Weights for Attention-Weighted Late Fusion
DEFAULT_FUSION_WEIGHTS = {
    "facial": 0.40,
    "speech": 0.35,
    "linguistic": 0.25
}

# Physical Distress Thresholds
PERCLOS_FATIGUE_THRESHOLD = 0.25      # Eye closure > 25% over rolling window
YAWN_MAR_THRESHOLD = 0.55             # Mouth Aspect Ratio for yawn detection
BLINK_RATE_HIGH_THRESHOLD = 28.0      # Blinks/min (indicates acute cognitive stress)
BLINK_RATE_LOW_THRESHOLD = 8.0        # Blinks/min (indicates stare/extreme fatigue)
VOICE_TREMOR_THRESHOLD = 0.38         # Pitch instability index for panic/distress

# Risk Scoring Thresholds
RISK_LEVEL_THRESHOLDS = {
    "LEVEL_0_NOMINAL": 30.0,
    "LEVEL_1_MILD": 50.0,
    "LEVEL_2_MODERATE": 70.0,
    "LEVEL_3_CRITICAL": 100.0
}

# Web Server Config
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
WEBSOCKET_INTERVAL_MS = 250  # 4Hz telemetry broadcast
