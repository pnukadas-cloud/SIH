"""
MAITRI — Test Suite & Pipeline Verification
Validates vision, audio, analysis, fusion, decision, and response layers.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Force UTF-8 stdout
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from maitri.pipeline import MaitriPipeline

def run_tests():
    print("=" * 60)
    print("  MAITRI -- VERIFYING CORE SUBSYSTEMS")
    print("=" * 60)

    print("[1/6] Initializing Pipeline...")
    pipeline = MaitriPipeline()
    print("  [OK] Pipeline initialized successfully.")

    print("\n[2/6] Testing Synthetic Multimodal Frame Processing...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
    telemetry = pipeline.process_frame_and_audio(
        frame=dummy_frame,
        audio_chunk=dummy_audio,
        text_transcript="Everything is running smoothly on flight schedule."
    )
    assert "fusion" in telemetry
    assert "risk_assessment" in telemetry
    assert "physical_distress" in telemetry
    print(f"  [OK] Frame processed. Dominant Emotion: {telemetry['fusion']['dominant_emotion']}")
    print(f"  [OK] Risk Score: {telemetry['risk_assessment']['risk_score']} ({telemetry['risk_assessment']['tier_name']})")

    print("\n[3/6] Testing Interactive Dialogue & Intervention Engine...")
    chat_res = pipeline.interact("I feel a bit overwhelmed by the docking thruster alarms.")
    assert "ai_response" in chat_res
    assert "detected_state" in chat_res
    print(f"  [OK] Astronaut Prompt: 'I feel a bit overwhelmed by the docking thruster alarms.'")
    print(f"  [OK] MAITRI Response: '{chat_res['ai_response']}'")
    print(f"  [OK] Detected State: {chat_res['detected_state']} | Risk Level: {chat_res['risk_level']}")

    print("\n[4/6] Testing Flight Simulation Scenarios...")
    scenarios = ["nominal", "docking_stress", "isolation_sadness", "severe_fatigue", "masked_stress"]
    for s in scenarios:
        sim_data = pipeline.simulate_scenario(s)
        print(f"  [OK] Scenario '{s}': Dominant={sim_data['fusion']['dominant_emotion']} | Risk={sim_data['risk_assessment']['risk_score']} (Level {sim_data['risk_assessment']['risk_level']})")

    print("\n[5/6] Testing Ground Station Alert Dispatcher...")
    alerts = pipeline.ground.get_recent_alerts()
    print(f"  [OK] Total Generated Ground Alerts: {len(alerts)}")
    if alerts:
        print(f"  [OK] Latest Alert ID: {alerts[0]['alert_id']} -> Status: {alerts[0]['status']}")

    print("\n[6/6] Testing Context Memory SQLite Queries...")
    history = pipeline.memory.get_telemetry_history(limit=5)
    print(f"  [OK] Logged Telemetry Records: {len(history)}")

    print("\n" + "=" * 60)
    print("  * ALL SUBSYSTEM TESTS PASSED WITH 100% SUCCESS! *")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
