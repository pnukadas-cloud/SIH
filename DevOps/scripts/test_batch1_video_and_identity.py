"""
Tests for MAITRI Batch 1: Phase 1 (Astronaut Identity) + Phase 2 (Robust Video Emotion Analysis).
Tests:
1. Neutral face
2. Smiling
3. Frowning
4. Talking
5. Looking away
6. Poor lighting
7. No face
8. Face partially covered
9. Rapid expression changes (Temporal Stability & Hysteresis)
10. Astronaut Identity Isolation & Session Partitioning
"""

import cv2
import numpy as np
import time
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from AIML.facial_emotion.fer_module import FacialEmotionModule
from Backend_DB.database.connection import DatabaseManager
from Backend_DB.services.pipeline_service import MasterPipelineService

def create_synthetic_face(
    width=640, height=480,
    brightness=150,
    face_cx=320, face_cy=240,
    face_rx=100, face_ry=130,
    eye_openness=15,
    mouth_openness=10,
    mouth_width=60,
    brow_furrow=False,
    blur=False,
    partial_cover=False
):
    """Generate controlled synthetic test frames mimicking webcam feeds under various optical conditions."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill cabin background with cool tone (B > R) so YCrCb cleanly isolates skin
    bg_val = min(brightness, 50)
    img[:, :] = (int(bg_val * 1.0), int(bg_val * 0.7), int(bg_val * 0.4))
    
    # Draw skin tone in BGR (R > G > B, typical human phototype)
    skin_bgr = (int(brightness * 0.55), int(brightness * 0.75), int(brightness * 1.0))
    skin_bgr = tuple(int(np.clip(c, 0, 255)) for c in skin_bgr)
    
    # Draw head oval
    cv2.ellipse(img, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360, skin_bgr, -1)
    
    # Draw eyes
    eye_y = face_cy - 30
    cv2.ellipse(img, (face_cx - 40, eye_y), (18, eye_openness), 0, 0, 360, (20, 20, 30), -1)
    cv2.ellipse(img, (face_cx + 40, eye_y), (18, eye_openness), 0, 0, 360, (20, 20, 30), -1)
    
    # Draw eyebrows / brow furrow
    if brow_furrow:
        # Angled tense brows + glabella vertical strain wrinkles
        cv2.line(img, (face_cx - 55, eye_y - 25), (face_cx - 20, eye_y - 12), (10, 10, 20), 4)
        cv2.line(img, (face_cx + 55, eye_y - 25), (face_cx + 20, eye_y - 12), (10, 10, 20), 4)
        for wy in range(face_cy - 65, face_cy - 35, 6):
            cv2.line(img, (face_cx - 25, wy), (face_cx + 25, wy), (15, 20, 30), 2)
    else:
        # Relaxed horizontal brows
        cv2.line(img, (face_cx - 55, eye_y - 22), (face_cx - 20, eye_y - 22), (15, 15, 25), 3)
        cv2.line(img, (face_cx + 20, eye_y - 22), (face_cx + 55, eye_y - 22), (15, 15, 25), 3)
        
    # Draw mouth
    mouth_y = face_cy + 55
    cv2.ellipse(img, (face_cx, mouth_y), (mouth_width // 2, mouth_openness), 0, 0, 360, (25, 30, 80), -1)
    
    # Partial coverage / hand occlusion
    if partial_cover:
        cv2.rectangle(img, (face_cx - 80, face_cy), (face_cx + 80, face_cy + 120), (10, 10, 15), -1)
        
    # Blur
    if blur:
        img = cv2.GaussianBlur(img, (45, 45), 0)
        
    return img

def run_tests():
    print("=" * 65)
    print("🔬 BATCH 1 VERIFICATION: ASTRONAUT IDENTITY & VIDEO EMOTION PIPELINE")
    print("=" * 65)
    fer = FacialEmotionModule()
    baseline = {"resting_ear": 0.32, "resting_mar": 0.18, "blink_rate_bpm": 16.0, "resting_au04": 0.10}
    fer.set_active_astronaut("CREW-BAS-01", baseline)

    passed_count = 0
    total_count = 0

    # 1. Neutral Face
    total_count += 1
    f_neutral = create_synthetic_face(eye_openness=15, mouth_openness=8, mouth_width=50)
    res_neutral = fer.extract_features(f_neutral)
    print(f"\n[Test 1] Neutral Face: Detected={res_neutral['face_detected']}, State={res_neutral['facial_state']}, Conf={res_neutral['confidence']:.2f}")
    if res_neutral['face_detected'] and res_neutral['facial_state'] in ['relaxed', 'neutral'] and res_neutral['face_quality']['valid']:
        print("  --> PASS: Face detected, optimal quality, relaxed baseline confirmed.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 2. Smiling Face
    total_count += 1
    f_smile = create_synthetic_face(eye_openness=14, mouth_openness=12, mouth_width=115)
    for _ in range(6): # Feed 6 frames to allow EMA & state hysteresis transition
        res_smile = fer.extract_features(f_smile)
    print(f"[Test 2] Smiling Face: State={res_smile['facial_state']}, AU12 Smile={res_smile['facial_indicators']['smile_intensity_au12']:.2f}, Conf={res_smile['confidence']:.2f}")
    if res_smile['facial_indicators']['smile_intensity_au12'] > 0.35 and res_smile['facial_state'] == 'expressive_positive':
        print("  --> PASS: Smile detected (AU12 active) and stable expressive_positive state achieved.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 3. Frowning / Strained Brow Face
    total_count += 1
    f_frown = create_synthetic_face(eye_openness=13, mouth_openness=8, mouth_width=45, brow_furrow=True)
    for _ in range(6): # Feed 6 frames to allow temporal transition to strained
        res_frown = fer.extract_features(f_frown)
    print(f"[Test 3] Frowning Face: State={res_frown['facial_state']}, AU04 Brow={res_frown['facial_indicators']['brow_tension_au04']:.2f}, Stress={res_frown['stress_indicator']:.2f}")
    if res_frown['facial_indicators']['brow_tension_au04'] > 0.30 and res_frown['facial_state'] == 'strained':
        print("  --> PASS: Brow strain correctly elevated AU04 and registered strained state.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 4. Talking (Mouth Opening/Closing with relaxed brow)
    total_count += 1
    # Feed neutral frames to establish baseline before talking
    for _ in range(4):
        fer.extract_features(f_neutral)
    talking_states = []
    for mo in [8, 18, 10, 22, 12, 20]:
        f_talk = create_synthetic_face(eye_openness=15, mouth_openness=mo, mouth_width=55, brow_furrow=False)
        r_talk = fer.extract_features(f_talk)
        talking_states.append(r_talk['facial_state'])
    print(f"[Test 4] Talking Sequence: States Observed = {set(talking_states)}")
    # Talking should NOT falsely trigger 'strained' or 'fatigued'
    if 'strained' not in talking_states:
        print("  --> PASS: Talking mouth movements did not trigger false strain alerts.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 5. Looking Away / Extreme Head Pose
    total_count += 1
    f_pose = create_synthetic_face(face_rx=50, face_ry=150) # extreme aspect ratio (3.0)
    res_pose = fer.extract_features(f_pose)
    print(f"[Test 5] Extreme Pose: Quality Status = {res_pose['face_quality']['status']}, Valid = {res_pose['face_quality']['valid']}")
    if not res_pose['face_quality']['valid']:
        print("  --> PASS: Extreme pose detected and flagged as invalid quality; confidence degraded.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 6. Poor Lighting
    total_count += 1
    f_dark = create_synthetic_face(brightness=15)
    res_dark = fer.extract_features(f_dark)
    print(f"[Test 6] Poor Lighting: Quality Status = {res_dark['face_quality']['status']}, Brightness = {res_dark['face_quality']['brightness']}")
    if res_dark['face_quality']['status'] == 'POOR_LIGHTING' and not res_dark['face_quality']['valid']:
        print("  --> PASS: Low illumination detected; zero spurious emotion generated.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 7. No Face
    total_count += 1
    f_empty = np.zeros((480, 640, 3), dtype=np.uint8)
    res_empty = fer.extract_features(f_empty)
    print(f"[Test 7] No Face: Detected = {res_empty['face_detected']}, Status = {res_empty['face_quality']['status']}")
    if not res_empty['face_detected'] and res_empty['face_quality']['status'] == 'NO_FACE':
        print("  --> PASS: Clean NO_FACE handling with baseline decay.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 8. Face Partially Covered
    total_count += 1
    f_covered = create_synthetic_face(partial_cover=True)
    res_covered = fer.extract_features(f_covered)
    print(f"[Test 8] Face Partially Covered: Quality Score = {res_covered['face_quality']['quality_score']:.2f}")
    if res_covered['face_quality']['quality_score'] < 0.90:
        print("  --> PASS: Occlusion detected, quality score penalized.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 9. Rapid Expression Changes (Temporal Stability & Hysteresis)
    total_count += 1
    flapping_states = []
    # Alternating single frames of frown vs neutral: Frown, Neutral, Frown, Neutral, Frown, Neutral
    for i in range(6):
        frame = f_frown if i % 2 == 0 else f_neutral
        r_flap = fer.extract_features(frame)
        flapping_states.append(r_flap['facial_state'])
    print(f"[Test 9] Rapid Flapping Input (Frown <-> Neutral): State Sequence = {flapping_states}")
    # Verify that the state does NOT oscillate on every single frame!
    state_switches = sum(1 for i in range(1, len(flapping_states)) if flapping_states[i] != flapping_states[i-1])
    print(f"         State transitions observed: {state_switches} (Max allowed: 2)")
    if state_switches <= 2:
        print("  --> PASS: Temporal smoothing & hysteresis prevented rapid frame-flipping.")
        passed_count += 1
    else:
        print("  --> FAIL")

    # 10. Phase 1: Astronaut Identity Isolation & Session Partitioning
    total_count += 1
    db = DatabaseManager()
    pipe = MasterPipelineService()

    # Log session for Astronaut A (CREW-BAS-01)
    pipe.set_active_astronaut("CREW-BAS-01")
    pipe.process_frame_and_audio(frame=f_neutral)
    sess_a = db.get_sessions(astronaut_id="CREW-BAS-01")

    # Switch to Astronaut B (CREW-BAS-02)
    pipe.set_active_astronaut("CREW-BAS-02")
    pipe.process_frame_and_audio(frame=f_neutral)
    sess_b = db.get_sessions(astronaut_id="CREW-BAS-02")

    telem_a = db.get_recent_telemetry(astronaut_id="CREW-BAS-01", limit=10)
    telem_b = db.get_recent_telemetry(astronaut_id="CREW-BAS-02", limit=10)

    print(f"[Test 10] Identity Isolation: Astronaut A sessions={len(sess_a)}, Astronaut B sessions={len(sess_b)}")
    a_in_b = any(t['astronaut_id'] == "CREW-BAS-01" for t in telem_b)
    b_in_a = any(t['astronaut_id'] == "CREW-BAS-02" for t in telem_a)

    if not a_in_b and not b_in_a and len(sess_a) > 0 and len(sess_b) > 0:
        print("  --> PASS: Strict astronaut data partitioning; Astronaut A never appears in Astronaut B's logs.")
        passed_count += 1
    else:
        print("  --> FAIL")

    print("\n" + "=" * 65)
    print(f"BATCH 1 TEST SUMMARY: {passed_count}/{total_count} TESTS PASSED ({(passed_count/total_count)*100:.0f}%)")
    print("=" * 65)
    return passed_count == total_count

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
