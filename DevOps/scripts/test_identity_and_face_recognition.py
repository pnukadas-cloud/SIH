"""
DevOps Script — Identity System & Biometric Face Recognition Verification
Tests:
1. Database-backed Astronaut Profiles (AST-001 Aryan, AST-002 Riya, AST-003 Karan)
2. Biometric Face Embedding Extraction (128-D L2 Unit Vectors)
3. 5 Identity States (IDENTIFIED, UNKNOWN, LOW_CONFIDENCE, NO_FACE, MULTIPLE_FACES)
4. Dynamic Astronaut Enrollment Flow (/api/auth/enroll)
5. Automatic Profile & Session Loading (Aryan -> Riya switch)
6. Strict Telemetry, Dialogue & Session Data Isolation
7. Database-backed Authentication (/api/auth/login)
"""

import os
import sys
import time
import json
import numpy as np
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Backend_DB.database.connection import DatabaseManager
from AIML.facial_emotion.face_recognizer import FaceRecognizer
from Backend_DB.services.pipeline_service import MasterPipelineService
from DevOps.scripts.test_batch1_video_and_identity import create_synthetic_face

def run_identity_tests():
    print("=" * 68)
    print("🚀 MAITRI IDENTITY SYSTEM & BIOMETRIC FACE RECOGNITION VERIFICATION")
    print("=" * 68)

    db = DatabaseManager()
    recognizer = FaceRecognizer()
    pipeline = MasterPipelineService()

    passed = 0
    total = 0

    # -------------------------------------------------------------
    # Test 1: Database-Backed Astronaut Profiles
    # -------------------------------------------------------------
    total += 1
    astronauts = db.list_astronauts()
    ids = [a["astronaut_id"] for a in astronauts]
    print(f"\n[Test 1] Enrolled Astronauts in SQLite: {ids}")
    if "AST-001" in ids and "AST-002" in ids and "AST-003" in ids:
        astro1 = db.get_astronaut("AST-001")
        print(f"  --> PASS: Found {len(astronauts)} database-backed profiles. AST-001: {astro1['name']} ({astro1['role']})")
        passed += 1
    else:
        print("  --> FAIL: Missing seeded astronauts in SQLite database.")

    # -------------------------------------------------------------
    # Test 2: Biometric Face Embedding (128-D L2 Unit Vector)
    # -------------------------------------------------------------
    total += 1
    f_sample = create_synthetic_face()
    emb = recognizer.extract_face_embedding(f_sample)
    if emb and len(emb) == 128 and abs(np.linalg.norm(emb) - 1.0) < 1e-3:
        print(f"[Test 2] Face Embedding Extraction: Dims={len(emb)}, Norm={np.linalg.norm(emb):.4f}")
        print("  --> PASS: 128-D spatial LBP and multi-spectral chrominance vector generated.")
        passed += 1
    else:
        print(f"  --> FAIL: Invalid embedding output: {emb}")

    # -------------------------------------------------------------
    # Test 3: Biometric Enrollment Flow (Enroll New Astronaut)
    # -------------------------------------------------------------
    total += 1
    new_astro_id = f"AST-TEST-{int(time.time())}"
    enroll_img = create_synthetic_face(face_rx=105, face_ry=125, eye_openness=14, mouth_width=65)
    test_embedding = recognizer.extract_face_embedding(enroll_img)
    
    import hashlib
    db.upsert_astronaut(
        astronaut_id=new_astro_id,
        name="Cmdr. Vikram Sen",
        callsign="SURYA-TEST",
        role="Payload Specialist",
        username=new_astro_id.lower(),
        password_hash=hashlib.sha256("testpass123".encode()).hexdigest(),
        profile={"mission_assignment": "Expedition 1", "clearance_level": "Level-2"},
        baseline={"resting_heart_rate_bpm": 65, "resting_ear": 0.33},
        face_embedding=test_embedding
    )
    enrolled_record = db.get_astronaut(new_astro_id)
    if enrolled_record and enrolled_record["name"] == "Cmdr. Vikram Sen" and len(enrolled_record["face_embedding"]) == 128:
        print(f"[Test 3] Dynamic Astronaut Enrollment: Enrolled {enrolled_record['name']} ({new_astro_id})")
        print("  --> PASS: Profile and biometric representation successfully saved to SQLite.")
        passed += 1
    else:
        print("  --> FAIL: Enrollment record not found or incomplete.")

    # -------------------------------------------------------------
    # Test 4: Identity State: IDENTIFIED
    # -------------------------------------------------------------
    total += 1
    res_id = recognizer.recognize_face(enroll_img)
    print(f"[Test 4] State: IDENTIFIED -> Status={res_id['status']}, Astronaut={res_id.get('name')}, Conf={res_id.get('confidence')}")
    if res_id["status"] == "IDENTIFIED" and res_id["astronaut_id"] == new_astro_id:
        print("  --> PASS: Known face recognized accurately with high confidence.")
        passed += 1
    else:
        print(f"  --> FAIL: Expected IDENTIFIED for {new_astro_id}, got: {res_id}")

    # -------------------------------------------------------------
    # Test 5: Identity State: UNKNOWN
    # -------------------------------------------------------------
    total += 1
    # Save a clean database state with only known astronauts
    # Create an alien synthetic face whose features differ significantly
    f_unknown = np.zeros((480, 640, 3), dtype=np.uint8)
    f_unknown[:, :] = (80, 20, 10)
    cv2.ellipse(f_unknown, (320, 240), (60, 80), 0, 0, 360, (40, 80, 140), -1)
    res_unknown = recognizer.recognize_face(f_unknown)
    print(f"[Test 5] State: UNKNOWN -> Status={res_unknown['status']}, Msg='{res_unknown.get('message')}'")
    if res_unknown["status"] in ["UNKNOWN", "LOW_CONFIDENCE"]:
        print("  --> PASS: Unmatched face correctly rejected as UNKNOWN/LOW_CONFIDENCE; zero false assignment.")
        passed += 1
    else:
        print(f"  --> FAIL: Unexpected match: {res_unknown}")

    # -------------------------------------------------------------
    # Test 6: Identity State: NO_FACE
    # -------------------------------------------------------------
    total += 1
    f_empty = np.zeros((480, 640, 3), dtype=np.uint8)
    res_noface = recognizer.recognize_face(f_empty)
    print(f"[Test 6] State: NO_FACE -> Status={res_noface['status']}, Msg='{res_noface.get('message')}'")
    if res_noface["status"] == "NO_FACE":
        print("  --> PASS: Empty camera frame yields NO_FACE.")
        passed += 1
    else:
        print(f"  --> FAIL: Expected NO_FACE, got: {res_noface}")

    # -------------------------------------------------------------
    # Test 7: Identity State: MULTIPLE_FACES
    # -------------------------------------------------------------
    total += 1
    f_multi = create_synthetic_face()
    # Draw a second face
    cv2.ellipse(f_multi, (120, 240), (60, 80), 0, 0, 360, (100, 140, 190), -1)
    res_multi = recognizer.recognize_face(f_multi)
    print(f"[Test 7] State: MULTIPLE_FACES -> Status={res_multi['status']}, Msg='{res_multi.get('message')}'")
    if res_multi["status"] == "MULTIPLE_FACES":
        print("  --> PASS: Multiple faces detected; identity confirmation aborted for safety.")
        passed += 1
    else:
        print(f"  --> FAIL: Expected MULTIPLE_FACES, got: {res_multi}")

    # -------------------------------------------------------------
    # Test 8: Automatic Profile & Session Switch (Aryan -> Riya)
    # -------------------------------------------------------------
    total += 1
    # 1. Switch to Aryan (AST-001)
    astro_aryan = pipeline.set_active_astronaut("AST-001")
    session_aryan = pipeline.current_session_id
    pipeline.interact("MAITRI, I am reporting for my orbital shift.")
    
    # 2. Switch to Riya (AST-002)
    astro_riya = pipeline.set_active_astronaut("AST-002")
    session_riya = pipeline.current_session_id
    pipeline.interact("MAITRI, checklist status for cooling loop.")

    print(f"[Test 8] Astronaut Session Switch: Aryan Session={session_aryan} | Riya Session={session_riya}")
    if (session_aryan != session_riya and 
        astro_aryan["name"] == "Aryan" and 
        astro_riya["name"] == "Riya" and 
        pipeline.active_astronaut["astronaut_id"] == "AST-002"):
        print("  --> PASS: Automatic profile transition; previous session closed, new isolated session started.")
        passed += 1
    else:
        print("  --> FAIL: Session transition failed.")

    # -------------------------------------------------------------
    # Test 9: Data Isolation: Aryan vs Riya Dialogues & Sessions
    # -------------------------------------------------------------
    total += 1
    aryan_sessions = db.get_sessions("AST-001")
    riya_sessions = db.get_sessions("AST-002")
    aryan_dialogues = db.get_dialogues("AST-001")
    riya_dialogues = db.get_dialogues("AST-002")

    aryan_in_riya_sessions = any(s["astronaut_id"] == "AST-001" for s in riya_sessions)
    riya_in_aryan_sessions = any(s["astronaut_id"] == "AST-002" for s in aryan_sessions)
    
    aryan_dialogue_texts = [d["message"] for d in aryan_dialogues]
    riya_dialogue_texts = [d["message"] for d in riya_dialogues]

    print(f"[Test 9] Data Isolation: Aryan sessions={len(aryan_sessions)}, Riya sessions={len(riya_sessions)}")
    print(f"         Aryan dialogues={len(aryan_dialogues)}, Riya dialogues={len(riya_dialogues)}")

    if (not aryan_in_riya_sessions and 
        not riya_in_aryan_sessions and 
        "MAITRI, I am reporting for my orbital shift." in aryan_dialogue_texts and
        "MAITRI, checklist status for cooling loop." in riya_dialogue_texts and
        "MAITRI, I am reporting for my orbital shift." not in riya_dialogue_texts):
        print("  --> PASS: Strict data partitioning confirmed; Aryan's data never appears in Riya's logs.")
        passed += 1
    else:
        print("  --> FAIL: Data leaked across astronaut boundaries.")

    # -------------------------------------------------------------
    # Test 10: Database-Backed Authentication (/api/auth/login)
    # -------------------------------------------------------------
    total += 1
    from Security_API.authentication.auth_manager import AuthManager
    auth_aryan = AuthManager.verify_credentials("aryan", "password123")
    auth_riya = AuthManager.verify_credentials("AST-002", "password123")
    auth_bad = AuthManager.verify_credentials("aryan", "wrongpass")

    print(f"[Test 10] Database Credentials Verification: Aryan={bool(auth_aryan)}, Riya={bool(auth_riya)}, BadPass={bool(auth_bad)}")
    if auth_aryan and auth_riya and not auth_bad and auth_aryan["user_id"] == "AST-001":
        print("  --> PASS: Database-backed authentication successful with SHA-256 password security.")
        passed += 1
    else:
        print("  --> FAIL: Credential verification error.")

    print("\n" + "=" * 68)
    print(f"SUMMARY: {passed}/{total} IDENTITY TESTS PASSED ({(passed/total)*100:.0f}%)")
    print("=" * 68)
    return passed == total

if __name__ == "__main__":
    success = run_identity_tests()
    sys.exit(0 if success else 1)
