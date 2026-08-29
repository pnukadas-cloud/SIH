"""
DevOps — Automated Test Suite for MAITRI
Validates endpoints, authentication, RBAC authorization, telemetry, chat, and exports.
"""

import urllib.request
import urllib.error
import json
import time

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def make_request(path, method="GET", data=None, headers=None):
    url = f"{BASE_URL}{path}"
    headers = headers or {}
    if data and isinstance(data, dict):
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        body = data

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode(), resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers
    except Exception as e:
        return 0, str(e).encode(), {}

def run_all_tests():
    print("=" * 60)
    print("🚀 MAITRI AUTOMATED VERIFICATION TEST SUITE")
    print("=" * 60)
    passed = 0
    total = 0

    # 1. System Health
    total += 1
    code, body, _ = make_request("/api/status")
    if code == 200:
        print("✅ [1/9] System Health Endpoint: PASSED")
        passed += 1
    else:
        print(f"❌ [1/9] System Health Endpoint: FAILED (Code {code})")

    # 2. Login as Astronaut
    total += 1
    code, body, _ = make_request("/api/auth/login", method="POST", data={
        "username_or_id": "CREW-BAS-01",
        "password": "astronaut123"
    })
    ast_token = None
    if code == 200:
        res = json.loads(body.decode())
        ast_token = res.get("token")
        print(f"✅ [2/9] Astronaut Authentication: PASSED (Token issued)")
        passed += 1
    else:
        print(f"❌ [2/9] Astronaut Authentication: FAILED (Code {code})")

    # 3. Login as Flight Surgeon (Admin)
    total += 1
    code, body, _ = make_request("/api/auth/login", method="POST", data={
        "username_or_id": "ADMIN-MED-01",
        "password": "isro_surgeon2025"
    })
    adm_token = None
    if code == 200:
        res = json.loads(body.decode())
        adm_token = res.get("token")
        print(f"✅ [3/9] Flight Surgeon Authentication: PASSED (Admin token issued)")
        passed += 1
    else:
        print(f"❌ [3/9] Flight Surgeon Authentication: FAILED (Code {code})")

    # 4. RBAC Protection Check: Astronaut attempting to access Admin endpoint
    total += 1
    code, _, _ = make_request("/api/admin/crew-summary", headers={"X-MAITRI-Auth": ast_token or ""})
    if code == 403:
        print("✅ [4/9] RBAC Unauthorized Route Guard: PASSED (Astronaut blocked with HTTP 403)")
        passed += 1
    else:
        print(f"❌ [4/9] RBAC Unauthorized Route Guard: FAILED (Expected 403, got {code})")

    # 5. RBAC Authorized Check: Admin accessing Admin endpoint
    total += 1
    code, body, _ = make_request("/api/admin/crew-summary", headers={"X-MAITRI-Auth": adm_token or ""})
    if code == 200:
        print("✅ [5/9] RBAC Authorized Admin Access: PASSED (Admin granted access with HTTP 200)")
        passed += 1
    else:
        print(f"❌ [5/9] RBAC Authorized Admin Access: FAILED (Expected 200, got {code})")

    # 6. Flight Simulation Scenario
    total += 1
    code, body, _ = make_request("/api/simulate/docking_stress", method="POST")
    if code == 200:
        sim_data = json.loads(body.decode())
        dom = sim_data.get("fusion", {}).get("dominant_emotion")
        print(f"✅ [6/9] Simulation Pipeline: PASSED (Docking Stress -> Dominant: {dom})")
        passed += 1
    else:
        print(f"❌ [6/9] Simulation Pipeline: FAILED (Code {code})")

    # 7. MAITRI Conversational AI
    total += 1
    code, body, _ = make_request("/api/interact", method="POST", data={"message": "Station status report"})
    if code == 200:
        chat_data = json.loads(body.decode())
        reply = chat_data.get("ai_response", "")
        print(f"✅ [7/9] Conversational AI: PASSED (Response: {reply[:45]}...)")
        passed += 1
    else:
        print(f"❌ [7/9] Conversational AI: FAILED (Code {code})")

    # 8. Vector PDF Report Export
    total += 1
    code, pdf_bytes, hdrs = make_request("/api/export/pdf?astronaut_id=CREW-BAS-01")
    if code == 200 and pdf_bytes.startswith(b"%PDF"):
        print(f"✅ [8/9] Vector PDF Report Export: PASSED (Valid PDF header, size: {len(pdf_bytes)} bytes)")
        passed += 1
    else:
        print(f"❌ [8/9] Vector PDF Report Export: FAILED (Code {code})")

    # 9. Structured JSON & JPG Passport Exports
    total += 1
    code_j, json_bytes, _ = make_request("/api/export/json?astronaut_id=CREW-BAS-01")
    code_i, jpg_bytes, _ = make_request("/api/export/jpg?astronaut_id=CREW-BAS-01")
    if code_j == 200 and code_i == 200 and len(jpg_bytes) > 1000:
        print(f"✅ [9/9] JSON & JPG Passport Exports: PASSED (JSON size: {len(json_bytes)}b, JPG size: {len(jpg_bytes)}b)")
        passed += 1
    else:
        print(f"❌ [9/9] JSON & JPG Passport Exports: FAILED (JSON {code_j}, JPG {code_i})")

    print("=" * 60)
    print(f"RESULT: {passed}/{total} TESTS PASSED ({passed/total*100:.0f}%)")
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    run_all_tests()
