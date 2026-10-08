"""
SecureVision - Automated Verification Suite for Phase 7
Tests:
  1. Flask authentication flow (unauthenticated redirect, invalid login, valid login, logout).
  2. Protected routes (/ and /alerts).
  3. API endpoints: /api/status, /api/alerts, /api/alerts/hourly, /api/tamper/reset.
  4. Video streaming endpoint (/video_feed) multipart MJPEG format.
  5. Evidence file serving (/evidence/<filename>).
  6. Thread-safe single camera background streaming & dynamic status changes.
"""

import json
import time
from pathlib import Path
import config
from app import app
from core.streamer import get_streamer


def test_auth_and_routes():
    print("\n--- TEST 1 & 2: Flask Authentication & Dashboard Routing ---")
    client = app.test_client()

    # 1. Unauthenticated request to / should redirect to /login
    res = client.get("/")
    assert res.status_code == 302, f"Expected 302 redirect for unauthenticated user, got {res.status_code}"
    assert "/login" in res.headers["Location"]
    print("[PASS] Unauthenticated access properly redirected to /login.")

    # 2. Invalid credentials login attempt
    res = client.post("/login", data={"username": "admin", "password": "wrongpassword"})
    assert res.status_code == 200
    assert b"Invalid administrator credentials" in res.data
    print("[PASS] Invalid credentials rejected.")

    # 3. Valid credentials login
    res = client.post(
        "/login",
        data={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"Live Video Stream" in res.data
    assert b"SecureVision" in res.data
    print("[PASS] Valid admin login succeeded and rendered dashboard.")

    # 4. Access alerts history page while logged in
    res = client.get("/alerts")
    assert res.status_code == 200
    assert b"Security Alert Logs & Evidence" in res.data
    print("[PASS] /alerts page rendered successfully with history table.")

    # 5. Logout
    res = client.get("/logout", follow_redirects=True)
    assert res.status_code == 200
    assert b"System Authentication" in res.data
    print("[PASS] Logout cleared session and redirected to login.")


def test_api_endpoints():
    print("\n--- TEST 3: REST API Endpoints Verification ---")
    client = app.test_client()
    # Log in
    client.post("/login", data={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD})

    # 1. /api/status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, dict), "api/status should return JSON object"
    required_keys = [
        "camera_online",
        "people_count",
        "motion",
        "zone_breach",
        "tamper_status",
        "alerts_today",
        "fps",
    ]
    for key in required_keys:
        assert key in data, f"Key '{key}' missing from /api/status response!"
    print(f"[PASS] /api/status returned live values: Camera={data['camera_online']}, People={data['people_count']}, Motion={data['motion']}, Breach={data['zone_breach']}, AlertsToday={data['alerts_today']}")

    # 2. /api/alerts
    res = client.get("/api/alerts?limit=10")
    assert res.status_code == 200
    alerts = res.get_json()
    assert isinstance(alerts, list), "/api/alerts should return a list"
    assert len(alerts) >= 1, "Expected alerts from previous phases"
    print(f"[PASS] /api/alerts returned {len(alerts)} alert records from SQLite.")

    # 3. /api/alerts/hourly
    res = client.get("/api/alerts/hourly")
    assert res.status_code == 200
    hourly = res.get_json()
    assert isinstance(hourly, dict) and len(hourly) == 24
    print(f"[PASS] /api/alerts/hourly returned 24-hour distribution.")

    # 4. /api/tamper/reset
    res = client.post("/api/tamper/reset")
    assert res.status_code == 200
    reset_data = res.get_json()
    assert reset_data.get("success"), "Tamper reset failed"
    print("[PASS] /api/tamper/reset executed successfully.")


def test_video_and_evidence_serving():
    print("\n--- TEST 4 & 5: Video Feed & Evidence Serving ---")
    client = app.test_client()

    # 1. Evidence image serving
    evidence_files = list(Path("evidence").glob("*.jpg"))
    assert len(evidence_files) > 0, "No evidence images found."
    test_file = evidence_files[0].name

    res = client.get(f"/evidence/{test_file}")
    assert res.status_code == 200
    assert res.mimetype == "image/jpeg"
    assert len(res.data) > 1000
    print(f"[PASS] /evidence/{test_file} served image ({len(res.data):,} bytes, mimetype=image/jpeg).")

    # 2. Video stream MJPEG verification
    res = client.get("/video_feed")
    assert res.status_code == 200
    assert "multipart/x-mixed-replace" in res.mimetype
    # Read first stream chunk
    chunk = next(res.response)
    assert b"--frame" in chunk or b"\xff\xd8" in chunk  # JPEG SOI marker
    print("[PASS] /video_feed serves valid multipart/x-mixed-replace MJPEG stream.")


def test_streamer_thread_safety():
    print("\n--- TEST 6: Thread-Safe Streamer Singleton Verification ---")
    s1 = get_streamer()
    s2 = get_streamer()
    assert s1 is s2, "VideoPipelineStreamer should be a singleton instance (camera opened only once)."
    assert s1.running, "Streamer worker thread should be actively running."

    # Give background thread a moment to compute frames
    time.sleep(1.0)
    status = s1.get_status()
    assert status["camera_online"], "Camera should be online in background streamer."
    print(f"[PASS] Background streamer running with camera online at {status['fps']} FPS.")


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 7 Verification Suite")
    print("=" * 60)

    test_auth_and_routes()
    test_api_endpoints()
    test_video_and_evidence_serving()
    test_streamer_thread_safety()

    print("\n" + "=" * 60)
    print(" ALL PHASE 7 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
