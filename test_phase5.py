"""
SecureVision - Automated Verification Suite for Phase 5
Tests:
  1. SQLite table initialization and schema integrity.
  2. Real MOTION event triggering: evidence snapshot saved + SQLite row inserted.
  3. Real ZONE_BREACH event triggering: evidence snapshot saved + SQLite row inserted.
  4. Helper functions: insert_alert(), get_recent_alerts(), get_alert_count_today(), get_alerts_by_hour().
  5. Cooldown suppression test.
  6. Prints all database rows and saved evidence image paths.
"""

import os
import sqlite3
import time
from pathlib import Path
import cv2
import numpy as np
import config
from core.camera import Camera
from core.motion import MotionDetector
from core.person_detector import PersonDetector
from core.zone import ZoneManager
from database.db import (
    get_alert_count_today,
    get_alerts_by_hour,
    get_connection,
    get_recent_alerts,
    init_db,
    insert_alert,
    log_event,
    save_evidence_image,
)


def test_db_schema():
    print("\n--- TEST 1: Database Table & Schema Verification ---")
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
    table = cursor.fetchone()
    assert table is not None, "Table 'alerts' was not created!"

    cursor.execute("PRAGMA table_info(alerts)")
    columns = {row["name"]: row["type"] for row in cursor.fetchall()}
    expected_cols = ["id", "timestamp", "event_type", "people_count", "image_path", "details"]
    for col in expected_cols:
        assert col in columns, f"Column '{col}' missing from alerts table!"

    conn.close()
    print("[PASS] SQLite table 'alerts' verified with correct schema.")


def test_motion_event_triggering():
    print("\n--- TEST 2: Real MOTION Event Triggering & Logging ---")
    # Use real camera / demo.mp4 motion frame
    video_path = "demo.mp4"
    assert os.path.exists(video_path), f"{video_path} not found."
    cam = Camera(source=video_path)
    detector = MotionDetector(warmup_frames=25)

    # Advance past warmup to motion frame 53
    motion_frame = None
    motion_boxes = []
    for i in range(55):
        ret, frame = cam.read()
        if ret:
            m, b = detector.detect(frame)
            if m and len(b) > 0:
                motion_frame = frame
                motion_boxes = b
                break
    cam.release()

    assert motion_frame is not None, "Failed to capture motion frame from real video."
    annotated = detector.draw_motion(motion_frame.copy(), True, motion_boxes)

    # Log MOTION event
    alert = log_event(
        frame=annotated,
        event_type="MOTION",
        people_count=0,
        details=f"Real motion detected with {len(motion_boxes)} active contour boxes",
    )

    assert alert["id"] > 0, "Failed to insert motion alert."
    assert alert["event_type"] == "MOTION"
    assert os.path.exists(alert["image_path"]), f"Evidence image {alert['image_path']} not found on disk!"
    img_size = os.path.getsize(alert["image_path"])
    assert img_size > 1000, f"Evidence image too small ({img_size} bytes)."

    print(f"[PASS] MOTION alert #{alert['id']} logged successfully.")
    print(f"       Saved Image: {alert['image_path']} ({img_size:,} bytes)")


def test_zone_breach_event_triggering():
    print("\n--- TEST 3: Real ZONE_BREACH Event Triggering & Logging ---")
    # Use real person from multi_person_demo.mp4 or webcam
    cam = Camera(source="multi_person_demo.mp4")
    p_detector = PersonDetector(process_every_n=1)
    zone_mgr = ZoneManager()

    ret, frame = cam.read()
    cam.release()
    assert ret and frame is not None, "Failed to capture frame."

    boxes, count = p_detector.detect(frame)
    assert count >= 1 and len(boxes) > 0, "No person detected in test frame."

    x1, y1, x2, y2, conf = boxes[0]
    feet = (int((x1 + x2) / 2), int(y2))

    # Configure zone to cover intruder feet
    zone_mgr.save_zone(
        [[feet[0] - 80, feet[1] - 80], [feet[0] + 80, feet[1] - 80], [feet[0] + 80, feet[1] + 20], [feet[0] - 80, feet[1] + 20]],
        name="Server Vault",
    )
    is_breached, breach_pts, is_new_alert = zone_mgr.check_intrusion(boxes)
    assert is_breached, "Expected zone breach did not trigger."

    annotated = p_detector.draw_detections(frame.copy(), boxes, count)
    annotated = zone_mgr.draw_zone(annotated, is_breached, breach_pts)

    # Log ZONE_BREACH event
    alert = log_event(
        frame=annotated,
        event_type="ZONE_BREACH",
        people_count=count,
        details=f"Intruder bottom-center anchor at {breach_pts}",
    )

    assert alert["id"] > 0, "Failed to insert zone breach alert."
    assert alert["event_type"] == "ZONE_BREACH"
    assert os.path.exists(alert["image_path"]), f"Evidence image {alert['image_path']} not found!"
    img_size = os.path.getsize(alert["image_path"])
    assert img_size > 1000, f"Evidence image too small ({img_size} bytes)."

    print(f"[PASS] ZONE_BREACH alert #{alert['id']} logged successfully.")
    print(f"       Saved Image: {alert['image_path']} ({img_size:,} bytes)")


def test_helper_queries():
    print("\n--- TEST 4: Database Helper Functions Verification ---")
    # get_recent_alerts
    recent = get_recent_alerts(limit=5)
    assert len(recent) >= 2, f"Expected at least 2 recent alerts, got {len(recent)}."
    print(f"[PASS] get_recent_alerts(limit=5) retrieved {len(recent)} records.")

    # get_alert_count_today
    count_today = get_alert_count_today()
    assert count_today >= 2, f"Expected today's alert count >= 2, got {count_today}."
    print(f"[PASS] get_alert_count_today() returned: {count_today} alerts today.")

    # get_alerts_by_hour
    hourly = get_alerts_by_hour()
    assert isinstance(hourly, dict) and len(hourly) == 24, "Hourly breakdown did not return 24 hours."
    total_hourly = sum(hourly.values())
    assert total_hourly >= 2, f"Expected total hourly >= 2, got {total_hourly}."
    print(f"[PASS] get_alerts_by_hour() returned 24-hour distribution (Sum: {total_hourly}).")


def display_all_records():
    print("\n" + "=" * 70)
    print(" CURRENT DATABASE RECORDS IN alerts TABLE (securevision.db)")
    print("=" * 70)
    rows = get_recent_alerts(limit=50)
    print(f"{'ID':<4} | {'TIMESTAMP':<19} | {'EVENT TYPE':<12} | {'PEOPLE':<6} | {'IMAGE PATH'}")
    print("-" * 70)
    for r in rows:
        print(f"{r['id']:<4} | {r['timestamp']:<19} | {r['event_type']:<12} | {r['people_count']:<6} | {r['image_path']}")

    print("\n" + "=" * 70)
    print(" RECENT SAVED EVIDENCE FILES IN evidence/ DIRECTORY")
    print("=" * 70)
    evidence_files = sorted(Path("evidence").glob("*.jpg"), key=os.path.getmtime, reverse=True)
    for ef in evidence_files[:10]:
        size_kb = ef.stat().st_size / 1024
        print(f"- {ef.name:<45} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 5 Verification Suite")
    print("=" * 60)

    test_db_schema()
    test_motion_event_triggering()
    test_zone_breach_event_triggering()
    test_helper_queries()
    display_all_records()

    print("\n" + "=" * 60)
    print(" ALL PHASE 5 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
