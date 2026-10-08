"""
SecureVision - Automated Verification Suite for Phase 4
Tests:
  1. Polygon configuration persistence (saving & loading from zones.json).
  2. Outside zone state: Person outside polygon -> No breach, normal zone overlay.
  3. Inside zone state: Person inside polygon -> RESTRICTED AREA BREACH, bright red border.
  4. Cooldown logic: Consecutive breaches within ALERT_COOLDOWN_SECONDS suppressed.
  5. Real human detection in/out of zone with actual computed bottom-center coordinates.
"""

import os
import time
import json
import cv2
import numpy as np
import config
from core.camera import Camera
from core.person_detector import PersonDetector
from core.zone import ZoneManager


def test_zone_persistence():
    print("\n--- TEST 1: Zone Configuration File Persistence (zones.json) ---")
    zone_mgr = ZoneManager()

    test_points = [[100, 150], [450, 150], [500, 450], [80, 450]]
    test_name = "Server Room Vault"

    # Save custom zone
    success = zone_mgr.save_zone(test_points, name=test_name)
    assert success, "Failed to save zone to zones.json."
    assert os.path.exists(config.ZONES_FILE), f"File {config.ZONES_FILE} does not exist."

    # Load in new manager
    new_mgr = ZoneManager()
    assert new_mgr.points == test_points, f"Loaded points {new_mgr.points} did not match saved {test_points}."
    assert new_mgr.zone_name == test_name, f"Loaded name '{new_mgr.zone_name}' did not match '{test_name}'."
    print(f"[PASS] Successfully saved and reloaded {len(test_points)} vertices for '{test_name}'.")


def test_outside_zone():
    print("\n--- TEST 2: Person Outside Zone (No Breach) ---")
    zone_mgr = ZoneManager()
    # Define restricted zone on right half of frame: x from 350 to 600, y from 200 to 460
    zone_mgr.save_zone([[350, 200], [600, 200], [600, 460], [350, 460]], name="Secure Area")

    # Person located on LEFT side: x1=50, y1=100, x2=200, y2=400 -> bottom-center = (125, 400)
    person_box = (50, 100, 200, 400, 0.88)
    is_breached, breach_pts, is_new_alert = zone_mgr.check_intrusion([person_box])

    print(f"[PASS] Outside zone result: Breached={is_breached}, Points={breach_pts}")
    assert not is_breached, "Person outside zone was incorrectly marked as breached!"
    assert len(breach_pts) == 0, "No breach points should be returned."

    # Render normal zone frame
    frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 40, dtype=np.uint8)
    annotated = zone_mgr.draw_zone(frame, is_breached, breach_pts)
    cv2.imwrite("evidence/phase4_zone_normal.jpg", annotated)
    print("[PASS] Normal zone evidence saved to evidence/phase4_zone_normal.jpg")


def test_inside_zone_and_cooldown():
    print("\n--- TEST 3 & 4: Person Inside Zone (Breach Alert) & Cooldown Verification ---")
    zone_mgr = ZoneManager(cooldown_seconds=3.0)
    # Define zone in center: x from 200 to 500, y from 200 to 460
    zone_mgr.save_zone([[200, 200], [500, 200], [500, 460], [200, 460]], name="Hazard Zone")

    # Person whose feet land INSIDE: x1=250, y1=100, x2=450, y2=380 -> bottom-center = (350, 380)
    intruding_box = (250, 100, 450, 380, 0.92)

    # Frame 1: First breach -> must trigger alert
    is_breached, breach_pts, is_new_alert = zone_mgr.check_intrusion([intruding_box])
    print(f"[PASS] First breach result: Breached={is_breached}, Breach Points={breach_pts}, New Alert={is_new_alert}")
    assert is_breached, "Person inside zone was not detected!"
    assert len(breach_pts) == 1 and breach_pts[0] == (350, 380), "Breach point mismatch."
    assert is_new_alert, "First breach event should trigger is_new_alert = True."

    # Render breached frame
    frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 40, dtype=np.uint8)
    annotated = zone_mgr.draw_zone(frame, is_breached, breach_pts)
    cv2.imwrite("evidence/phase4_zone_breached.jpg", annotated)
    print("[PASS] Breached evidence saved to evidence/phase4_zone_breached.jpg")

    # Frame 2 (immediate next frame): Still inside zone, but within cooldown window -> is_new_alert must be False
    is_breached2, _, is_new_alert2 = zone_mgr.check_intrusion([intruding_box])
    print(f"[PASS] Cooldown suppression: Breached={is_breached2}, New Alert={is_new_alert2}")
    assert is_breached2, "Should remain breached while person is in zone."
    assert not is_new_alert2, "Alert cooldown failed: duplicate alert triggered immediately!"

    # Simulate waiting past cooldown
    print("Testing alert re-trigger after cooldown expiry...")
    zone_mgr.last_alert_time = time.time() - 4.0  # simulate 4 seconds elapsed (> 3.0s cooldown)
    _, _, is_new_alert3 = zone_mgr.check_intrusion([intruding_box])
    assert is_new_alert3, "Alert should re-trigger once cooldown expires."
    print("[PASS] Cooldown verified: Alert re-triggered after cooldown elapsed.")


def test_real_human_in_and_out_of_zone():
    print("\n--- TEST 5: Real Camera Human Detection In vs Out of Zone ---")
    cam = Camera(source=0)
    assert cam.is_opened(), "Could not open camera."

    detector = PersonDetector(process_every_n=1)
    zone_mgr = ZoneManager()

    # Capture frames until person is detected (or fall back to multi_person_demo.mp4)
    boxes, count = [], 0
    for _ in range(15):
        ret, frame = cam.read()
        if ret:
            boxes, count = detector.detect(frame)
            if count >= 1 and len(boxes) > 0:
                break
    cam.release()

    if count == 0:
        # Fallback to recorded video with real persons if user stepped away from webcam
        print("[Info] No person currently in webcam frame, evaluating on multi_person_demo.mp4...")
        vid_cam = Camera(source="multi_person_demo.mp4")
        ret, frame = vid_cam.read()
        boxes, count = detector.detect(frame)
        vid_cam.release()

    assert count >= 1 and len(boxes) > 0, "No human detected in frame."
    x1, y1, x2, y2, conf = boxes[0]
    person_bottom_center = (int((x1 + x2) / 2), int(y2))
    print(f"Detected real person: Box=({x1}, {y1}, {x2}, {y2}), Feet Point={person_bottom_center}")

    # Case A: Zone encompassing the person's feet (Breach)
    px, py = person_bottom_center
    zone_in = [[px - 80, py - 60], [px + 80, py - 60], [px + 80, py + 20], [px - 80, py + 20]]
    zone_mgr.save_zone(zone_in, name="Active User Zone")
    breached_in, pts_in, _ = zone_mgr.check_intrusion(boxes)
    assert breached_in, f"Feet at {person_bottom_center} should be inside zone {zone_in}."
    print(f"[PASS] Walked-in test: Correctly triggered RESTRICTED AREA BREACH at {pts_in}.")

    # Case B: Zone away from person's feet (Walked-out / Normal)
    # Put zone far away (e.g. top corner)
    zone_out = [[10, 50], [100, 50], [100, 150], [10, 150]]
    zone_mgr.save_zone(zone_out, name="Remote Zone")
    breached_out, pts_out, _ = zone_mgr.check_intrusion(boxes)
    assert not breached_out, f"Feet at {person_bottom_center} should be OUTSIDE zone {zone_out}."
    print("[PASS] Walked-out test: Correctly reported NO BREACH when outside zone.")


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 4 Verification Suite")
    print("=" * 60)

    test_zone_persistence()
    test_outside_zone()
    test_inside_zone_and_cooldown()
    test_real_human_in_and_out_of_zone()

    print("\n" + "=" * 60)
    print(" ALL PHASE 4 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
