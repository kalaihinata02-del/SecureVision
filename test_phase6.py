"""
SecureVision - Automated Verification Suite for Phase 6
Tests:
  1. Covered/Blocked lens detection (dark/low variance frames).
  2. Blur/Defocus detection (Laplacian variance drop).
  3. Camera Moved/Redirected detection (histogram correlation against baseline).
  4. Temporal persistence confirmation (suppresses transient fluctuations).
  5. Resetting baseline reference frame.
  6. Database logging with event_type = TAMPER and evidence snapshot persistence.
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import config
from core.camera import Camera
from core.tamper import TamperDetector
from database.db import get_connection, get_recent_alerts, init_db, log_event


def test_covered_lens():
    print("\n--- TEST 1: Covered / Blocked Lens Detection ---")
    detector = TamperDetector()

    # Capture reference frame from webcam / demo
    cam = Camera(source=config.CAMERA_SOURCE)
    ret, normal_frame = cam.read()
    cam.release()
    assert ret and normal_frame is not None, "Failed to capture reference frame."

    # Establish baseline
    detector.set_reference_frame(normal_frame)

    # Simulate covered lens (hand blocking camera -> nearly zero brightness & variance)
    covered_frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 12, dtype=np.uint8)

    # Frame 1: Immediate frame -> must NOT trigger yet due to persistence requirement
    t0 = 1000.0
    is_t, t_type, metrics, _ = detector.detect(covered_frame, current_timestamp=t0)
    assert not is_t, f"Should not trigger immediately before {config.TAMPER_CONFIRM_SECONDS}s persistence."
    print(f"[PASS] Instantaneous frame: Suppressed (brightness={metrics['brightness']}, var={metrics['variance']}).")

    # Frame 2: Sustained condition past TAMPER_CONFIRM_SECONDS
    t1 = t0 + config.TAMPER_CONFIRM_SECONDS + 0.2
    is_t, t_type, metrics, is_alert = detector.detect(covered_frame, current_timestamp=t1)
    assert is_t and t_type == "COVERED_LENS", f"Expected COVERED_LENS, got {t_type}"
    print(f"[PASS] Sustained covered lens verified: Type={t_type}, Brightness={metrics['brightness']} < {config.TAMPER_DARK_MEAN_THRESHOLD}")

    # Draw and log event
    annotated = detector.draw_tamper(covered_frame.copy(), is_t, t_type, metrics)
    alert = log_event(
        frame=annotated,
        event_type="TAMPER",
        people_count=0,
        details=f"Tampering: {t_type} | Brightness: {metrics['brightness']}, Variance: {metrics['variance']}",
    )
    assert alert["id"] > 0
    print(f"[PASS] Covered lens evidence saved to: {alert['image_path']}")


def test_defocus_blur():
    print("\n--- TEST 2: Blur / Defocus Lens Detection ---")
    detector = TamperDetector()

    # Capture sharp baseline frame
    cam = Camera(source=config.CAMERA_SOURCE)
    ret, normal_frame = cam.read()
    cam.release()
    assert ret and normal_frame is not None

    detector.set_reference_frame(normal_frame)

    # Measure normal Laplacian variance
    gray_normal = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY)
    normal_lap = float(cv2.Laplacian(gray_normal, cv2.CV_64F).var())
    print(f"       Normal frame sharpness (Laplacian var): {normal_lap:.1f}")

    # Create realistically defocused/blurred frame (simulating optical blur or Vaseline on lens)
    blurred_frame = cv2.GaussianBlur(normal_frame, (51, 51), 0)
    gray_blur = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    blurred_lap = float(cv2.Laplacian(gray_blur, cv2.CV_64F).var())
    print(f"       Blurred frame sharpness (Laplacian var): {blurred_lap:.1f} (Threshold: {config.TAMPER_BLUR_LAPLACIAN_THRESHOLD})")
    assert blurred_lap < config.TAMPER_BLUR_LAPLACIAN_THRESHOLD, "Blurred frame didn't drop below blur threshold."

    # Sustained detection past confirm seconds
    t0 = 2000.0
    detector.detect(blurred_frame, current_timestamp=t0)
    t1 = t0 + config.TAMPER_CONFIRM_SECONDS + 0.2
    is_t, t_type, metrics, is_alert = detector.detect(blurred_frame, current_timestamp=t1)

    assert is_t and t_type == "DEFOCUS_BLUR", f"Expected DEFOCUS_BLUR, got {t_type}"
    print(f"[PASS] Sustained blur verified: Type={t_type}, LaplacianVar={metrics['laplacian_var']}")

    # Draw and log event
    annotated = detector.draw_tamper(blurred_frame.copy(), is_t, t_type, metrics)
    alert = log_event(
        frame=annotated,
        event_type="TAMPER",
        people_count=0,
        details=f"Tampering: {t_type} | Laplacian Var: {metrics['laplacian_var']}",
    )
    assert alert["id"] > 0
    print(f"[PASS] Blur evidence saved to: {alert['image_path']}")


def test_camera_redirected():
    print("\n--- TEST 3: Camera Moved / Redirected Detection ---")
    detector = TamperDetector()

    # Baseline frame (e.g. room / office view)
    ref_img = cv2.imread("evidence/zidane.jpg")
    ref_frame = cv2.resize(ref_img, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    detector.set_reference_frame(ref_frame)

    # Completely different view (simulating camera rotated away to blank ceiling or different scene)
    redirected_img = cv2.imread("evidence/bus.jpg")
    redirected_frame = cv2.resize(redirected_img, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

    # Test similarity
    hsv_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2HSV)
    hsv_redir = cv2.cvtColor(redirected_frame, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv_ref], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv_redir], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    print(f"       Scene correlation: {corr:.3f} (Threshold: {config.TAMPER_SCENE_SIMILARITY_THRESHOLD})")
    assert corr < config.TAMPER_SCENE_SIMILARITY_THRESHOLD, "Different scene similarity was too high."

    # Sustained detection past confirm seconds
    t0 = 3000.0
    detector.detect(redirected_frame, current_timestamp=t0)
    t1 = t0 + config.TAMPER_CONFIRM_SECONDS + 0.2
    is_t, t_type, metrics, is_alert = detector.detect(redirected_frame, current_timestamp=t1)

    assert is_t and t_type == "CAMERA_REDIRECTED", f"Expected CAMERA_REDIRECTED, got {t_type}"
    print(f"[PASS] Camera redirection verified: Type={t_type}, Similarity={metrics['scene_similarity']}")

    # Draw and log event
    annotated = detector.draw_tamper(redirected_frame.copy(), is_t, t_type, metrics)
    alert = log_event(
        frame=annotated,
        event_type="TAMPER",
        people_count=0,
        details=f"Tampering: {t_type} | Scene Similarity: {metrics['scene_similarity']}",
    )
    assert alert["id"] > 0
    print(f"[PASS] Redirected camera evidence saved to: {alert['image_path']}")


def test_reference_reset():
    print("\n--- TEST 4: Reference Frame Reset Capability ---")
    detector = TamperDetector()

    # Initial frame
    img1 = cv2.resize(cv2.imread("evidence/zidane.jpg"), (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    detector.set_reference_frame(img1)
    assert detector.reference_initialized

    # Reset
    detector.reset_reference_frame()
    assert not detector.reference_initialized, "Reference should be uninitialized after reset."

    # Next frame automatically becomes new baseline
    img2 = cv2.resize(cv2.imread("evidence/bus.jpg"), (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    is_t, _, _, _ = detector.detect(img2)
    assert detector.reference_initialized, "New frame should initialize baseline."
    assert not is_t, "New baseline should not be flagged as tampered."
    print("[PASS] Reference frame successfully reset and re-established.")


def display_tamper_db_records():
    print("\n" + "=" * 70)
    print(" TAMPERING ALERTS LOGGED IN SQLITE (securevision.db)")
    print("=" * 70)
    conn = get_connection()
    cursor = conn.execute(
        "SELECT id, timestamp, event_type, details, image_path FROM alerts WHERE event_type = 'TAMPER' ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    print(f"{'ID':<4} | {'TIMESTAMP':<19} | {'DETAILS':<35} | {'IMAGE PATH'}")
    print("-" * 70)
    for r in rows:
        details_snippet = (r["details"][:32] + "...") if len(r["details"]) > 32 else r["details"]
        print(f"{r['id']:<4} | {r['timestamp']:<19} | {details_snippet:<35} | {r['image_path']}")
    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 6 Verification Suite")
    print("=" * 60)

    test_covered_lens()
    test_defocus_blur()
    test_camera_redirected()
    test_reference_reset()
    display_tamper_db_records()

    print("\n" + "=" * 60)
    print(" ALL PHASE 6 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
