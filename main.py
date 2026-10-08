"""
SecureVision - Main Live Monitoring Feed (Phases 1 - 6)
Integrates:
  - Real-time video capture (webcam or video file) with auto-looping
  - Dynamic HUD overlay (FPS, timestamp, brand banner)
  - MOG2 background subtraction motion detection with noise filtering
  - Ultralytics YOLOv8n person detection (class 0) with periodic inference
  - Geometric restricted zone polygon intrusion detection with cooldown
  - Automated evidence snapshot storage and SQLite alert logging
  - Camera tampering detection (covered, blurred, redirected) with persistence
Controls:
  - 'q': Quit application
  - 't': Reset baseline reference frame for tampering detection
"""

import time
from datetime import datetime
import cv2
import config
from core.camera import Camera
from core.motion import MotionDetector
from core.person_detector import PersonDetector
from core.zone import ZoneManager
from core.tamper import TamperDetector
from database.db import init_db, log_event


def draw_overlay(frame, fps: float):
    """
    Draw a translucent top information bar with timestamp and real-time FPS.
    """
    height, width = frame.shape[:2]

    # Draw dark semi-transparent top header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Prepare text
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fps_str = f"FPS: {fps:.1f}"
    brand_str = "SECUREVISION [LIVE]"

    # Draw Brand/Status tag (Green/Cyan)
    cv2.putText(
        frame,
        brand_str,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 115),
        2,
        cv2.LINE_AA,
    )

    # Draw Timestamp (White)
    cv2.putText(
        frame,
        timestamp_str,
        (width // 2 - 80, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # Draw FPS (Yellow)
    cv2.putText(
        frame,
        fps_str,
        (width - 110, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def run_live_feed(source=None):
    """
    Run the live feed display loop with motion, person, zone intrusion,
    and camera tampering monitoring.
    :param source: Override camera source if provided.
    """
    effective_source = config.CAMERA_SOURCE if source is None else source
    print("=" * 60)
    print(" SecureVision - Smart Security Monitoring Feed")
    print("=" * 60)
    print(f"Opening camera source: {effective_source}")
    print("Controls: 'q' to quit | 't' to reset tamper reference frame.")

    # Initialize SQLite database
    init_db()

    cam = Camera(source=effective_source)
    if not cam.is_opened():
        print("[Error] Failed to open camera stream. Exiting.")
        return

    # Initialize detectors, zone manager, and tamper monitor
    motion_detector = MotionDetector()
    person_detector = PersonDetector()
    zone_manager = ZoneManager()
    tamper_detector = TamperDetector()

    window_name = "SecureVision - Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    fps = 0.0

    # Cooldown timer for motion events
    last_motion_alert_time = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("[Warning] Frame read failed or stream closed.")
                break

            # Calculate FPS using exponential moving average
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)

            # 1. Camera Tampering Detection (Phase 6)
            is_tampered, tamper_type, tamper_metrics, is_new_tamper = tamper_detector.detect(frame)

            # 2. Detect Motion (Phase 2)
            motion_detected, motion_boxes = motion_detector.detect(frame)

            # 3. Detect Persons (Phase 3)
            person_boxes, person_count = person_detector.detect(frame)

            # 4. Restricted Zone Intrusion (Phase 4)
            is_breached, breach_points, is_new_breach = zone_manager.check_intrusion(person_boxes)

            # 5. Render Annotations
            annotated_frame = motion_detector.draw_motion(frame, motion_detected, motion_boxes)
            annotated_frame = person_detector.draw_detections(annotated_frame, person_boxes, person_count)
            annotated_frame = zone_manager.draw_zone(annotated_frame, is_breached, breach_points)
            annotated_frame = tamper_detector.draw_tamper(annotated_frame, is_tampered, tamper_type, tamper_metrics)

            # 6. Draw Top HUD (Phase 1)
            annotated_frame = draw_overlay(annotated_frame, fps)

            # 7. Event Logging & Evidence Saving (Phase 5)
            now_sec = time.time()
            if is_new_tamper:
                # CAMERA TAMPERING EVENT
                log_event(
                    frame=annotated_frame,
                    event_type="TAMPER",
                    people_count=person_count,
                    details=f"Tampering detected: {tamper_type} | {tamper_metrics}",
                )

            elif is_new_breach:
                # ZONE BREACH EVENT
                log_event(
                    frame=annotated_frame,
                    event_type="ZONE_BREACH",
                    people_count=person_count,
                    details=f"Restricted zone breach at {breach_points}",
                )

            elif motion_detected and not motion_detector.is_warming_up:
                # MOTION EVENT (cooldown protected)
                if (now_sec - last_motion_alert_time) >= config.ALERT_COOLDOWN_SECONDS:
                    log_event(
                        frame=annotated_frame,
                        event_type="MOTION",
                        people_count=person_count,
                        details=f"Active motion boxes: {len(motion_boxes)}",
                    )
                    last_motion_alert_time = now_sec

            # Display frame
            cv2.imshow(window_name, annotated_frame)

            # Check keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit requested by user ('q' key pressed).")
                break
            elif key in (ord('t'), ord('T')):
                tamper_detector.reset_reference_frame()
                print("[User Action] Tamper reference frame reset to current view.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Camera released and display windows closed.")


if __name__ == "__main__":
    run_live_feed()
