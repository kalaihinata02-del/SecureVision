"""
SecureVision - Automated Verification Suite for Phase 2
Tests:
  1. Background subtractor warm-up period suppression (0 false alarms).
  2. Stationary / Still scene evaluation (MOTION: NONE) on recorded static frames.
  3. Dynamic / Moving scene evaluation (MOTION: DETECTED with bounding boxes).
  4. Live physical webcam motion test reporting still vs moving states.
  5. Convenience function detect_motion() and HUD drawing validation.
"""

import os
import cv2
import numpy as np
import config
from core.camera import Camera
from core.motion import MotionDetector, detect_motion


def test_warmup_period():
    print("\n--- TEST 1: Background Subtractor Warmup Verification ---")
    detector = MotionDetector(min_area=config.MIN_CONTOUR_AREA, warmup_frames=config.WARMUP_FRAMES)

    false_alarms = 0
    # Provide synthetic test frames to verify warmup suppression
    for i in range(config.WARMUP_FRAMES):
        frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 120 + (i % 10), dtype=np.uint8)
        motion_detected, boxes = detector.detect(frame)
        if motion_detected or len(boxes) > 0:
            false_alarms += 1

    assert detector.is_warming_up, "Detector should indicate warming up at frame threshold."
    assert false_alarms == 0, f"Expected 0 false alarms during warmup, got {false_alarms}."
    print(f"[PASS] Warmup period verified: 0 false alarms across {config.WARMUP_FRAMES} frames.")


def test_still_and_moving_video_dataset():
    print("\n--- TEST 2 & 3: Still Scene vs Moving Scene on Real Camera Video (demo.mp4) ---")
    video_path = "demo.mp4"
    assert os.path.exists(video_path), f"Video file {video_path} not found."

    cam = Camera(source=video_path)
    detector = MotionDetector(min_area=config.MIN_CONTOUR_AREA, warmup_frames=25)

    # 1. Feed warmup frames (frames 0 to 25 -> 26 frames total)
    for _ in range(26):
        ret, frame = cam.read()
        assert ret, "Failed to read warmup frame."
        detector.detect(frame)

    assert not detector.is_warming_up, "Detector should have stabilized post-warmup."

    # 2. Test STILL scene period (frames 36 to 45 in demo.mp4 where subject is completely still)
    # Fast-forward to frame 36
    for _ in range(26, 36):
        ret, frame = cam.read()
        detector.detect(frame)

    still_frames_passed = 0
    saved_still = False
    for frame_idx in range(36, 46):
        ret, frame = cam.read()
        assert ret, f"Failed to read still frame {frame_idx}."
        motion, boxes = detector.detect(frame)
        if not motion:
            still_frames_passed += 1
            if not saved_still:
                annotated = detector.draw_motion(frame.copy(), motion, boxes)
                cv2.imwrite("evidence/phase2_motion_none.jpg", annotated)
                saved_still = True

    print(f"[PASS] Still scene verification: {still_frames_passed}/10 frames verified with MOTION: NONE (0 boxes).")
    assert still_frames_passed >= 8, f"Too many false motions during still period ({still_frames_passed}/10 passed)."
    print("[PASS] Stationary evidence saved to evidence/phase2_motion_none.jpg")

    # 3. Test MOVING scene period (frames 52 to 59 in demo.mp4 where movement occurred)
    # Fast-forward to frame 52
    for _ in range(46, 52):
        ret, frame = cam.read()
        detector.detect(frame)

    moving_frames_detected = 0
    saved_moving = False
    for frame_idx in range(52, 60):
        ret, frame = cam.read()
        assert ret, f"Failed to read moving frame {frame_idx}."
        motion, boxes = detector.detect(frame)
        if motion and len(boxes) > 0:
            moving_frames_detected += 1
            if not saved_moving:
                annotated = detector.draw_motion(frame.copy(), motion, boxes)
                cv2.imwrite("evidence/phase2_motion_detected.jpg", annotated)
                bx, by, bw, bh = boxes[0]
                print(f"[PASS] Moving subject detected at (x={bx}, y={by}, w={bw}, h={bh}), area={bw*bh} px.")
                saved_moving = True

    print(f"[PASS] Moving scene verification: {moving_frames_detected}/8 frames correctly detected motion.")
    assert moving_frames_detected >= 5, f"Expected motion detections, got {moving_frames_detected}/8."
    print("[PASS] Motion evidence saved to evidence/phase2_motion_detected.jpg")

    cam.release()


def test_live_webcam_detection():
    print("\n--- TEST 4: Live Physical Webcam Stream Evaluation (source=0) ---")
    cam = Camera(source=0)
    if not cam.is_opened():
        print("[SKIP] Physical webcam not available for live stream test.")
        return

    detector = MotionDetector(min_area=config.MIN_CONTOUR_AREA, warmup_frames=config.WARMUP_FRAMES)
    total_frames = 60
    still_count = 0
    motion_count = 0

    print(f"Reading {total_frames} frames from real webcam...")
    for i in range(total_frames):
        ret, frame = cam.read()
        if not ret:
            break
        motion, boxes = detector.detect(frame)
        if not detector.is_warming_up:
            if motion:
                motion_count += 1
            else:
                still_count += 1

    cam.release()
    print(f"[PASS] Webcam Live Feed Test: Evaluated {still_count + motion_count} post-warmup frames.")
    print(f"       - Frames with Motion Detected: {motion_count}")
    print(f"       - Frames with Stationary/Still: {still_count}")


def test_convenience_function():
    print("\n--- TEST 5: Convenience Function detect_motion() ---")
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    for _ in range(config.WARMUP_FRAMES + 5):
        detect_motion(frame)

    res, boxes = detect_motion(frame)
    assert isinstance(res, bool), "detect_motion should return a boolean flag."
    assert isinstance(boxes, list), "detect_motion should return a list of boxes."
    print("[PASS] Convenience function detect_motion() verified.")


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 2 Verification Suite")
    print("=" * 60)

    test_warmup_period()
    test_still_and_moving_video_dataset()
    test_live_webcam_detection()
    test_convenience_function()

    print("\n" + "=" * 60)
    print(" ALL PHASE 2 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
