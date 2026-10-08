"""
SecureVision - Automated Verification Suite for Phase 1
Tests:
  1. Real webcam initialization, frame capture, resolution, and release.
  2. Video file playback and automatic looping beyond EOF.
  3. Error resilience for invalid camera / non-existent video sources.
  4. Timestamp and FPS overlay rendering integrity.
"""

import os
import time
import numpy as np
from core.camera import Camera
import main
import config


def test_real_webcam():
    print("\n--- TEST 1: Real Webcam Capture (source=0) ---")
    cam = Camera(source=0)
    assert cam.is_opened(), "Failed: Real webcam (index 0) could not be opened."
    print("[PASS] Webcam opened successfully.")

    frames_read = 0
    start_time = time.perf_counter()
    for i in range(30):
        ret, frame = cam.read()
        assert ret, f"Failed: Could not read frame {i} from webcam."
        assert isinstance(frame, np.ndarray), "Failed: Frame is not a numpy array."
        assert frame.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), (
            f"Failed: Frame shape {frame.shape} does not match expected "
            f"({config.FRAME_HEIGHT}, {config.FRAME_WIDTH}, 3)."
        )
        frames_read += 1

    elapsed = time.perf_counter() - start_time
    measured_fps = frames_read / elapsed if elapsed > 0 else 0
    print(f"[PASS] Successfully read {frames_read} frames. Measured FPS: {measured_fps:.2f}")

    cam.release()
    assert not cam.is_opened(), "Failed: Camera not properly closed after release()."
    print("[PASS] Webcam released properly.")


def test_video_file_and_looping():
    print("\n--- TEST 2: Video File Input & Auto-Looping (demo.mp4) ---")
    video_path = "demo.mp4"
    assert os.path.exists(video_path), f"Video file {video_path} not found."

    cam = Camera(source=video_path)
    assert cam.is_opened(), f"Failed: Could not open video file {video_path}."
    print(f"[PASS] Video file '{video_path}' opened successfully.")

    # demo.mp4 contains 60 frames. Reading 90 frames forces loop-around.
    target_frames = 90
    frames_read = 0
    for i in range(target_frames):
        ret, frame = cam.read()
        assert ret, f"Failed: Reading frame {i} failed (looping mechanism failed)."
        assert frame is not None, f"Failed: Frame {i} is None."
        assert frame.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), (
            f"Frame shape {frame.shape} unexpected."
        )
        frames_read += 1

    print(f"[PASS] Successfully read {frames_read} frames across video loop cycle.")

    cam.release()
    assert not cam.is_opened(), "Failed: Video file camera not properly released."
    print("[PASS] Video source released properly.")


def test_error_handling():
    print("\n--- TEST 3: Error Resilience (Non-existent sources) ---")
    
    # Non-existent file
    cam_bad_file = Camera(source="non_existent_dummy_video.mp4")
    assert not cam_bad_file.is_opened(), "Failed: Non-existent file reported as opened."
    ret, frame = cam_bad_file.read()
    assert not ret and frame is None, "Failed: Read on unopened file did not return (False, None)."
    print("[PASS] Non-existent video file handled gracefully.")

    # High index invalid webcam
    cam_bad_idx = Camera(source=999)
    assert not cam_bad_idx.is_opened(), "Failed: Invalid camera index reported as opened."
    ret, frame = cam_bad_idx.read()
    assert not ret and frame is None, "Failed: Read on unopened camera index did not return (False, None)."
    print("[PASS] Non-existent webcam index handled gracefully.")


def test_overlay_rendering():
    print("\n--- TEST 4: Timestamp & FPS Overlay Verification ---")
    dummy_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    annotated = main.draw_overlay(dummy_frame.copy(), fps=29.8)

    assert annotated.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), "Overlay modified shape."
    # Check that overlay actually modified pixels (not blank black anymore)
    assert np.any(annotated > 0), "Overlay failed to render any pixels onto frame."
    print("[PASS] Timestamp, brand badge, and FPS overlay rendered successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 1 Verification Suite")
    print("=" * 60)

    test_real_webcam()
    test_video_file_and_looping()
    test_error_handling()
    test_overlay_rendering()

    print("\n" + "=" * 60)
    print(" ALL PHASE 1 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
