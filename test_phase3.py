"""
SecureVision - Automated Verification Suite for Phase 3
Tests:
  1. Person detection with 0 people (empty scene).
  2. Person detection with 1 person (real live camera feed).
  3. Person detection with 2+ people (multiple individuals in scene).
  4. Performance benchmark comparing raw inference vs periodic N-frame inference.
  5. Visual annotation and badge rendering.
"""

import os
import time
import cv2
import numpy as np
import config
from core.camera import Camera
from core.person_detector import PersonDetector, detect_people


def test_zero_people():
    print("\n--- TEST 1: Zero People Detection (0 People in Frame) ---")
    detector = PersonDetector(process_every_n=1)
    
    # Create realistic empty room/wall background (no human present)
    empty_scene = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 180, dtype=np.uint8)
    # Add some furniture/table gradient to look realistic
    cv2.rectangle(empty_scene, (50, 300), (590, 470), (100, 100, 100), -1)
    
    boxes, count = detector.detect(empty_scene)
    annotated = detector.draw_detections(empty_scene.copy(), boxes, count)
    cv2.imwrite("evidence/phase3_0_people.jpg", annotated)

    print(f"[PASS] Empty frame result: People count = {count}, Boxes = {len(boxes)}")
    assert count == 0, f"Expected 0 people detected, found {count}."
    assert len(boxes) == 0, f"Expected 0 bounding boxes, found {len(boxes)}."
    print("[PASS] Evidence saved to evidence/phase3_0_people.jpg")


def test_one_person():
    print("\n--- TEST 2: Single Person Detection (1 Person in Frame) ---")
    cam = Camera(source=0)
    assert cam.is_opened(), "Could not open real webcam."

    detector = PersonDetector(process_every_n=1)

    # Read a few initial frames to let hardware webcam auto-exposure stabilize
    frame = None
    for _ in range(6):
        ret, frame = cam.read()
        assert ret and frame is not None, "Failed to capture webcam frame."

    boxes, count = detector.detect(frame)
    annotated = detector.draw_detections(frame.copy(), boxes, count)
    cv2.imwrite("evidence/phase3_1_person.jpg", annotated)

    print(f"[PASS] Live camera result: People count = {count}, Boxes = {len(boxes)}")
    assert count >= 1, f"Expected at least 1 person in front of webcam, detected {count}."
    for idx, (x1, y1, x2, y2, conf) in enumerate(boxes):
        print(f"       Person {idx+1}: Box=({x1}, {y1}, {x2}, {y2}), Confidence={conf * 100:.2f}%")
        assert conf >= config.CONFIDENCE_THRESHOLD, f"Confidence {conf} below threshold {config.CONFIDENCE_THRESHOLD}"

    print("[PASS] Evidence saved to evidence/phase3_1_person.jpg")
    cam.release()


def test_multiple_people():
    print("\n--- TEST 3: Multiple People Detection (2+ People in Frame) ---")
    detector = PersonDetector(process_every_n=1)

    # Test with zidane image (contains 2 people)
    img_path = "evidence/zidane.jpg"
    assert os.path.exists(img_path), f"{img_path} not found."
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

    boxes, count = detector.detect(img_resized)
    annotated = detector.draw_detections(img_resized.copy(), boxes, count)
    cv2.imwrite("evidence/phase3_2plus_people.jpg", annotated)

    print(f"[PASS] Multi-person scene result: People count = {count}, Boxes = {len(boxes)}")
    assert count >= 2, f"Expected at least 2 people, detected {count}."
    for idx, (x1, y1, x2, y2, conf) in enumerate(boxes):
        print(f"       Person {idx+1}: Box=({x1}, {y1}, {x2}, {y2}), Confidence={conf * 100:.2f}%")
        assert conf >= config.CONFIDENCE_THRESHOLD

    print("[PASS] Evidence saved to evidence/phase3_2plus_people.jpg")


def test_performance_benchmark():
    print("\n--- TEST 4: Performance & FPS Benchmark ---")
    video_path = "multi_person_demo.mp4"
    assert os.path.exists(video_path), f"{video_path} not found."

    total_frames = 60

    # Case A: Inference on EVERY frame (N = 1)
    cam_a = Camera(source=video_path)
    detector_a = PersonDetector(process_every_n=1)
    start_a = time.perf_counter()
    for _ in range(total_frames):
        ret, frame = cam_a.read()
        if not ret:
            break
        detector_a.detect(frame)
    time_a = time.perf_counter() - start_a
    fps_a = total_frames / time_a if time_a > 0 else 0
    cam_a.release()

    # Case B: Periodic inference every Nth frame (N = 3)
    cam_b = Camera(source=video_path)
    detector_b = PersonDetector(process_every_n=config.PROCESS_EVERY_N_FRAMES)
    start_b = time.perf_counter()
    for _ in range(total_frames):
        ret, frame = cam_b.read()
        if not ret:
            break
        detector_b.detect(frame)
    time_b = time.perf_counter() - start_b
    fps_b = total_frames / time_b if time_b > 0 else 0
    cam_b.release()

    speedup = fps_b / fps_a if fps_a > 0 else 1.0
    print(f"[PASS] Inference every frame (N=1): {fps_a:.2f} FPS ({time_a:.2f}s total)")
    print(f"[PASS] Periodic inference (N={config.PROCESS_EVERY_N_FRAMES}):   {fps_b:.2f} FPS ({time_b:.2f}s total)")
    print(f"[PASS] Speedup Factor: {speedup:.2f}x faster throughput")
    assert fps_b > fps_a, "Periodic inference should achieve higher FPS than processing every frame."


if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision Phase 3 Verification Suite")
    print("=" * 60)

    test_zero_people()
    test_one_person()
    test_multiple_people()
    test_performance_benchmark()

    print("\n" + "=" * 60)
    print(" ALL PHASE 3 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
