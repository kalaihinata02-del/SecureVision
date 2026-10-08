"""
SecureVision - Restricted Zone Setup Tool (Phase 4)
Interactive utility to configure restricted zone polygons.
Click mouse points on the frame to define vertices.
Controls:
  - Left Mouse Click: Add polygon vertex
  - Enter: Save zone to zones.json and exit
  - 'r': Reset / clear current points
  - 'q' or Esc: Cancel and exit
"""

import sys
import cv2
import numpy as np
import config
from core.camera import Camera
from core.zone import ZoneManager


clicked_points = []


def on_mouse(event, x, y, flags, param):
    """Handle mouse click events to record polygon vertices."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Vertex added at ({x}, {y}) [Total: {len(clicked_points)} points]")


def draw_setup_canvas(base_frame, points):
    """Render vertices, connecting edges, and instructions on the canvas."""
    canvas = base_frame.copy()
    overlay = base_frame.copy()
    h, w = canvas.shape[:2]

    # Draw semi-transparent polygon preview if >= 3 points
    if len(points) >= 3:
        pts_arr = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_arr], (0, 30, 200))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        cv2.polylines(canvas, [pts_arr], isClosed=True, color=(0, 60, 255), thickness=2, lineType=cv2.LINE_AA)
    elif len(points) == 2:
        cv2.line(canvas, tuple(points[0]), tuple(points[1]), (0, 60, 255), 2, lineType=cv2.LINE_AA)

    # Draw vertices
    for i, pt in enumerate(points):
        cv2.circle(canvas, tuple(pt), 6, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(pt), 9, (255, 255, 255), 1)
        cv2.putText(
            canvas,
            f"P{i+1}",
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Header instructions HUD
    hud_bg = canvas.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.8, canvas, 0.2, 0, canvas)

    cv2.putText(
        canvas,
        "ZONE SETUP: Click vertices | 'Enter' = Save | 'r' = Reset | 'q' = Cancel",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 230, 115),
        1,
        cv2.LINE_AA,
    )

    # Point counter badge
    pts_text = f"Points: {len(points)}"
    cv2.putText(
        canvas,
        pts_text,
        (w - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def run_zone_setup(source=None):
    """Launch the interactive zone configuration window."""
    global clicked_points
    effective_source = config.CAMERA_SOURCE if source is None else source

    print("=" * 60)
    print(" SecureVision - Interactive Restricted Zone Setup Tool")
    print("=" * 60)
    print(f"Opening camera source: {effective_source}")

    cam = Camera(source=effective_source)
    if not cam.is_opened():
        print(f"[Error] Could not open video source {effective_source}.")
        return False

    # Capture a clear reference frame
    ref_frame = None
    for _ in range(10):  # Allow sensor exposure to settle
        ret, frame = cam.read()
        if ret:
            ref_frame = frame

    cam.release()

    if ref_frame is None:
        print("[Error] Failed to capture reference frame.")
        return False

    # Load existing points if available
    zone_mgr = ZoneManager()
    if zone_mgr.points and len(zone_mgr.points) >= 3:
        clicked_points = [list(pt) for pt in zone_mgr.points]
        print(f"Loaded existing zone with {len(clicked_points)} points.")

    window_name = "SecureVision - Restricted Zone Setup"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nInstructions:")
    print("  - Left click on window to add vertices")
    print("  - Press 'Enter' to SAVE and exit")
    print("  - Press 'r' to RESET/CLEAR points")
    print("  - Press 'q' or Esc to cancel\n")

    saved = False
    try:
        while True:
            display_canvas = draw_setup_canvas(ref_frame, clicked_points)
            cv2.imshow(window_name, display_canvas)

            key = cv2.waitKey(20) & 0xFF

            # Enter key (13 on Windows/Linux, 10 on some platforms)
            if key in (13, 10):
                if len(clicked_points) < 3:
                    print("[Warning] A polygon requires at least 3 points! Please click more points.")
                else:
                    success = zone_mgr.save_zone(clicked_points)
                    if success:
                        print(f"\n[Success] Zone successfully saved to {config.ZONES_FILE} with {len(clicked_points)} vertices!")
                        saved = True
                    break

            # 'r' or 'R' key to reset
            elif key in (ord('r'), ord('R')):
                clicked_points = []
                print("[Info] Points cleared. Click to define a new polygon.")

            # 'q' or Esc key to cancel
            elif key in (ord('q'), ord('Q'), 27):
                print("[Info] Zone setup cancelled without saving.")
                break

    finally:
        cv2.destroyAllWindows()

    return saved


if __name__ == "__main__":
    run_zone_setup()
