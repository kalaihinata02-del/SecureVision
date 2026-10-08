"""
test_auto_alert.py - Automated unit tests for AutoAlertManager state machine.

Run:
    .\venv\Scripts\python.exe test_auto_alert.py
"""

import sys
import types
import time
import threading
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# Stub only the modules that would trigger heavy I/O at import time.
# We must NOT stub "core" itself (that is a real package on disk).
# We DO stub "core.alerts" to prevent real Twilio / SMTP calls.
# We DO stub "database.db" to prevent real SQLite access.
# We DO stub "cv2" to prevent OpenCV calls.
# ---------------------------------------------------------------------------

def _stub_module(full_name):
    """Insert an empty module into sys.modules under full_name."""
    parts = full_name.split(".")
    # Ensure parent packages exist in sys.modules
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            mod = types.ModuleType(parent)
            sys.modules[parent] = mod
    mod = types.ModuleType(full_name)
    sys.modules[full_name] = mod
    # Attach as attribute on parent if possible
    if len(parts) > 1:
        parent_mod = sys.modules[".".join(parts[:-1])]
        setattr(parent_mod, parts[-1], mod)
    return mod

# Stub cv2
cv2_stub = _stub_module("cv2")
cv2_stub.imwrite = lambda *a, **k: True
cv2_stub.imencode = lambda fmt, img, params=None: (True, b"\xff\xd8")
cv2_stub.IMWRITE_JPEG_QUALITY = 95

# Stub database.db (but keep "database" as an importable package stub too)
_stub_module("database")
db_stub = _stub_module("database.db")
db_stub.get_alert_count_today = lambda: 0
db_stub.log_event = lambda *a, **k: {"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00", "event_type": "PERSON_DETECTED", "people_count": 1, "details": ""}

# Stub core.alerts only — do NOT stub "core" itself
# First make sure "core" is the real package (already on disk)
import importlib
import core  # noqa: F401 — ensure real core package is loaded

alerts_stub = _stub_module("core.alerts")
alerts_stub.send_security_alert_async = MagicMock()

# Now import the real AutoAlertManager
from core.auto_alert import AutoAlertManager, AutoAlertState  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
PERSON_HIGH = (100, 100, 200, 300, 0.92)
PERSON_LOW  = (100, 100, 200, 300, 0.30)


def make_manager(armed=True):
    """Return an AutoAlertManager with a clean, reset state."""
    mgr = AutoAlertManager.__new__(AutoAlertManager)
    mgr.lock = threading.Lock()
    mgr.state = AutoAlertState.IDLE
    mgr.armed = armed
    mgr.confirm_start_time = None
    mgr.absent_start_time = None
    mgr.last_alert_time = None
    mgr.last_alert_time_str = "None"
    mgr.last_alert_type = "None"
    mgr.last_alert_count = 0
    mgr.alerts_sent_today = 0
    return mgr


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestAutoAlertStateMachine(unittest.TestCase):

    def test_01_idle_to_confirming(self):
        """TC-01: IDLE -> CONFIRMING on first valid person."""
        mgr = make_manager()
        with patch("core.auto_alert.send_security_alert_async"), \
             patch("core.auto_alert.log_event", return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}):
            mgr.update(BLANK_FRAME, [PERSON_HIGH])
        self.assertEqual(mgr.state, AutoAlertState.CONFIRMING)

    def test_02_confirming_to_alerted(self):
        """TC-02 (strengthened): CONFIRMING -> ALERTED; state is exactly ALERTED/COOLDOWN and dispatch called exactly once."""
        import config as cfg
        orig_c = cfg.PERSON_CONFIRM_SECONDS
        orig_cl = cfg.ALERT_COOLDOWN_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        cfg.ALERT_COOLDOWN_SECONDS = 999.0   # keep in COOLDOWN so state is deterministic
        try:
            with patch("core.auto_alert.send_security_alert_async") as mock_send, \
                 patch("core.auto_alert.log_event",
                        return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True
                mgr.update(BLANK_FRAME, [PERSON_HIGH])   # -> CONFIRMING
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_HIGH])   # -> ALERTED (dispatch here)
        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig_c
            cfg.ALERT_COOLDOWN_SECONDS = orig_cl
        self.assertIn(mgr.state, (AutoAlertState.ALERTED, AutoAlertState.COOLDOWN),
                      f"Expected ALERTED or COOLDOWN, got {mgr.state}")
        mock_send.assert_called_once()

    def test_03_flicker_rejected(self):
        """TC-03: Person disappears before confirm window -> reset to IDLE, no alert."""
        import config as cfg
        orig = cfg.PERSON_CONFIRM_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 5.0
        try:
            with patch("core.auto_alert.send_security_alert_async") as mock_send, \
                 patch("core.auto_alert.log_event", return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}):
                mgr.update(BLANK_FRAME, [PERSON_HIGH])
                mgr.update(BLANK_FRAME, [])
        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig
        self.assertEqual(mgr.state, AutoAlertState.IDLE)
        mock_send.assert_not_called()

    def test_04_no_duplicate_alert(self):
        """TC-04: Only ONE alert dispatched per continuous presence."""
        import config as cfg
        orig = cfg.PERSON_CONFIRM_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        calls = []
        try:
            with patch("core.auto_alert.send_security_alert_async",
                       side_effect=lambda *a, **k: calls.append(1)), \
                 patch("core.auto_alert.log_event", return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True
                mgr.update(BLANK_FRAME, [PERSON_HIGH])
                time.sleep(0.12)
                for _ in range(5):
                    mgr.update(BLANK_FRAME, [PERSON_HIGH])
        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig
        self.assertEqual(len(calls), 1)

    def test_05_absence_reset(self):
        """TC-05: ALERTED -> IDLE after person absent for absent_seconds."""
        import config as cfg
        orig_c = cfg.PERSON_CONFIRM_SECONDS
        orig_a = cfg.PERSON_ABSENT_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        cfg.PERSON_ABSENT_SECONDS = 0.05
        try:
            with patch("core.auto_alert.send_security_alert_async"), \
                 patch("core.auto_alert.log_event", return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True
                mgr.update(BLANK_FRAME, [PERSON_HIGH])
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_HIGH])  # -> ALERTED
                mgr.update(BLANK_FRAME, [])              # start absent timer
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [])              # -> IDLE
        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig_c
            cfg.PERSON_ABSENT_SECONDS = orig_a
        self.assertEqual(mgr.state, AutoAlertState.IDLE)

    def test_06_low_confidence_ignored(self):
        """TC-06: Detection below MIN_PERSON_CONFIDENCE stays IDLE."""
        import config as cfg
        orig = cfg.MIN_PERSON_CONFIDENCE
        mgr = make_manager()
        cfg.MIN_PERSON_CONFIDENCE = 0.60
        try:
            with patch("core.auto_alert.send_security_alert_async") as mock_send, \
                 patch("core.auto_alert.log_event", return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}):
                mgr.update(BLANK_FRAME, [PERSON_LOW])
        finally:
            cfg.MIN_PERSON_CONFIDENCE = orig
        self.assertEqual(mgr.state, AutoAlertState.IDLE)
        mock_send.assert_not_called()

    def test_07_disarmed_suppresses_alert(self):
        """TC-07: DISARMED system does not dispatch alert even after confirm."""
        import config as cfg
        orig = cfg.PERSON_CONFIRM_SECONDS
        mgr = make_manager(armed=False)
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        try:
            with patch("core.auto_alert.send_security_alert_async") as mock_send, \
                 patch("core.auto_alert.log_event", return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}):
                mgr.update(BLANK_FRAME, [PERSON_HIGH])
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_HIGH])
        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig
        mock_send.assert_not_called()

    def test_08_toggle_armed(self):
        """TC-08: toggle_armed() flips state correctly."""
        mgr = make_manager(armed=True)
        self.assertTrue(mgr.is_armed())
        mgr.toggle_armed()
        self.assertFalse(mgr.is_armed())
        mgr.toggle_armed()
        self.assertTrue(mgr.is_armed())

    def test_09_get_status_shape(self):
        """TC-09: get_status() returns all required keys."""
        mgr = make_manager()
        status = mgr.get_status()
        required = {"armed", "state", "last_alert_time", "last_alert_type",
                    "alerts_sent_today", "tracked_people"}
        missing = required - set(status.keys())
        self.assertFalse(missing, f"get_status() missing keys: {missing}")

    # ------------------------------------------------------------------
    # TC-10: Count escalation (1 -> 2) triggers new alert AFTER cooldown
    # ------------------------------------------------------------------
    def test_10_count_escalation_after_cooldown(self):
        """TC-10: People count 1->2 fires a second alert when ALERT_COOLDOWN_SECONDS has passed."""
        import config as cfg
        orig_c = cfg.PERSON_CONFIRM_SECONDS
        orig_cl = cfg.ALERT_COOLDOWN_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        cfg.ALERT_COOLDOWN_SECONDS = 0.05   # very short for test

        PERSON_BOX_1 = (100, 100, 200, 300, 0.92)
        PERSON_BOX_2A = (100, 100, 200, 300, 0.91)
        PERSON_BOX_2B = (150, 100, 200, 300, 0.90)

        calls = []
        try:
            with patch("core.auto_alert.send_security_alert_async",
                       side_effect=lambda *a, **k: calls.append(k.get("event_type", a))), \
                 patch("core.auto_alert.log_event",
                        return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True

                # Step 1: confirm and alert on 1 person
                mgr.update(BLANK_FRAME, [PERSON_BOX_1])      # IDLE -> CONFIRMING
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_BOX_1])      # -> ALERTED (alert #1)

                # Step 2: wait for cooldown to expire
                time.sleep(0.12)

                # Step 3: 2 people in frame now — should trigger alert #2
                mgr.update(BLANK_FRAME, [PERSON_BOX_2A, PERSON_BOX_2B])

        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig_c
            cfg.ALERT_COOLDOWN_SECONDS = orig_cl

        self.assertEqual(len(calls), 2,
                         f"Expected 2 alerts (initial + count escalation), got {len(calls)}")

    # ------------------------------------------------------------------
    # TC-11: Count escalation within cooldown window does NOT alert
    # ------------------------------------------------------------------
    def test_11_count_escalation_within_cooldown_blocked(self):
        """TC-11: People count 1->2 within ALERT_COOLDOWN_SECONDS does NOT trigger a second alert."""
        import config as cfg
        orig_c = cfg.PERSON_CONFIRM_SECONDS
        orig_cl = cfg.ALERT_COOLDOWN_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        cfg.ALERT_COOLDOWN_SECONDS = 999.0   # cooldown never expires during test

        PERSON_BOX_1 = (100, 100, 200, 300, 0.92)
        PERSON_BOX_2A = (100, 100, 200, 300, 0.91)
        PERSON_BOX_2B = (150, 100, 200, 300, 0.90)

        calls = []
        try:
            with patch("core.auto_alert.send_security_alert_async",
                       side_effect=lambda *a, **k: calls.append(1)), \
                 patch("core.auto_alert.log_event",
                        return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True

                # Alert on 1 person
                mgr.update(BLANK_FRAME, [PERSON_BOX_1])
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_BOX_1])      # alert #1

                # 2 people immediately — cooldown still active
                mgr.update(BLANK_FRAME, [PERSON_BOX_2A, PERSON_BOX_2B])

        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig_c
            cfg.ALERT_COOLDOWN_SECONDS = orig_cl

        self.assertEqual(len(calls), 1,
                         f"Expected 1 alert (cooldown blocks escalation), got {len(calls)}")

    # ------------------------------------------------------------------
    # TC-12: Zone breach at same time as person -> ONE alert, ZONE_BREACH
    # ------------------------------------------------------------------
    def test_12_zone_breach_priority_single_alert(self):
        """TC-12: When zone breach + person detected simultaneously, only ONE ZONE_BREACH alert is sent."""
        import config as cfg
        orig_c = cfg.PERSON_CONFIRM_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05

        captured_types = []

        def fake_dispatch(*args, **kwargs):
            # event_type is passed as first positional arg or keyword
            et = kwargs.get("event_type") or (args[0] if args else "UNKNOWN")
            captured_types.append(et)

        try:
            with patch("core.auto_alert.send_security_alert_async",
                       side_effect=fake_dispatch), \
                 patch("core.auto_alert.log_event",
                        return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True

                mgr.update(BLANK_FRAME, [PERSON_HIGH])        # -> CONFIRMING
                time.sleep(0.12)
                # Second call: person confirmed AND zone is breached simultaneously
                mgr.update(BLANK_FRAME, [PERSON_HIGH], is_breached=True)

        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig_c

        # Exactly ONE alert dispatched
        self.assertEqual(len(captured_types), 1,
                         f"Expected exactly 1 alert, got {len(captured_types)}")
        # It must be ZONE_BREACH (higher priority than PERSON_DETECTED)
        self.assertEqual(captured_types[0], "ZONE_BREACH",
                         f"Expected ZONE_BREACH priority, got {captured_types[0]}")

    # ------------------------------------------------------------------
    # TC-13: Person returns after absence reset -> new alert after cooldown
    # ------------------------------------------------------------------
    def test_13_return_after_absence_triggers_new_alert(self):
        """TC-13: After ALERTED->IDLE absence reset, returning person triggers a new alert (once cooldown is done)."""
        import config as cfg
        orig_c = cfg.PERSON_CONFIRM_SECONDS
        orig_a = cfg.PERSON_ABSENT_SECONDS
        orig_cl = cfg.ALERT_COOLDOWN_SECONDS
        mgr = make_manager()
        cfg.PERSON_CONFIRM_SECONDS = 0.05
        cfg.PERSON_ABSENT_SECONDS = 0.05
        cfg.ALERT_COOLDOWN_SECONDS = 0.05   # short cooldown so second alert is allowed

        calls = []
        try:
            with patch("core.auto_alert.send_security_alert_async",
                       side_effect=lambda *a, **k: calls.append(1)), \
                 patch("core.auto_alert.log_event",
                        return_value={"id": 1, "image_path": None, "timestamp": "2026-01-01 00:00:00"}), \
                 patch("core.auto_alert.cv2") as mock_cv2:
                mock_cv2.imwrite.return_value = True

                # -- First presence: IDLE -> CONFIRMING -> ALERTED --
                mgr.update(BLANK_FRAME, [PERSON_HIGH])
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_HIGH])        # alert #1 dispatched

                # -- Person leaves: ALERTED -> IDLE --
                mgr.update(BLANK_FRAME, [])
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [])                   # -> IDLE

                self.assertEqual(mgr.state, AutoAlertState.IDLE,
                                 "State should be IDLE after absence reset")

                # -- Person returns: should confirm and alert again --
                mgr.update(BLANK_FRAME, [PERSON_HIGH])        # IDLE -> CONFIRMING
                time.sleep(0.12)
                mgr.update(BLANK_FRAME, [PERSON_HIGH])        # -> ALERTED (alert #2)

        finally:
            cfg.PERSON_CONFIRM_SECONDS = orig_c
            cfg.PERSON_ABSENT_SECONDS = orig_a
            cfg.ALERT_COOLDOWN_SECONDS = orig_cl

        self.assertEqual(len(calls), 2,
                         f"Expected 2 alerts (first presence + return after absence), got {len(calls)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("  SecureVision - AutoAlertManager State Machine Unit Tests")
    print("=" * 64)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAutoAlertStateMachine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 64)
    if result.wasSuccessful():
        print("OK  ALL 13 TESTS PASSED")
    else:
        print(f"FAIL  {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 64)
    sys.exit(0 if result.wasSuccessful() else 1)



