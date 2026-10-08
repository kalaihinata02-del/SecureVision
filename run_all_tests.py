"""
SecureVision - Unified Master Test Runner
Executes Phase 1 through Phase 8 verification test suites in sequence.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TEST_SCRIPTS = [
    ("Phase 1: Camera Feed & Video Looping", "test_phase1.py"),
    ("Phase 2: MOG2 Motion Detection", "test_phase2.py"),
    ("Phase 3: YOLOv8n Person Detection", "test_phase3.py"),
    ("Phase 4: Restricted Area Zone Intrusion", "test_phase4.py"),
    ("Phase 5: Evidence Snapshot & SQLite Logging", "test_phase5.py"),
    ("Phase 6: Camera Tampering Detection", "test_phase6.py"),
    ("Phase 7: Flask Dashboard & REST APIs", "test_phase7.py"),
    ("Phase 8: Twilio WhatsApp & SMTP Email Alerts", "test_phase8.py"),
]


def run_master_suite():
    print("=" * 70)
    print(" SECUREVISION MASTER TEST RUNNER - FULL REGRESSION SUITE")
    print("=" * 70)

    results = []

    for phase_name, script_name in TEST_SCRIPTS:
        script_path = BASE_DIR / script_name
        if not script_path.exists():
            print(f"[SKIP] {phase_name}: {script_name} not found.")
            results.append((phase_name, False, "File missing"))
            continue

        print(f"\n>>> Running: {phase_name} ({script_name}) ...")
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

        success = proc.returncode == 0
        status_tag = "[PASSED]" if success else "[FAILED]"
        print(f"{status_tag} {phase_name} (Exit code: {proc.returncode})")

        if not success:
            print("--- STDOUT ---")
            print(proc.stdout[-500:] if proc.stdout else "")
            print("--- STDERR ---")
            print(proc.stderr[-500:] if proc.stderr else "")

        results.append((phase_name, success, "Exit code 0" if success else proc.stderr[:100]))

    print("\n" + "=" * 70)
    print(" FINAL MASTER TEST RESULTS SUMMARY")
    print("=" * 70)

    total_passed = sum(1 for _, ok, _ in results if ok)
    total_tests = len(results)

    for phase_name, ok, note in results:
        mark = "[PASS]" if ok else "[FAIL]"
        print(f" {mark:<8} | {phase_name}")

    print("-" * 70)
    print(f" TOTAL SCORE: {total_passed} / {total_tests} PHASES PASSED ({total_passed/total_tests*100:.1f}%)")
    print("=" * 70)

    if total_passed < total_tests:
        sys.exit(1)


if __name__ == "__main__":
    run_master_suite()
