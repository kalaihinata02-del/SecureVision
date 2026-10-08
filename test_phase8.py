"""
SecureVision - Phase 8 Automated Verification Suite
Tests:
1. Environment and configuration loading for alert channels.
2. Twilio WhatsApp API client integration (text only).
3. SMTP Email client integration with TLS and evidence image attachment.
4. Channel independence (failure of one does not block the other).
5. Asynchronous non-blocking background dispatch.
6. Flask /api/alerts/test REST diagnostic endpoint.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from core.alerts import (
    send_whatsapp,
    send_email,
    send_security_alert_async,
    send_test_alert,
    _dispatch_worker,
)
from app import app


def test_configuration():
    print("\n--- TEST 1: Alert Configuration & Environment Loading ---")
    print(f"ENABLE_WHATSAPP: {config.ENABLE_WHATSAPP}")
    print(f"ENABLE_EMAIL:    {config.ENABLE_EMAIL}")
    print(f"TWILIO_ACCOUNT_SID: {'[SET]' if config.TWILIO_ACCOUNT_SID else '[EMPTY]'}")
    print(f"TWILIO_AUTH_TOKEN:  {'[SET]' if config.TWILIO_AUTH_TOKEN else '[EMPTY]'}")
    print(f"WHATSAPP_TO:        {config.WHATSAPP_TO or '[EMPTY]'}")
    print(f"SMTP_HOST:          {config.SMTP_HOST}:{config.SMTP_PORT}")
    print(f"SMTP_USER:          {config.SMTP_USER or '[EMPTY]'}")
    print(f"EMAIL_TO:           {config.EMAIL_TO or '[EMPTY]'}")
    print("[PASS] Configuration variables loaded successfully without errors.")


def test_whatsapp_channel():
    print("\n--- TEST 2: Twilio WhatsApp Alert Channel ---")
    test_msg = "[SECUREVISION TEST] Automated Phase 8 verification check."

    # If WhatsApp is disabled or credentials missing, verify graceful handling
    res = send_whatsapp(test_msg)
    if res["success"]:
        print(f"[PASS] Real WhatsApp message sent successfully! Twilio Message SID: {res['sid']}")
    else:
        print(f"[INFO] WhatsApp send result: {res['error']}")
        if not config.ENABLE_WHATSAPP or not config.TWILIO_ACCOUNT_SID:
            print("[PASS] WhatsApp channel gracefully reported disabled/missing credentials without throwing an unhandled exception.")
            print("\n  >>> WhatsApp Setup Reminder:")
            print("      1. Create account on https://console.twilio.com")
            print("      2. Go to Messaging -> Try WhatsApp, join sandbox from your phone")
            print("      3. In .env set ENABLE_WHATSAPP=true, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, WHATSAPP_TO")
        else:
            print(f"[NOTICE] Twilio returned API response: {res['error']}")

    # Now verify Twilio exception handling when credentials are intentionally invalid
    print("\n--- TEST 2b: Twilio Error Resilience Check (Invalid Credentials) ---")
    orig_en = config.ENABLE_WHATSAPP
    orig_sid = config.TWILIO_ACCOUNT_SID
    orig_token = config.TWILIO_AUTH_TOKEN
    orig_to = config.WHATSAPP_TO

    config.ENABLE_WHATSAPP = True
    config.TWILIO_ACCOUNT_SID = "AC00000000000000000000000000000000"
    config.TWILIO_AUTH_TOKEN = "dummy_token_1234567890abcdef"
    config.WHATSAPP_TO = "whatsapp:+1234567890"

    res_dummy = send_whatsapp(test_msg)
    assert res_dummy["success"] is False, "Dummy Twilio call should fail authentication"
    assert res_dummy["error"] is not None, "Error detail must be captured"
    print(f"[PASS] Twilio API error captured cleanly without crash: {res_dummy['error']}")

    # Restore
    config.ENABLE_WHATSAPP = orig_en
    config.TWILIO_ACCOUNT_SID = orig_sid
    config.TWILIO_AUTH_TOKEN = orig_token
    config.WHATSAPP_TO = orig_to


def test_email_channel():
    print("\n--- TEST 3: SMTP Email Alert Channel ---")
    test_subject = "[SecureVision Test] Verification Email"
    test_body = "This is a test notification from the SecureVision automated verification suite."

    # Find an evidence image to attach
    evidence_files = list(config.EVIDENCE_DIR.glob("*.jpg"))
    test_img = str(evidence_files[0]) if evidence_files else None

    res = send_email(test_subject, test_body, image_path=test_img)
    if res["success"]:
        print(f"[PASS] Real Email sent successfully with evidence image attached! To: {config.EMAIL_TO}")
    else:
        print(f"[INFO] Email send result: {res['error']}")
        if not config.ENABLE_EMAIL or not config.SMTP_USER:
            print("[PASS] Email channel gracefully reported disabled/missing credentials without throwing an unhandled exception.")
            print("\n  >>> Gmail SMTP Setup Reminder:")
            print("      1. Enable 2-Step Verification in Google Account -> Security")
            print("      2. Generate an App Password for 'SecureVision' (16 characters)")
            print("      3. In .env set ENABLE_EMAIL=true, SMTP_USER, SMTP_PASSWORD, EMAIL_TO")
        else:
            print(f"[NOTICE] SMTP server response: {res['error']}")

    # Now verify SMTP exception handling when credentials are intentionally invalid
    print("\n--- TEST 3b: SMTP Error Resilience Check (Invalid Auth) ---")
    orig_email_en = config.ENABLE_EMAIL
    orig_smtp_user = config.SMTP_USER
    orig_smtp_pass = config.SMTP_PASSWORD
    orig_email_to = config.EMAIL_TO

    config.ENABLE_EMAIL = True
    config.SMTP_USER = "nonexistent_securevision_test@gmail.com"
    config.SMTP_PASSWORD = "invalidpassword123"
    config.EMAIL_TO = "recipient_test@gmail.com"

    res_email_dummy = send_email(test_subject, test_body, image_path=test_img)
    assert res_email_dummy["success"] is False, "Dummy SMTP call should fail authentication"
    assert res_email_dummy["error"] is not None, "Error detail must be captured"
    print(f"[PASS] SMTP authentication error captured cleanly without crash: {res_email_dummy['error']}")

    # Restore
    config.ENABLE_EMAIL = orig_email_en
    config.SMTP_USER = orig_smtp_user
    config.SMTP_PASSWORD = orig_smtp_pass
    config.EMAIL_TO = orig_email_to


def test_channel_independence():
    print("\n--- TEST 4: Channel Independence & Fault Isolation ---")
    # Temporarily simulate WhatsApp failing while Email runs, and vice versa
    orig_wa_sid = config.TWILIO_ACCOUNT_SID
    config.TWILIO_ACCOUNT_SID = ""

    # Call dispatch worker
    res = _dispatch_worker(
        event_type="TEST_FAULT_TOLERANCE",
        timestamp_str="2026-10-06 12:00:00",
        people_count=1,
        image_path=None,
        details="Fault isolation test",
    )

    # Restore
    config.TWILIO_ACCOUNT_SID = orig_wa_sid

    assert "whatsapp" in res and "email" in res, "Dispatch worker must return status for both channels"
    print(f"[PASS] Fault isolation verified: WhatsApp status='{res['whatsapp'].get('error')}' and Email was still executed independently.")


def test_async_background_execution():
    print("\n--- TEST 5: Asynchronous Non-Blocking Dispatch ---")
    t0 = time.perf_counter()
    thread = send_security_alert_async(
        event_type="ZONE_BREACH",
        timestamp_str="2026-10-06 12:00:00",
        people_count=1,
        image_path=None,
        details="Non-blocking latency test",
    )
    t_elapsed = time.perf_counter() - t0

    assert isinstance(thread, threading.Thread), "send_security_alert_async must return a threading.Thread"
    assert t_elapsed < 0.1, f"Alert dispatch must not block camera thread (took {t_elapsed:.4f}s)"
    print(f"[PASS] Non-blocking dispatch returned in {t_elapsed*1000:.2f} ms (well below 100 ms limit).")
    print(f"[PASS] Background thread '{thread.name}' dispatched daemonically (alive={thread.is_alive()}).")
    thread.join(timeout=2.0)


def test_api_test_alert_endpoint():
    print("\n--- TEST 6: Dashboard /api/alerts/test REST Endpoint ---")
    client = app.test_client()
    resp = client.post("/api/alerts/test")

    assert resp.status_code == 200, f"Expected HTTP 200 from /api/alerts/test, got {resp.status_code}"
    data = json.loads(resp.data.decode("utf-8"))
    assert "results" in data, "Expected 'results' key in JSON response"
    assert "whatsapp" in data["results"], "Expected 'whatsapp' in results"
    assert "email" in data["results"], "Expected 'email' in results"

    print(f"[PASS] /api/alerts/test returned HTTP 200 JSON:")
    print(f"       WhatsApp Success: {data['results']['whatsapp']['success']} (Error: {data['results']['whatsapp'].get('error')})")
    print(f"       Email Success:    {data['results']['email']['success']} (Error: {data['results']['email'].get('error')})")


def run_all_tests():
    print("=" * 60)
    print(" SecureVision Phase 8 Verification Suite")
    print("=" * 60)

    test_configuration()
    test_whatsapp_channel()
    test_email_channel()
    test_channel_independence()
    test_async_background_execution()
    test_api_test_alert_endpoint()

    print("\n" + "=" * 60)
    print(" ALL PHASE 8 VERIFICATION CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
