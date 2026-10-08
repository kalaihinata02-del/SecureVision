"""
SecureVision - Configuration Settings
Contains system-wide parameters, thresholds, and camera configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Camera & Video Feed Settings
# ==============================================================================
# CAMERA_SOURCE accepts:
#   - integer index (e.g., 0, 1) for physical webcams
#   - string path (e.g., "demo.mp4" or "videos/sample.mp4") for recorded video files
CAMERA_SOURCE = 0

# Target frame resolution for processing and display
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Target frame rate for video playback pacing (approximate)
TARGET_FPS = 30

# ==============================================================================
# Motion Detection Settings (Phase 2)
# ==============================================================================
# Minimum contour area (in pixels) to qualify as significant motion (filters noise)
MIN_CONTOUR_AREA = 1000

# Number of initial frames to allow background subtractor model to stabilize
WARMUP_FRAMES = 30

# Gaussian blur kernel size to eliminate sensor noise before background subtraction
GAUSSIAN_BLUR_KERNEL = (21, 21)

# ==============================================================================
# Person Detection Settings (Phase 3)
# ==============================================================================
# Path or name of YOLOv8 model weights
YOLO_MODEL_PATH = "yolov8n.pt"

# Minimum confidence score for person (class 0) detections
CONFIDENCE_THRESHOLD = 0.50

# Inference performance: run YOLO inference every Nth frame, reusing detections in between
PROCESS_EVERY_N_FRAMES = 3

# ==============================================================================
# Restricted Area Zone Settings (Phase 4)
# ==============================================================================
# File where defined polygon points are saved and loaded
ZONES_FILE = BASE_DIR / "zones.json"

# Alert cooldown in seconds to prevent flooding duplicate breach/person events
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "60.0").strip())

# Default polygon coordinates (fallback if zones.json has not been created yet)
# Covers a realistic lower central area of a 640x480 frame
DEFAULT_ZONE_POINTS = [
    [120, 220],
    [520, 220],
    [580, 460],
    [60, 460],
]

# ==============================================================================
# Camera Tampering Settings (Phase 6)
# ==============================================================================
# Threshold for covered/blocked lens (mean grayscale brightness < threshold)
TAMPER_DARK_MEAN_THRESHOLD = 30.0

# Threshold for low variance / uniform blocked image
TAMPER_LOW_VARIANCE_THRESHOLD = 20.0

# Threshold for blur/defocus (variance of Laplacian < threshold)
TAMPER_BLUR_LAPLACIAN_THRESHOLD = 45.0

# Threshold for camera redirection/movement (scene histogram similarity < threshold)
TAMPER_SCENE_SIMILARITY_THRESHOLD = 0.50

# Continuous persistence required before confirming tampering (prevents transient false alarms)
TAMPER_CONFIRM_SECONDS = 2.0

# Cooldown between consecutive tamper alert logs
TAMPER_COOLDOWN_SECONDS = 8.0

# ==============================================================================
# Storage Paths
# ==============================================================================
EVIDENCE_DIR = BASE_DIR / "evidence"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "securevision.db"
# Ensure runtime directories exist
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Flask Web Dashboard & Admin Auth (Phase 7)
# ==============================================================================
SECRET_KEY = "securevision-secret-key-college-project-2026"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

# ==============================================================================
# Alert Notification Settings (Phase 8: Twilio WhatsApp & SMTP Email)
# ==============================================================================
# Independent channel toggles
ENABLE_WHATSAPP = os.getenv("ENABLE_WHATSAPP", "false").strip().lower() in ("true", "1", "yes")
ENABLE_EMAIL = os.getenv("ENABLE_EMAIL", "false").strip().lower() in ("true", "1", "yes")

# Twilio WhatsApp API credentials (Text only via Twilio Sandbox)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
WHATSAPP_TO = os.getenv("WHATSAPP_TO", "").strip()

# SMTP Email credentials (TLS with attached evidence image)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587").strip() or 587)
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()

# ==============================================================================
# Automatic Person Alert Trigger Settings (Upgrade)
# ==============================================================================
AUTO_ALERT_ENABLED = os.getenv("AUTO_ALERT_ENABLED", "true").strip().lower() in ("true", "1", "yes")
ALERT_ON_PERSON = os.getenv("ALERT_ON_PERSON", "true").strip().lower() in ("true", "1", "yes")
PERSON_CONFIRM_SECONDS = float(os.getenv("PERSON_CONFIRM_SECONDS", "2.0").strip())
PERSON_ABSENT_SECONDS = float(os.getenv("PERSON_ABSENT_SECONDS", "10.0").strip())
MIN_PERSON_CONFIDENCE = float(os.getenv("MIN_PERSON_CONFIDENCE", "0.60").strip())


def reload_env():
    """Reload environment variables from .env to reflect runtime changes."""
    global ENABLE_WHATSAPP, ENABLE_EMAIL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
    global TWILIO_WHATSAPP_FROM, WHATSAPP_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO
    global AUTO_ALERT_ENABLED, ALERT_ON_PERSON, PERSON_CONFIRM_SECONDS, PERSON_ABSENT_SECONDS
    global ALERT_COOLDOWN_SECONDS, MIN_PERSON_CONFIDENCE

    load_dotenv(BASE_DIR / ".env", override=True)
    ENABLE_WHATSAPP = os.getenv("ENABLE_WHATSAPP", "false").strip().lower() in ("true", "1", "yes")
    ENABLE_EMAIL = os.getenv("ENABLE_EMAIL", "false").strip().lower() in ("true", "1", "yes")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
    WHATSAPP_TO = os.getenv("WHATSAPP_TO", "").strip()
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587").strip() or 587)
    SMTP_USER = os.getenv("SMTP_USER", "").strip()
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
    EMAIL_TO = os.getenv("EMAIL_TO", "").strip()

    AUTO_ALERT_ENABLED = os.getenv("AUTO_ALERT_ENABLED", "true").strip().lower() in ("true", "1", "yes")
    ALERT_ON_PERSON = os.getenv("ALERT_ON_PERSON", "true").strip().lower() in ("true", "1", "yes")
    PERSON_CONFIRM_SECONDS = float(os.getenv("PERSON_CONFIRM_SECONDS", "2.0").strip())
    PERSON_ABSENT_SECONDS = float(os.getenv("PERSON_ABSENT_SECONDS", "10.0").strip())
    ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "60.0").strip())
    MIN_PERSON_CONFIDENCE = float(os.getenv("MIN_PERSON_CONFIDENCE", "0.60").strip())
