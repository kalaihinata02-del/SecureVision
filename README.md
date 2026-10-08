# SecureVision - Smart AI Security Monitoring System

An AI-powered, autonomous smart security monitoring system developed with Python, OpenCV, Flask, SQLite, Ultralytics YOLOv8n, Twilio WhatsApp API, and SMTP Email alerting.

---

## 1. Overview

SecureVision is a production-grade Security Operations Center (SOC) surveillance platform designed for real-time threat detection, automated perimeter defense, and rapid emergency notification.

The system combines computer vision (MOG2 background subtraction, morphological processing, and geometric point-in-polygon analysis) with deep learning (YOLOv8n object detection) to detect moving targets, track human occupants, prevent restricted zone breaches, and defend against optical tampering attacks.

---

## 2. Key Features

- **Real-Time Video Ingestion**: Universal video capture abstraction supporting physical hardware webcams (index `0`) and pre-recorded video files (`"demo.mp4"`) with automatic loopback on EOF.
- **Noise-Filtered Motion Detection**: OpenCV BackgroundSubtractorMOG2 with warmup stabilization (30 frames), Gaussian filtering, and morphological cleanup.
- **YOLOv8n Person Classification**: Real-time COCO class 0 human detection with confidence scoring and periodic inference caching ($N=3$) achieving $\approx 45\text{--}53\text{ FPS}$ on CPU.
- **Restricted Zone Perimeter Defense**: Arbitrary polygon zone definition, mouse-driven coordinate configuration, and bottom-center foot anchor testing (`cv2.pointPolygonTest`).
- **Camera Tampering Detection**: Real-time protection against 3 physical attack vectors:
  - **Covered Lens**: Low mean brightness ($\mu < 30.0$) or near-zero variance ($\sigma^2 < 20.0$).
  - **Defocus / Blur**: Defocused lens via Laplacian variance ($\text{Var} < 45.0$).
  - **Camera Redirection / Movement**: Scene shift via 2D HSV histogram correlation ($d < 0.50$).
  - **Temporal Persistence Guard**: Requires continuous tampering for $\ge 2.0\text{s}$ before triggering.
- **Forensic Evidence Logging**: Automatic high-resolution JPEG evidence snapshot generation and SQLite audit database logging.
- **Web Operations Dashboard**: Modern dark-theme SOC web interface with real-time MJPEG live streaming, live status cards, dynamic toast notifications, Chart.js 24-hour analytics, and live threshold tuning.
- **Dual-Channel Alert Notifications**:
  - **WhatsApp (Twilio Sandbox)**: Instant mobile text alert detailing event type, timestamp, people count, and evidence filename.
  - **Email (SMTP with TLS)**: High-resolution email alert with forensic evidence JPEG snapshot attached.
  - **Decoupled Architecture**: Dispatched in background daemon threads ($< 2\text{ ms}$ latency), ensuring zero camera stutter or frame drops.

---

## 3. Project Structure

```
securevision/
├── app.py                 # Flask web dashboard application & REST APIs
├── main.py                # Standalone OpenCV HUD monitoring window
├── config.py              # Central system parameters, paths & thresholds
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template for alert credentials
├── .env                   # Secret credentials file (git-ignored)
├── demo.mp4               # Pre-recorded test video for offline demonstration
├── zones.json             # Defined restricted area polygon coordinates
├── setup_zone.py          # Mouse-driven polygon zone configuration tool
├── core/
│   ├── camera.py          # Resilient video capture abstraction
│   ├── motion.py          # MOG2 background subtractor motion engine
│   ├── person_detector.py # YOLOv8n deep inference engine
│   ├── zone.py            # Geometric polygon intrusion engine
│   ├── tamper.py          # Optical tampering & redirection engine
│   ├── streamer.py        # Thread-safe capture pipeline & MJPEG streamer
│   └── alerts.py          # Twilio WhatsApp & SMTP Email dispatchers
├── database/
│   ├── db.py              # SQLite schema, queries & evidence persistence
│   └── securevision.db    # Relational forensic audit log
├── evidence/              # Saved high-resolution JPEG evidence snapshots
├── tests/
│   └── TEST_CASES.md      # Comprehensive test checklist & verification matrix
├── docs/
│   ├── ARCHITECTURE.md    # System architecture, data flow & Mermaid diagrams
│   └── VIVA_NOTES.md      # Viva preparation guide, trade-offs & defense Q&A
├── templates/             # Jinja2 dashboard templates (layout, index, login, alerts)
└── static/
    ├── css/style.css      # Dark-theme SOC interface styling
    └── js/dashboard.js     # Client polling, toasts, Chart.js & settings management
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 or 3.11 installed.
- Physical webcam or sample video file.

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy the template and edit your `.env` file:
```powershell
copy .env.example .env
```

---

## 5. Alert Configuration (WhatsApp & Email)

SecureVision sends automated alerts **strictly for Zone Breaches and Camera Tampering** (respecting cooldown intervals) across two independent channels:

### 1. Twilio WhatsApp Sandbox (Text Alerts Only)
1. Create a free developer account at [Twilio](https://console.twilio.com).
2. Go to **Messaging** $\rightarrow$ **Try WhatsApp** to access the WhatsApp Sandbox page.
3. Open WhatsApp on your phone and send the join message (e.g., `join <sandbox-code>`) to the Twilio number (`+1 415 523 8886`).
4. > [!IMPORTANT]
   > **Sandbox Session Expiration**: The Twilio WhatsApp Sandbox session expires every **72 hours**. If alerts stop delivering during testing, simply resend the `join <code>` message from your phone to re-activate the sandbox session.
5. In your `.env` file, populate:
   ```ini
   ENABLE_WHATSAPP=true
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
   WHATSAPP_TO=whatsapp:+<your-mobile-number-with-country-code>
   ```

### 2. Gmail SMTP (Email Alerts with Evidence Image)
1. Navigate to your [Google Account Security Settings](https://myaccount.google.com/security).
2. Ensure **2-Step Verification** is turned ON.
3. Search for **App Passwords** in the search bar.
4. Create an App Password for `SecureVision`. Google will generate a 16-character passcode (e.g., `abcd efgh ijkl mnop`).
5. In your `.env` file, populate:
   ```ini
   ENABLE_EMAIL=true
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your_email@gmail.com
   SMTP_PASSWORD=your_16_character_app_password
   EMAIL_TO=recipient_security_officer@gmail.com
   ```

---

## 6. Running the System

### Option A: Web Operations Center (Recommended)
Launch the Flask web portal with real-time video streaming, live telemetry, and alerting:
```powershell
python app.py
```
- Open your browser at `http://127.0.0.1:5000`
- Log in with default credentials:
  - **Username**: `admin`
  - **Password**: `password123`
- Features:
  - **Live Video Stream**: Annotated MJPEG feed with bounding boxes, polygons, and tamper warnings.
  - **📲 Send Test Alert**: Tests both WhatsApp and Email channels instantly using the latest saved evidence snapshot.
  - **Reset Tamper Baseline**: Recalibrates the tamper detector to the current camera scene.
  - **⚙️ Engine Settings**: Tune motion contour area, YOLO confidence, and alert cooldown dynamically.
  - **Alert Log**: Inspect historical forensic captures and click thumbnails to inspect in the modal lightbox.

### Option B: Standalone Desktop OpenCV Window
Run the desktop OpenCV monitoring window directly:
```powershell
python main.py
```
- Press **'t'** to reset the camera tamper baseline reference frame.
- Press **'q'** to exit cleanly.

---

## 7. Demonstrating with a Video File

To test without a physical webcam, SecureVision natively supports video files:
1. Ensure a video file (e.g., `demo.mp4`) exists in the project root.
2. Open [`config.py`](file:///c:/Users/Sanju/OneDrive/Desktop/secure%20vision/config.py) and change:
   ```python
   CAMERA_SOURCE = "demo.mp4"
   ```
3. Run `python app.py` or `python main.py`. The video will loop automatically when it reaches the end.

---

## 8. Interactive Restricted Zone Setup

Define or adjust the polygon restricted area with the mouse-driven setup tool:
```powershell
python setup_zone.py
```
- **Left Click**: Click 3 or more points on the frame to draw vertices.
- **Enter**: Save the polygon coordinates to `zones.json` and exit.
- **'r'**: Reset / clear points to redraw.
- **'q' / Esc**: Exit without saving changes.

---

## 9. Automated Verification Test Suites

Execute verification scripts across all system phases:
```powershell
python test_phase1.py   # Webcam capture, video looping, FPS HUD overlay
python test_phase2.py   # MOG2 motion detection, warmup, contour filtering
python test_phase3.py   # YOLOv8n person detection benchmark & periodic caching
python test_phase4.py   # Restricted zone polygon intrusion logic & cooldown
python test_phase5.py   # Forensic evidence saving & SQLite audit logging
python test_phase6.py   # Camera tampering detection (covered, blur, moved)
python test_phase7.py   # Flask web portal, MJPEG stream, and REST endpoints
python test_phase8.py   # WhatsApp and Email alert notification engine
```

---

## 10. Troubleshooting

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Camera not opening (`[Error] Cannot open source`)** | Incorrect index or webcam in use by another app | Change `CAMERA_SOURCE = 1` or close applications using the webcam (Teams, Zoom). Alternatively test with `CAMERA_SOURCE = "demo.mp4"`. |
| **Port 5000 in use** | Another process is bound to port 5000 | In `config.py`, change `FLASK_PORT = 5050` or terminate the conflicting process. |
| **WhatsApp message not received** | Twilio 72-hour sandbox session expired | Resend the `join <code>` message from your WhatsApp mobile phone to `+1 415 523 8886`. Check that `ENABLE_WHATSAPP=true` in `.env`. |
| **Email authentication failed** | Used regular password instead of App Password | Generate a 16-character **Google App Password** (Security $\rightarrow$ 2-Step Verification $\rightarrow$ App Passwords) and place it in `SMTP_PASSWORD`. Regular Gmail passwords are rejected. |
| **Tamper false alarms on scene changes** | Reference baseline frame was calibrated to old scene | Click **Reset Tamper Baseline** on the dashboard (or press **'t'** in `main.py`) to recalibrate to the new view. |
| **Low FPS on older hardware** | Per-frame deep learning overhead | In `config.py`, increase `PROCESS_EVERY_N_FRAMES` to `4` or `5`. SecureVision will reuse cached bounding boxes between frames. |
