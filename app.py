"""
SecureVision - Flask Web Application Server (Phase 7)
Serves real-time SOC web dashboard, MJPEG live video stream, REST APIs,
evidence captures, and SQLite alert event history.
"""

from functools import wraps
import os
from pathlib import Path
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
import config
from core.streamer import get_streamer
from database.db import get_alert_count_today, get_alerts_by_hour, get_recent_alerts, init_db


app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Ensure SQLite database is initialized
init_db()


def login_required(f):
    """Decorator to require session-based administrator login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# ==============================================================================
# Authentication Routes
# ==============================================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    """Administrator login page and authentication handler."""
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session["logged_in"] = True
            session["user"] = username
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))
        else:
            error = "Invalid administrator credentials. Please check your username and password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Clear user session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))


# ==============================================================================
# Main Pages
# ==============================================================================

@app.route("/")
@login_required
def dashboard():
    """Main SOC Dashboard page."""
    return render_template("index.html")


@app.route("/alerts")
@login_required
def alerts_page():
    """Dedicated alert history log page with thumbnails and filters."""
    alerts = get_recent_alerts(limit=100)
    return render_template("alerts.html", alerts=alerts)


# ==============================================================================
# Video Streaming & Evidence Files
# ==============================================================================

@app.route("/video_feed")
def video_feed():
    """
    MJPEG video streaming route. Serves live annotated frames generated
    by the background thread pipeline.
    """
    streamer = get_streamer()
    return Response(
        streamer.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    """Serve saved forensic evidence snapshots from the evidence directory."""
    return send_from_directory(config.EVIDENCE_DIR, filename)


# ==============================================================================
# REST API Endpoints
# ==============================================================================

@app.route("/api/status")
def api_status():
    """
    Returns live computed diagnostics:
    camera_online, people_count, motion, zone_breach, tamper_status, alerts_today.
    """
    streamer = get_streamer()
    status = streamer.get_status()
    return jsonify(status)


@app.route("/api/alerts")
def api_alerts():
    """Returns recent alerts from the SQLite database."""
    limit = request.args.get("limit", default=50, type=int)
    alerts = get_recent_alerts(limit=limit)
    return jsonify(alerts)


@app.route("/api/alerts/hourly")
def api_alerts_hourly():
    """Returns hourly breakdown of alerts today for Chart.js."""
    date_str = request.args.get("date", default=None, type=str)
    hourly = get_alerts_by_hour(date_str=date_str)
    return jsonify(hourly)


@app.route("/api/tamper/reset", methods=["POST"])
def api_tamper_reset():
    """Reset the camera tamper baseline reference frame."""
    streamer = get_streamer()
    streamer.reset_tamper_reference()
    return jsonify({"success": True, "message": "Tamper baseline reset successfully"})


@app.route("/api/alerts/test", methods=["POST"])
def api_alerts_test():
    """
    Trigger a manual test alert across WhatsApp and Email channels
    using the latest available evidence image.
    """
    from core.alerts import send_test_alert
    results = send_test_alert()
    wa_ok = results.get("whatsapp", {}).get("success", False)
    email_ok = results.get("email", {}).get("success", False)

    return jsonify({
        "success": bool(wa_ok or email_ok),
        "results": results,
    })


@app.route("/api/armed", methods=["GET", "POST"])
def api_armed():
    """Get or set system armed/disarmed status."""
    streamer = get_streamer()
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form or {}
        armed = data.get("armed")
        if armed is not None:
            if isinstance(armed, str):
                armed = armed.lower() in ("true", "1", "armed", "yes")
            res = streamer.set_armed(bool(armed))
            return jsonify({"armed": res, "status": "ARMED" if res else "DISARMED"})
        else:
            res = streamer.toggle_armed()
            return jsonify({"armed": res, "status": "ARMED" if res else "DISARMED"})

    status = streamer.get_status()
    is_armed = status.get("armed", True)
    return jsonify({"armed": is_armed, "status": "ARMED" if is_armed else "DISARMED"})


@app.route("/api/armed/toggle", methods=["POST"])
def api_armed_toggle():
    """Toggle system armed/disarmed status."""
    streamer = get_streamer()
    res = streamer.toggle_armed()
    return jsonify({"armed": res, "status": "ARMED" if res else "DISARMED"})


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """
    Retrieve or dynamically update detection thresholds.
    Accepts: min_contour_area, conf_threshold, alert_cooldown,
    min_person_confidence, person_confirm_seconds, person_absent_seconds.
    """
    streamer = get_streamer()
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form or {}
        min_area = data.get("min_contour_area")
        conf = data.get("conf_threshold")
        cooldown = data.get("alert_cooldown")
        min_person_conf = data.get("min_person_confidence")
        confirm_sec = data.get("person_confirm_seconds")
        absent_sec = data.get("person_absent_seconds")

        updated = streamer.update_settings(
            min_contour_area=min_area if min_area not in (None, "") else None,
            conf_threshold=conf if conf not in (None, "") else None,
            alert_cooldown=cooldown if cooldown not in (None, "") else None,
            min_person_confidence=min_person_conf if min_person_conf not in (None, "") else None,
            person_confirm_seconds=confirm_sec if confirm_sec not in (None, "") else None,
            person_absent_seconds=absent_sec if absent_sec not in (None, "") else None,
        )
        return jsonify({"success": True, "settings": updated})

    return jsonify(streamer.get_settings())


# ==============================================================================
# Application Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" SecureVision - Smart AI Security Monitoring Web Portal")
    print(f" Web Dashboard: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f" Admin Credentials: {config.ADMIN_USERNAME} / {config.ADMIN_PASSWORD}")
    print("=" * 60)

    # Initialize video background streamer
    get_streamer()

    # Run Flask server
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
        threaded=True,
    )
