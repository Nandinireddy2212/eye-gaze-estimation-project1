from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import base64
import numpy as np
import cv2
from datetime import datetime

# ── Import our AI modules ──────────────────────────────────────────────────────
from core.gaze import estimate_gaze, get_iris_position_only, setup_calibration
from core.detector import run_all_detections
from core.heatmap  import generate_both_heatmaps
from core.logger   import start_logger, stop_logger, get_logger

app = Flask(__name__)
app.secret_key = "proctoreye_secret_2024"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_students():
    with open("data/students.json", "r") as f:
        return json.load(f)

def save_students(data):
    with open("data/students.json", "w") as f:
        json.dump(data, f, indent=2)

def load_questions():
    with open("data/questions.json", "r") as f:
        return json.load(f)

def decode_frame(base64_str):
    """Converts base64 image string from browser to OpenCV frame"""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Home
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return redirect(url_for("login"))

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Login
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip().upper()
        password   = request.form.get("password",   "").strip()
        name_input = request.form.get("name",        "").strip().lower()

        data    = load_students()
        student = next((s for s in data["students"]
                        if s["id"] == student_id), None)

        if not student:
            error = "❌ Student ID not found!"
        elif student["password"] != password:
            error = "❌ Wrong password!"
        elif student["name"].lower() != name_input:
            error = "❌ Name doesn't match our records!"
        else:
            session["student"] = student
            return redirect(url_for("profile"))

    return render_template("login.html", error=error)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Signup
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    error   = None
    success = None
    if request.method == "POST":
        name       = request.form.get("name",       "").strip()
        student_id = request.form.get("student_id", "").strip().upper()
        branch     = request.form.get("branch",     "").strip()
        year       = request.form.get("year",       "").strip()
        password   = request.form.get("password",   "").strip()

        if not all([name, student_id, branch, year, password]):
            error = "❌ Please fill in all fields!"
        else:
            data     = load_students()
            existing = next((s for s in data["students"]
                             if s["id"] == student_id), None)
            if existing:
                error = f"❌ Student ID {student_id} already exists!"
            else:
                new_student = {
                    "id":          student_id,
                    "password":    password,
                    "name":        name,
                    "branch":      branch,
                    "college":     "Chaitanya Bharathi Institute of Technology",
                    "year":        year,
                    "photo":       f"https://api.dicebear.com/7.x/thumbs/svg?seed={name.replace(' ','')}",
                    "exams_taken": 0,
                    "avg_score":   0
                }
                data["students"].append(new_student)
                save_students(data)
                success = "✅ Account created! You can now login."

    return render_template("signup.html", error=error, success=success)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Profile
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/profile")
def profile():
    if "student" not in session:
        return redirect(url_for("login"))
    data  = load_questions()
    exams = data["exams"]
    return render_template("profile.html",
                           student=session["student"],
                           exams=exams)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Calibration
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/calibration/<exam_id>")
def calibration(exam_id):
    if "student" not in session:
        return redirect(url_for("login"))
    session["exam_id"] = exam_id
    return render_template("calibration.html", exam_id=exam_id)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Get iris position during calibration (NEW)
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/get_iris_position", methods=["POST"])
def get_iris_position():
    try:
        data       = request.get_json()
        frame_data = data.get("frame", "")
        if not frame_data:
            return jsonify({"detected": False})
        frame = decode_frame(frame_data)
        if frame is None:
            return jsonify({"detected": False})
        result = get_iris_position_only(frame)
        return jsonify(result if result else {"detected": False})
    except Exception as e:
        print(f"Iris error: {e}")
        return jsonify({"detected": False})

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Save Calibration + Train Polynomial Regression (UPDATED)
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/save_calibration", methods=["POST"])
def save_calibration():
    data         = request.get_json()
    calib_points = data.get("calibration", [])
    screen_w     = data.get("screen_width",  1920)
    screen_h     = data.get("screen_height", 1080)

    session["calibration"]   = data
    session["screen_width"]  = screen_w
    session["screen_height"] = screen_h

    print(f"📍 Received {len(calib_points)} calibration points")

    # Train polynomial regression model
    success = setup_calibration(
        calib_points,
        screen_w=screen_w,
        screen_h=screen_h,
        frame_w=640,
        frame_h=480
    )
    print(f"Calibration: {'✅ Success' if success else '⚠️ Failed'}")

    exam_id = session.get("exam_id", "exam1")
    return jsonify({
        "status":     "ok" if success else "partial",
        "points":     len(calib_points),
        "calibrated": success,
        "redirect":   f"/exam/{exam_id}"
    })

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Exam (UPDATED — setup calibration when exam starts)
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/exam/<exam_id>")
def exam(exam_id):
    if "student" not in session:
        return redirect(url_for("login"))

    data      = load_questions()
    exam_data = next((e for e in data["exams"]
                      if e["id"] == exam_id), None)
    if not exam_data:
        return redirect(url_for("profile"))

    # ── Start logger ───────────────────────────────────────────────────────────
    student = session["student"]
    start_logger(student["id"], exam_id)
    session["frame_w"] = 640
    session["frame_h"] = 480

    # ── Setup calibration mapping ──────────────────────────────────────────────
    calib_data   = session.get("calibration", {})
    calib_points = calib_data.get("calibration", []) if calib_data else []
    if calib_points:
        success = setup_calibration(
            calib_points,
            screen_w=session.get("screen_width",  1920),
            screen_h=session.get("screen_height", 1080),
            frame_w=640,
            frame_h=480
        )
        print(f"Calibration setup: {'✅' if success else '⚠️'} {len(calib_points)} pts")
    else:
        print("⚠️ No calibration data — using uncalibrated gaze")

    return render_template("exam.html",
                           exam=exam_data,
                           student=student)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Analyze Frame (UPDATED — use calibrated screen coords)
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    if "student" not in session:
        return jsonify({"error": "not logged in"})

    data      = request.get_json()
    frame_b64 = data.get("frame", "")

    if not frame_b64:
        return jsonify({"error": "no frame"})

    frame = decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "decode failed"})

    # ── Get screen dimensions ──────────────────────────────────────────────────
    screen_w = session.get("screen_width",  1920)
    screen_h = session.get("screen_height", 1080)

    # ── Run gaze estimation ────────────────────────────────────────────────────
    gaze_result = estimate_gaze(frame, screen_w, screen_h)

    # ── Run all detections ─────────────────────────────────────────────────────
    det_result  = run_all_detections(frame)

    # ── Log events ────────────────────────────────────────────────────────────
    logger = get_logger()
    if logger:
        if gaze_result["detected"]:
            # Store RAW camera coords (notebook method!)
            logger.log_gaze_point(
                gaze_result["cam_x"],
                gaze_result["cam_y"]
            )
        if not gaze_result["detected"]:
            logger.log_event(
                "Face Not Visible",
                frame=frame,
                save_snapshot=True
            )

        direction = gaze_result["gaze_direction"]
        if direction != "Center" and gaze_result["detected"]:
            logger.log_event(
                f"Gaze Away - {direction}",
                frame=frame,
                save_snapshot=True
            )

        if det_result["phone"]:
            logger.log_event(
                "Phone Detected",
                frame=frame,
                save_snapshot=True
            )

        if det_result["multi_face"]:
            logger.log_event(
                "Multiple Faces Detected",
                frame=frame,
                save_snapshot=True
            )

        if det_result["lips"]:
            logger.log_event(
                "Lip Movement Detected",
                frame=frame,
                save_snapshot=False
            )

    response = {
        "gaze":     gaze_result["gaze_direction"],
        "faces":    int(det_result["faces"]),
        "phone":    bool(det_result["phone"]),
        "lips":     bool(det_result["lips"]),
        "gaze_x":   int(gaze_result["gaze_x"]),
        "gaze_y":   int(gaze_result["gaze_y"]),
        "detected": bool(gaze_result["detected"])
    }

    return jsonify(response)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Save Exam Page Screenshot
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/save_screenshot", methods=["POST"])
def save_screenshot():
    data       = request.get_json()
    screenshot = data.get("screenshot", "")
    if not screenshot:
        return jsonify({"status": "no screenshot"})
    frame = decode_frame(screenshot)
    if frame is not None:
        path = os.path.join("reports", "exam_screenshot.png")
        cv2.imwrite(path, frame)
        session["exam_screenshot"] = path
        print(f"Exam screenshot saved: {path}")
    return jsonify({"status": "ok"})

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Submit Exam (UPDATED — use screen coords for heatmap)
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/submit_exam", methods=["POST"])
def submit_exam():
    if "student" not in session:
        return redirect(url_for("login"))

    form_data = request.form
    exam_id   = form_data.get("exam_id")
    data      = load_questions()
    exam_data = next((e for e in data["exams"]
                      if e["id"] == exam_id), None)

    # ── Grade answers ──────────────────────────────────────────────────────────
    score   = 0
    results = []
    for q in exam_data["questions"]:
        selected   = form_data.get(f"q{q['id']}", "Not Answered")
        correct    = q["answer"]
        is_correct = selected == correct
        if is_correct:
            score += 1
        results.append({
            "question":   q["question"],
            "selected":   selected,
            "correct":    correct,
            "is_correct": is_correct
        })

    # ── Stop logger ────────────────────────────────────────────────────────────
    summary = stop_logger()

    # ── Generate heatmaps ──────────────────────────────────────────────────────
    heatmap_screen  = None
    heatmap_overlay = None

    if summary and len(summary["gaze_points"]) >= 5:
        screen_w = session.get("screen_width",  1920)
        screen_h = session.get("screen_height", 1080)

        calib_data   = session.get("calibration", {})
        calib_points = calib_data.get("calibration", []) if calib_data else []

        heatmap_screen, heatmap_overlay = generate_both_heatmaps(
            gaze_points          = summary["gaze_points"],
            calib_points         = calib_points,
            exam_screenshot_path = session.get("exam_screenshot", None),
            screen_w             = screen_w,
            screen_h             = screen_h,
            frame_w              = session.get("frame_w", 640),
            frame_h              = session.get("frame_h", 480),
            session_id           = summary["session_id"]
        )

    # ── Update student score ───────────────────────────────────────────────────
    students_data = load_students()
    for s in students_data["students"]:
        if s["id"] == session["student"]["id"]:
            total            = s["exams_taken"] * s["avg_score"] + \
                               (score / len(exam_data["questions"]) * 100)
            s["exams_taken"] += 1
            s["avg_score"]    = round(total / s["exams_taken"], 1)
            session["student"] = s
            break
    save_students(students_data)

    # ── Save results to session ────────────────────────────────────────────────
    session["exam_results"] = {
        "exam_title":      exam_data["title"],
        "score":           score,
        "total":           len(exam_data["questions"]),
        "results":         results,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "heatmap_screen":  heatmap_screen,
        "heatmap_overlay": heatmap_overlay,
        "snapshots":       summary["snapshots"]       if summary else [],
        "events":          summary["events"]          if summary else [],
        "phone_count":     summary["phone_count"]     if summary else 0,
        "face_count":      summary["face_count"]      if summary else 0,
        "lip_count":       summary["lip_count"]       if summary else 0,
        "gaze_away_count": summary["gaze_away_count"] if summary else 0
    }

    return redirect(url_for("report"))

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Report
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/report")
def report():
    if "student" not in session or "exam_results" not in session:
        return redirect(url_for("profile"))
    return render_template("report.html",
                           student=session["student"],
                           results=session["exam_results"])

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Serve report images
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/reports/<filename>")
def serve_report(filename):
    from flask import send_from_directory
    return send_from_directory("reports", filename)

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Download PDF
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/download_report")
def download_report():
    if "exam_results" not in session:
        return redirect(url_for("profile"))
    return redirect(url_for("report"))

# ───────────────────────────────────────────────────────────────────────────────
# ROUTE: Logout
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)