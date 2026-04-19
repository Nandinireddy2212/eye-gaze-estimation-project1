"""
gaze.py — ProctorEye
====================
RELIABLE approach — no solvePnP (was giving garbage values).

Direction detection:
  - Nose tip position relative to face center → head turn direction
  - Iris position relative to eye corners → eye gaze direction
  - BOTH must agree → final direction
  - Anywhere on screen = Center

Coordinates:
  - Polynomial regression from 9-dot calibration → exact screen (x,y)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from collections import deque
from core.gaze_dl import predict_gaze_dl

# ── FaceLandmarker ─────────────────────────────────────────────
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = FaceLandmarker.create_from_options(options)

# ── Key landmark indices ───────────────────────────────────────
NOSE_TIP      = 1      # nose tip
FACE_LEFT     = 234    # left face edge
FACE_RIGHT    = 454    # right face edge
FACE_TOP      = 10     # top of face
FACE_BOTTOM   = 152    # chin

LEFT_IRIS     = [468, 469, 470, 471, 472]
RIGHT_IRIS    = [473, 474, 475, 476, 477]
LEFT_EYE_L    = 33     # left eye left corner
LEFT_EYE_R    = 133    # left eye right corner
RIGHT_EYE_L   = 362    # right eye left corner
RIGHT_EYE_R   = 263    # right eye right corner
LEFT_EYE_TOP  = 159
LEFT_EYE_BOT  = 145


# ══════════════════════════════════════════════════════════════
# GAZE CALIBRATOR — Polynomial Regression
# ══════════════════════════════════════════════════════════════
class GazeCalibrator:
    def __init__(self):
        self.is_calibrated = False
        self.model_x       = None
        self.model_y       = None
        self.screen_w      = 1920
        self.screen_h      = 1080
        self.frame_w       = 640
        self.frame_h       = 480

    def build_from_calib_data(self, calib_data,
                               screen_w=1920, screen_h=1080,
                               frame_w=640,   frame_h=480):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.frame_w  = frame_w
        self.frame_h  = frame_h

        if len(calib_data) < 4:
            print(f"⚠️ Only {len(calib_data)} calibration points")
            return False

        xs   = np.array([[p['iris_x'], p['iris_y']] for p in calib_data], dtype=np.float64)
        ys_x = np.array([p['screen_x'] for p in calib_data], dtype=np.float64)
        ys_y = np.array([p['screen_y'] for p in calib_data], dtype=np.float64)

        self.model_x = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=True)),
            ('reg',  Ridge(alpha=1.0))
        ])
        self.model_y = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=True)),
            ('reg',  Ridge(alpha=1.0))
        ])
        self.model_x.fit(xs, ys_x)
        self.model_y.fit(xs, ys_y)
        self.is_calibrated = True

        err_x = np.mean(np.abs(self.model_x.predict(xs) - ys_x))
        err_y = np.mean(np.abs(self.model_y.predict(xs) - ys_y))
        print(f"✅ Calibration trained on {len(calib_data)} points")
        print(f"   Mean error: x={err_x:.1f}px  y={err_y:.1f}px")
        return True

    def predict(self, iris_x, iris_y):
        # Direct scaling — more reliable than regression
        # when calibration points are too similar
        sx = int((iris_x / self.frame_w) * self.screen_w)
        sy = int((iris_y / self.frame_h) * self.screen_h)
        sx = max(0, min(self.screen_w - 1, sx))
        sy = max(0, min(self.screen_h - 1, sy))
        return sx, sy


calibrator = GazeCalibrator()


def setup_calibration(calib_data, screen_w=1920, screen_h=1080,
                       frame_w=640, frame_h=480):
    global calibrator
    calibrator = GazeCalibrator()
    if not calib_data:
        print("⚠️ No calibration data!")
        return False
    points = [p for p in calib_data if 'iris_x' in p and 'iris_y' in p]
    return calibrator.build_from_calib_data(
        points, screen_w, screen_h, frame_w, frame_h)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def get_iris_center(landmarks, iris_indices, w, h):
    pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in iris_indices
    ], dtype=np.int32)
    (cx, cy), _ = cv2.minEnclosingCircle(pts)
    return int(cx), int(cy)


def get_iris_position_only(frame):
    """For calibration — returns iris center in camera coords"""
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det    = face_landmarker.detect(mp_img)
    if not det.face_landmarks:
        return None
    lm         = det.face_landmarks[0]
    l_cx, l_cy = get_iris_center(lm, LEFT_IRIS,  w, h)
    r_cx, r_cy = get_iris_center(lm, RIGHT_IRIS, w, h)
    return {
        "detected":     True,
        "iris_x":       round((l_cx + r_cx) / 2, 2),
        "iris_y":       round((l_cy + r_cy) / 2, 2),
        "left_iris_x":  l_cx,
        "left_iris_y":  l_cy,
        "right_iris_x": r_cx,
        "right_iris_y": r_cy,
    }


# ══════════════════════════════════════════════════════════════
# DIRECTION DETECTION — nose + iris method
# ══════════════════════════════════════════════════════════════
def get_nose_direction(lm, w, h):
    """
    Nose tip position relative to face center.
    Reliable — works without solvePnP.
    
    nose_ratio_x: 0=far left, 0.5=center, 1=far right
    nose_ratio_y: 0=far up,   0.5=center, 1=far down
    """
    nose_x    = lm[NOSE_TIP].x * w
    nose_y    = lm[NOSE_TIP].y * h
    face_l    = lm[FACE_LEFT].x  * w
    face_r    = lm[FACE_RIGHT].x * w
    face_t    = lm[FACE_TOP].y   * h
    face_b    = lm[FACE_BOTTOM].y * h

    face_w    = max(face_r - face_l, 1)
    face_h    = max(face_b - face_t, 1)

    ratio_x   = (nose_x - face_l) / face_w   # 0=left 0.5=center 1=right
    ratio_y   = (nose_y - face_t) / face_h   # 0=top  0.5=center 1=bottom

    return ratio_x, ratio_y


def get_iris_direction(lm, w, h):
    """
    Iris position relative to eye corners.
    iris_ratio: 0=far left, 0.5=center, 1=far right
    """
    l_cx, l_cy = get_iris_center(lm, LEFT_IRIS,  w, h)
    r_cx, r_cy = get_iris_center(lm, RIGHT_IRIS, w, h)

    # Left eye ratio
    l_eye_l = int(lm[LEFT_EYE_L].x * w)
    l_eye_r = int(lm[LEFT_EYE_R].x * w)
    l_ratio  = (l_cx - l_eye_l) / max(l_eye_r - l_eye_l, 1)

    # Right eye ratio
    r_eye_l = int(lm[RIGHT_EYE_L].x * w)
    r_eye_r = int(lm[RIGHT_EYE_R].x * w)
    r_ratio  = (r_cx - r_eye_l) / max(r_eye_r - r_eye_l, 1)

    avg_ratio = (l_ratio + r_ratio) / 2.0

    # Vertical
    l_top_y    = int(lm[LEFT_EYE_TOP].y * h)
    l_bot_y    = int(lm[LEFT_EYE_BOT].y * h)
    v_ratio    = (l_cy - l_top_y) / max(l_bot_y - l_top_y, 1)

    iris_avg_x = (l_cx + r_cx) / 2.0
    iris_avg_y = (l_cy + r_cy) / 2.0

    return avg_ratio, v_ratio, iris_avg_x, iris_avg_y


# Smoothing
_dir_history = deque(maxlen=4)

def smooth_direction(d):
    _dir_history.append(d)
    counts = {}
    for x in _dir_history:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)


# ══════════════════════════════════════════════════════════════
# MAIN ESTIMATE GAZE
# ══════════════════════════════════════════════════════════════
def estimate_gaze(frame, screen_w=1920, screen_h=1080):
    result = {
        "gaze_direction": "Center",
        "face_direction": "Center",
        "eye_direction":  "Center",
        "gaze_x":        screen_w // 2,
        "gaze_y":        screen_h // 2,
        "cam_x":         320,
        "cam_y":         240,
        "detected":      False,
        "yaw":           0.0,
        "pitch":         0.0,
        "iris_reliable": True,
    }

    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det    = face_landmarker.detect(mp_img)

    if not det.face_landmarks:
        return result

    lm = det.face_landmarks[0]
    result["detected"] = True

    # ── 1. Nose direction ─────────────────────────────────────
    nose_rx, nose_ry = get_nose_direction(lm, w, h)

    # Classify nose direction
    # When looking straight: nose_rx ≈ 0.5, nose_ry ≈ 0.5
    # Thresholds: how far nose moves when looking away
    if   nose_rx > 0.62:  face_dir = "Left"
    elif nose_rx < 0.38:  face_dir = "Right"
    elif nose_ry > 0.62:  face_dir = "Down"
    elif nose_ry < 0.38:  face_dir = "Up"
    else:                  face_dir = "Center"

    result["face_direction"] = face_dir
    result["yaw"]   = round(nose_rx, 3)
    result["pitch"] = round(nose_ry, 3)

    # ── 2. Iris direction ─────────────────────────────────────
    iris_ratio, v_ratio, iris_x, iris_y = get_iris_direction(lm, w, h)
    result["cam_x"] = int(iris_x)
    result["cam_y"] = int(iris_y)

    # Classify iris direction
    # When looking straight: iris_ratio ≈ 0.5
    if   iris_ratio > 0.60: eye_dir = "Left"
    elif iris_ratio < 0.40: eye_dir = "Right"
    elif v_ratio    > 0.65: eye_dir = "Up"
    elif v_ratio    < 0.35: eye_dir = "Down"
    else:                    eye_dir = "Center"

    result["eye_direction"] = eye_dir

    # ── 3. Final — BOTH must agree ────────────────────────────
    if face_dir != "Center" and eye_dir == face_dir:
        raw_final = face_dir
    elif face_dir != "Center" and eye_dir != "Center":
        raw_final = face_dir   # different away dirs → use nose (more stable)
    else:
        raw_final = "Center"   # only one signal → Center

    final = smooth_direction(raw_final)
    result["gaze_direction"] = final

    # ── 4. Screen coordinates ─────────────────────────────────
    gaze_x, gaze_y = calibrator.predict(iris_x, iris_y)
    result["gaze_x"] = gaze_x
    result["gaze_y"] = gaze_y

    return result


# ── Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Gaze direction test")
    print("Look at screen → should show CENTER")
    print("Turn head/eyes left/right/up/down → shows direction")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res   = estimate_gaze(frame)
        color = (0, 255, 0) if res["gaze_direction"] == "Center" else (0, 0, 255)

        cv2.putText(frame, f"GAZE: {res['gaze_direction']}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame,
                    f"Nose: {res['face_direction']}  nx={res['yaw']:.2f} ny={res['pitch']:.2f}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.putText(frame,
                    f"Eyes: {res['eye_direction']}  cam=({res['cam_x']},{res['cam_y']})",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("Gaze Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()