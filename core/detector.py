import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from ultralytics import YOLO
import math

# ── YOLOv8n for phone detection ────────────────────────────────────────────────
# Auto downloads yolov8n.pt on first run
yolo_model = YOLO("yolov8n.pt")

# ── MediaPipe FaceLandmarker for lip + multi-face ──────────────────────────────
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=4,              # detect up to 4 faces (for multi-face detection)
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detector = FaceLandmarker.create_from_options(face_options)

# ── Lip landmark indices (MediaPipe) ──────────────────────────────────────────
# Outer lip points
LIP_TOP    = 13    # top of upper lip
LIP_BOTTOM = 14    # bottom of lower lip
LIP_MOVEMENT_THRESHOLD = 5.0 


def calculate_lip_distance(landmarks, w, h):
    """
    Exact method from your notebook!
    Uses euclidean distance between landmark 13 and 14.
    """
    top_lip    = np.array([landmarks[LIP_TOP].x    * w,
                           landmarks[LIP_TOP].y    * h])
    bottom_lip = np.array([landmarks[LIP_BOTTOM].x * w,
                           landmarks[LIP_BOTTOM].y * h])
    distance   = np.linalg.norm(top_lip - bottom_lip)
    return round(distance, 2)


def detect_phone(frame):
    """
    Uses YOLOv8n to detect mobile phones in frame.
    YOLO class 67 = cell phone in COCO dataset.
    Returns:
      {
        detected : bool,
        boxes    : list of [x1,y1,x2,y2] for each detected phone,
        confidence: float
      }
    """
    result = {
        "detected": False,
        "boxes": [],
        "confidence": 0.0
    }

    # Run YOLO — only check every other frame for performance
    # class 67=phone, 65=remote (back of phone looks like remote)
    # Run without class filter first to catch anything phone-like
    results = yolo_model(frame, verbose=False, classes=[67, 65, 63, 76])

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.20:  # very low threshold 
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Aspect ratio check — phones are rectangular
                bw = x2 - x1
                bh = y2 - y1
                aspect = bh / max(bw, 1)

                # Valid phone shape: portrait (tall) or landscape (wide)
                # aspect > 0.5 means not a tiny square blob
                if aspect > 0.5 or aspect < 2.0:
                    result["detected"]   = True
                    result["confidence"] = round(conf, 2)
                    result["boxes"].append([x1, y1, x2, y2])

    return result


def detect_faces_and_lips(frame):
    """
    Detects:
    1. Number of faces (for multi-face detection)
    2. Lip movement (MAR based)

    Returns:
      {
        face_count  : int,
        multi_face  : bool,
        lip_moving  : bool,
        mar         : float
      }
    """
    result = {
        "face_count": 0,
        "multi_face": False,
        "lip_moving": False,
        "mar":        0.0
    }

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det  = face_detector.detect(mp_img)

    if not det.face_landmarks:
        return result

    face_count = len(det.face_landmarks)
    result["face_count"] = face_count
    result["multi_face"] = face_count > 1

    # Calculate MAR for first (main) face
    lm  = det.face_landmarks[0]
    lip_dist = calculate_lip_distance(lm, w, h)
    result["mar"]        = lip_dist
    result["lip_moving"] = lip_dist > LIP_MOVEMENT_THRESHOLD

    return result


def run_all_detections(frame):
    """
    Master function — runs all detections on a single frame.
    Called from Flask /analyze_frame route.

    Returns combined dict with all detection results.
    """
    phone_result = detect_phone(frame)
    face_result  = detect_faces_and_lips(frame)

    return {
        "phone":      phone_result["detected"],
        "phone_conf": phone_result["confidence"],
        "phone_boxes":phone_result["boxes"],
        "faces":      str(face_result["face_count"]),
        "multi_face": face_result["multi_face"],
        "lips":       face_result["lip_moving"],
        "mar":        face_result["mar"]
    }


# ── Visual test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Detector test... Press Q to quit")
    print("Try: holding phone, having 2 faces, opening mouth")

    frame_count = 0
    last_results = {
        "phone": False, "phone_conf": 0.0,
        "phone_boxes": [], "faces": "1",
        "multi_face": False, "lips": False, "mar": 0.0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run every 3 frames, but KEEP last result so it doesn't blink
        if frame_count % 3 == 0:
            last_results = run_all_detections(frame)
    
        results = last_results

        # Run detections every 2 frames for performance
        if frame_count % 2 == 0:
            results = run_all_detections(frame)

            # Draw phone boxes
            for box in results.get("phone_boxes", []):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"PHONE {results['phone_conf']}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Status text
            y = 40
            # Phone
            phone_color = (0,0,255) if results["phone"] else (0,255,0)
            cv2.putText(frame,
                        f"Phone: {'DETECTED!' if results['phone'] else 'None'}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, phone_color, 2)
            y += 35

            # Faces
            face_color = (0,0,255) if results["multi_face"] else (0,255,0)
            cv2.putText(frame,
                        f"Faces: {results['faces']} "
                        f"{'⚠ MULTIPLE!' if results['multi_face'] else 'OK'}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, face_color, 2)
            y += 35

            # Lips
            lip_color = (0,165,255) if results["lips"] else (0,255,0)
            cv2.putText(frame,
                        f"Lips: {'MOVING! MAR={}'.format(results['mar']) if results['lips'] else 'Still'}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, lip_color, 2)

        cv2.imshow("Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()