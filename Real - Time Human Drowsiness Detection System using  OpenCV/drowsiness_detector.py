"""
drowsiness_detector.py
======================
Core drowsiness detection module using dlib's 68-point facial landmark predictor.
Works with webcam feed and can also process single frames (used by app.py).
"""

import cv2
import dlib
import json
import os
import time
import datetime
import numpy as np
from scipy.spatial import distance


# CONSTANTS

# All outputs land next to this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LANDMARK_MODEL_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
EAR_THRESHOLD       = 0.25          # Eye Aspect Ratio below this → drowsy
CONSEC_FRAMES       = 20            # Consecutive frames below threshold before alert
THRESHOLD_SAVE_PATH = os.path.join(BASE_DIR, "ear_threshold.json")

# dlib landmark indices for each eye
RIGHT_EYE_IDX = list(range(42, 48))
LEFT_EYE_IDX  = list(range(36, 42))



# MODEL / CONFIG SAVE & LOAD


def save_threshold(threshold: float, path: str = THRESHOLD_SAVE_PATH) -> None:
    """Persist the EAR threshold so it can be reloaded without retraining."""
    config = {
        "ear_threshold": threshold,
        "consecutive_frames": CONSEC_FRAMES,
        "description": (
            "Eye Aspect Ratio (EAR) threshold for drowsiness detection. "
            "Values below this indicate closed/droopy eyes."
        ),
        "landmark_indices": {
            "right_eye": RIGHT_EYE_IDX,
            "left_eye":  LEFT_EYE_IDX,
        },
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Threshold config saved → {path}")


def load_threshold(path: str = THRESHOLD_SAVE_PATH) -> dict:
    """Load persisted EAR threshold config."""
    if not os.path.exists(path):
        print(f"[WARN] {path} not found – using defaults.")
        return {"ear_threshold": EAR_THRESHOLD, "consecutive_frames": CONSEC_FRAMES}
    with open(path, "r") as f:
        config = json.load(f)
    print(f"[INFO] Threshold config loaded ← {path}  (EAR = {config['ear_threshold']})")
    return config



# CORE DETECTION HELPERS

def eye_aspect_ratio(eye_points: list) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) using the 6 landmark points of one eye.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    poi_A = distance.euclidean(eye_points[1], eye_points[5])
    poi_B = distance.euclidean(eye_points[2], eye_points[4])
    poi_C = distance.euclidean(eye_points[0], eye_points[3])
    return (poi_A + poi_B) / (2.0 * poi_C)


def extract_eye_points(landmarks, indices: list) -> list:
    """Extract (x, y) tuples for a given set of landmark indices."""
    return [(landmarks.part(n).x, landmarks.part(n).y) for n in indices]


def draw_eye_contour(frame, eye_points: list, color: tuple) -> None:
    """Draw the eye contour polygon on frame."""
    pts = np.array(eye_points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def analyze_frame(frame: np.ndarray, detector, predictor, threshold: float):
    """
    Analyze a single BGR frame and return:
        - annotated_frame  : frame with landmarks + status overlay
        - drowsy            : bool
        - avg_ear           : float  (mean EAR of both eyes, -1 if no face)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    drowsy   = False
    avg_ear  = -1.0

    for face in faces:
        landmarks = predictor(gray, face)

        right_eye = extract_eye_points(landmarks, RIGHT_EYE_IDX)
        left_eye  = extract_eye_points(landmarks, LEFT_EYE_IDX)

        draw_eye_contour(frame, right_eye, (0, 255, 0))
        draw_eye_contour(frame, left_eye,  (255, 255, 0))

        ear_right = eye_aspect_ratio(right_eye)
        ear_left  = eye_aspect_ratio(left_eye)
        avg_ear   = round((ear_right + ear_left) / 2.0, 3)

        # Status overlay 
        ear_text = f"EAR: {avg_ear}"
        cv2.putText(frame, ear_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if avg_ear < threshold:
            drowsy = True
            cv2.putText(frame, "DROWSINESS DETECTED!", (30, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "  WAKE UP!", (30, 145),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 80, 255), 2)
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1], frame.shape[0]), (0, 0, 255), 4)
        else:
            cv2.putText(frame, "Alert & Awake", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

    return frame, drowsy, avg_ear



# LIVE WEBCAM DEMO  (run this file directly)

def run_webcam(camera_index: int = 0) -> None:
    """Run drowsiness detection on live webcam feed."""

    # Load / save config 
    config   = load_threshold()
    ear_thr  = config["ear_threshold"]
    consec   = config["consecutive_frames"]
    save_threshold(ear_thr)          # ensure file always exists after first run

    #  Load models
    if not os.path.exists(LANDMARK_MODEL_PATH):
        raise FileNotFoundError(
            f"Landmark model not found at '{LANDMARK_MODEL_PATH}'.\n"
            "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

    # pyttsx3(audio alert)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        audio_enabled = True
    except Exception:
        audio_enabled = False
        print("[WARN] pyttsx3 not available – audio alerts disabled.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    # ── Set up video writer (save next to this script) ──
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path    = os.path.join(BASE_DIR, f"webcam_{timestamp}.mp4")
    # Try H.264 first, fall back to mp4v if not supported
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"[INFO] Recording → {out_path}")

    frame_counter = 0
    last_alert    = 0

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, drowsy, ear = analyze_frame(frame, detector, predictor, ear_thr)
        writer.write(annotated)          # ← save every annotated frame

        if drowsy:
            frame_counter += 1
            if frame_counter >= consec:
                now = time.time()
                if audio_enabled and (now - last_alert) > 3:   # throttle alerts
                    engine.say("Wake up! Drowsiness detected.")
                    engine.runAndWait()
                    last_alert = now
        else:
            frame_counter = 0

        cv2.imshow("Drowsiness Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Webcam recording saved → {out_path}")


if __name__ == "__main__":
    # Save default threshold config on first run
    save_threshold(EAR_THRESHOLD)
    run_webcam(camera_index=0)
