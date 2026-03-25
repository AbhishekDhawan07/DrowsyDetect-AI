"""
app.py  –  Hugging Face Spaces deployment
==========================================
Gradio interface for the Drowsiness Detection project.
Video-only mode: uploads a video, processes it frame-by-frame,
saves the annotated output next to app.py, and returns it to the UI.

"""

import os
import datetime
import numpy as np
import cv2
import dlib
import gradio as gr

from drowsiness_detector import (
    analyze_frame,
    load_threshold,
    save_threshold,
    EAR_THRESHOLD,
    LANDMARK_MODEL_PATH,
    THRESHOLD_SAVE_PATH,
)

# All saved outputs land next to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# LOAD MODELS  (cached at module level)


_detector  = None
_predictor = None
_config    = None


def _load_models():
    global _detector, _predictor, _config

    if _config is None:
        if not os.path.exists(THRESHOLD_SAVE_PATH):
            save_threshold(EAR_THRESHOLD)
        _config = load_threshold()

    if _detector is None:
        _detector = dlib.get_frontal_face_detector()

    if _predictor is None:
        if not os.path.exists(LANDMARK_MODEL_PATH):
            raise FileNotFoundError(
                f"'{LANDMARK_MODEL_PATH}' not found in Space repository.\n"
                "Please add it via Git LFS (see README for instructions)."
            )
        _predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

    return _detector, _predictor, _config["ear_threshold"]



# GRADIO INFERENCE FUNCTION (for video )

def predict_video(video_path: str, ear_threshold: float):
    """
    Process an uploaded video file frame-by-frame.
    Returns : path to annotated output video, summary string.
    Saves   : annotated video as  video_YYYYMMDD_HHMMSS.mp4  next to app.py
    """
    detector, predictor, _ = _load_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Save annotated video next to app.py 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(BASE_DIR, f"video_{timestamp}.mp4")
    # Try H.264 first, fall back to mp4v if not supported
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"[INFO] Saving annotated video → {out_path}")

    total_frames  = 0
    drowsy_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, drowsy, _ = analyze_frame(frame, detector, predictor, ear_threshold)
        writer.write(annotated)
        total_frames  += 1
        drowsy_frames += int(drowsy)

    cap.release()
    writer.release()
    print(f"[INFO] Annotated video saved → {out_path}")

    pct     = (drowsy_frames / total_frames * 100) if total_frames else 0
    summary = (
        f"Processed **{total_frames}** frames | "
        f"Drowsy frames: **{drowsy_frames}** ({pct:.1f}%)\n\n"
        f"Saved to `{os.path.basename(out_path)}`"
    )
    return out_path, summary


def update_threshold(new_threshold: float):
    """Save a new EAR threshold and return confirmation."""
    save_threshold(new_threshold)
    global _config
    _config = load_threshold()
    return f"Threshold updated to **{new_threshold}** and saved to `{THRESHOLD_SAVE_PATH}`."



# GRADIO UI 

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Drowsiness Detection",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange"),
    ) as demo:

        gr.Markdown(
            """
            # 😴 Driver Drowsiness Detection
            ### Powered by dlib 68-point facial landmarks & Eye Aspect Ratio (EAR)

            Upload a **video** to check for signs of drowsiness.
            The model draws eye contours, flags drowsy frames, and saves the annotated output automatically.
            """
        )

        with gr.Tabs():

            # Tab 1 : Video
            with gr.TabItem("🎬 Video"):
                with gr.Row():
                    with gr.Column():
                        vid_input  = gr.Video(label="Upload Video (MP4 / AVI)")
                        vid_thresh = gr.Slider(
                            minimum=0.10, maximum=0.40, value=EAR_THRESHOLD, step=0.01,
                            label="EAR Threshold (lower = stricter)"
                        )
                        vid_btn    = gr.Button("Analyse Video", variant="primary")
                    with gr.Column():
                        vid_output  = gr.Video(label="Annotated Output (auto-saved)")
                        vid_summary = gr.Markdown()

                vid_btn.click(
                    fn=predict_video,
                    inputs=[vid_input, vid_thresh],
                    outputs=[vid_output, vid_summary],
                )

            # Tab 2 : Settings 
            with gr.TabItem("⚙️ Settings / Save Threshold"):
                gr.Markdown(
                    "Adjust and **permanently save** the EAR threshold used for detection."
                )
                save_slider = gr.Slider(
                    minimum=0.10, maximum=0.40, value=EAR_THRESHOLD, step=0.01,
                    label="New EAR Threshold"
                )
                save_btn = gr.Button("Save Threshold", variant="secondary")
                save_msg = gr.Markdown()

                save_btn.click(
                    fn=update_threshold,
                    inputs=[save_slider],
                    outputs=[save_msg],
                )

        gr.Markdown(
            """
            ---
            **How it works**
            - dlib's frontal face detector locates the face.
            - The 68-point shape predictor maps eye landmark coordinates.
            - Eye Aspect Ratio (EAR) = (vertical distances) / (2 × horizontal distance).
            - EAR < threshold for a sustained period → **Drowsiness Alert**.
            - Every processed video is **automatically saved** with a timestamp.

            **Download the landmark model**
            ```
            wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
            ```
            """
        )

    return demo


if __name__ == "__main__":
    _load_models()          # pre-warm models on local run
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
