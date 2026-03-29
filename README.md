<div align="center">

# 😴 DrowsyDetect-AI

### Real-Time Human Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-19.x-FF6F00?style=for-the-badge&logoColor=white)](http://dlib.net/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A production-ready, computer-vision-powered drowsiness detection system - deployed live as an interactive Gradio web app on Hugging Face Spaces.**

<br/>

[🚀 Try It Live →](https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system)

</div>

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 About the Project

**DrowsyDetect-AI** is a real-time drowsiness detection system that monitors a person's eye activity using computer vision and facial landmark geometry. It calculates the **Eye Aspect Ratio (EAR)** - a ratio derived from the vertical and horizontal distances of 6 eye landmark points - to detect whether eyes are drooping or closing.

When the EAR drops **below a configurable threshold** for a sustained number of frames, the system flags the person as drowsy and triggers an alert.

The system works in **two modes**:

| Mode | Description |
|------|-------------|
| 🎥 **Live Webcam** | Real-time detection with OpenCV window + audio alert via `pyttsx3` |
| 📤 **Video Upload** | Upload any pre-recorded video via the Gradio web app and receive an annotated output |

> 💡 Originally designed for **driver safety**, DrowsyDetect-AI is applicable to any human fatigue monitoring scenario - students, security personnel, remote workers, and more.

---

## 🌐 Demo

> **🔗 [https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system](https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system)**

The app is **live and free to use** - no installation, no sign-up. Upload any video and get instant drowsiness analysis with a fully annotated output.

<br/>

### 🎬 Hugging Face Deployment — Demo Video


https://github.com/user-attachments/assets/5677f71b-acb5-4c27-b84c-df7f4cb20825



> *The video above shows the full Hugging Face Spaces deployment in action - uploading a video, adjusting the EAR sensitivity slider, processing frames, and receiving the annotated output with drowsiness statistics.*

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10** | Core language |
| **OpenCV** | Frame capture, video I/O, drawing, codec handling |
| **dlib** | HOG-based face detection + 68-point landmark prediction |
| **SciPy** | Euclidean distance calculation for EAR |
| **NumPy** | Array operations on frame data |
| **Gradio** | Web UI + Hugging Face Spaces SDK |
| **pyttsx3** | Text-to-speech audio alerts (webcam mode) |

---

## ✨ Features

<details open>
<summary><b>🔍 Core Detection Engine</b></summary>
<br/>

- **68-Point Facial Landmark Detection** using dlib's pre-trained shape predictor
- **Eye Aspect Ratio (EAR)** computed for both eyes simultaneously and averaged
- **Consecutive frame analysis** - avoids false positives by requiring sustained low EAR
- **Real-time annotation** - eye contours, EAR value, and status text drawn directly on every frame

</details>

<details open>
<summary><b>🎛️ Fully Configurable</b></summary>
<br/>

- **Adjustable EAR threshold** - tune sensitivity live via a Gradio slider (range: 0.10 – 0.40)
- **Persistent threshold storage** - settings saved to `ear_threshold.json`, survive app restarts
- **Per-session override** - change the threshold mid-session without restarting
- **Configurable consecutive frames** - control how many frames of low EAR trigger an alert

</details>

<details open>
<summary><b>📹 Video Processing</b></summary>
<br/>

- Process any **MP4 or AVI** video file, frame by frame
- **Annotated output video** auto-saved with a timestamp (`video_YYYYMMDD_HHMMSS.mp4`)
- **H.264 / mp4v codec fallback** for maximum compatibility across platforms
- Summary stats returned: total frames, drowsy frame count, and drowsiness percentage

</details>

<details open>
<summary><b>🔊 Live Webcam Mode</b></summary>
<br/>

- Real-time OpenCV window with live eye-contour overlay
- **Audio alert** via `pyttsx3` - speaks *"Wake up! Drowsiness detected."*
- Alert **throttling** - no repeated alerts within 3 seconds
- Webcam session **auto-recorded** and saved locally with a timestamped filename

</details>

<details open>
<summary><b>🌐 Gradio Web App + Hugging Face Spaces</b></summary>
<br/>

- Two-tab interface: **🎬 Video Analysis** and **⚙️ Settings / Save Threshold**
- **Model caching** - dlib models loaded once at startup, reused across all requests
- Annotated output video rendered and downloadable directly in the browser
- Deployed on **Hugging Face Spaces** - accessible from any browser, globally

</details>

---

## 📊 Dataset

This project does not use a training dataset - it relies on a **pre-trained dlib model** for facial landmark detection.

| Model | Source | Size |
|-------|--------|------|
| `shape_predictor_68_face_landmarks.dat` | [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) | ~100 MB |

The model detects **68 facial landmarks** per face. Eye landmarks (points 36–47) are extracted to compute the Eye Aspect Ratio (EAR) in real time - no custom training required.

---

## 🧠 How It Works

The entire detection pipeline runs on a single formula — the **Eye Aspect Ratio (EAR)**:

```
         || p2 − p6 || + || p3 − p5 ||
EAR  =  ─────────────────────────────────
                 2 × || p1 − p4 ||
```

Where `p1`–`p6` are the 6 eye landmark points ordered clockwise around the eye. When the eye is **fully open**, EAR hovers around `0.28–0.35`. As eyes **close or droop**, EAR falls toward `0.0`.

### Pipeline — Step by Step

```
  Input Frame (BGR)
        │
        ▼
  ┌─────────────────────┐
  │  Grayscale Convert  │  ← cv2.COLOR_BGR2GRAY
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │   Face Detection    │  ← dlib.get_frontal_face_detector()
  └─────────────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  68-Point Landmark Map   │  ← shape_predictor_68_face_landmarks.dat
  └──────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Extract Eye Coordinates                 │
  │  Right Eye → landmarks [42 : 48]         │
  │  Left Eye  → landmarks [36 : 42]         │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Compute EAR (each eye) → Average        │
  └──────────────────────────────────────────┘
        │
        ├──── avg EAR < threshold ? ────► 🚨 DROWSINESS ALERT
        │                                   Red border + text overlay
        │                                   Audio alert (webcam mode)
        │
        └──── avg EAR ≥ threshold  ────► ✅ Alert & Awake
                                            Green text overlay
```

### Visual Output

| State | What You See on Frame |
|-------|-----------------------|
| 😴 **Drowsy** | Red border around entire frame · `DROWSINESS DETECTED!`  `WAKE UP!`  EAR value in cyan |
| 😊 **Awake** | No border · `Alert & Awake` in green · EAR value in cyan |

---

## 📁 Project Structure

```
DrowsyDetect-AI/
├── 📄 README.md                          ← You are here!!
└── 📂 Real - Time Human Drowsiness Detection System/
    ├── 📄 app.py
    ├── 📄 drowsiness_detector.py
    ├── 🤖 shape_predictor_68_face_landmarks.dat
    ├── ⚙️  ear_threshold.json
    ├── 📋 requirements.txt
    ├── 📦 packages.txt
    ├── 🎬 Local_Test_Drowsiness.mp4
    ├── 🌐 Web_App_drowsiness_Gradio.mp4
    ├── 📷 Drowsiness_Webcam_for_Gradio.mp4
    └── 🤗 Hugging_Face_Drowsiness.mp4
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `app.py` | Gradio UI with two tabs (Video + Settings). Loads dlib models once and caches them. Calls `analyze_frame()` for every video frame and saves annotated output automatically. |
| `drowsiness_detector.py` | Contains all detection logic: `eye_aspect_ratio()`, `analyze_frame()`, `run_webcam()`, `save_threshold()`, `load_threshold()`. Run directly for live webcam mode. |
| `shape_predictor_68_face_landmarks.dat` | Pre-trained dlib model for 68-point facial landmark detection. Must be present in the project root. (~100 MB, tracked via Git LFS) |
| `ear_threshold.json` | JSON config storing the EAR threshold and consecutive-frame count. Auto-created on first run. |
| `packages.txt` | System-level dependencies for Hugging Face Spaces (e.g., `cmake`, `libgl1`). |
| `requirements.txt` | All Python dependencies (`opencv-python`, `dlib`, `gradio`, `scipy`, etc.). |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- CMake (required to compile dlib)
- Webcam — only needed for live mode

### 1 — Clone the Repository

```bash
git clone https://github.com/Abhish07/DrowsyDetect-AI.git
cd DrowsyDetect-AI
```

### 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3 — Download the dlib Landmark Model

> ⚠️ The model file is ~100 MB. Download it separately and place it in the project root.

```bash
# Linux / macOS
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Windows (PowerShell)
Invoke-WebRequest -Uri "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -OutFile "model.bz2"
```

After extraction, confirm the file is at:
```
DrowsyDetect-AI/shape_predictor_68_face_landmarks.dat
```

### ▶️ Mode 1 — Live Webcam Detection

```bash
python drowsiness_detector.py
```

- Opens your default webcam (`camera_index=0`)
- Shows a real-time OpenCV window with annotated eye landmarks
- Plays audio alert when drowsiness is detected
- Press **`q`** to quit — session auto-saved as `webcam_YYYYMMDD_HHMMSS.mp4`

### ▶️ Mode 2 — Gradio Web App (Local)

```bash
python app.py
```

- Launches at `http://localhost:7860`
- A **public shareable URL** is also printed (via Gradio's `share=True`)
- Upload any `.mp4` or `.avi` file and click **Analyse Video**

### ▶️ Mode 3 — Hugging Face Spaces (No Install Required)

> 🔗 **[https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system](https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system)**

Open the link in any browser, upload your video, adjust the EAR slider, and get results instantly.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find a bug:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for road safety and human wellbeing**

⭐ If this project helped you, please give it a star!

[🚀 Try the Live App](https://huggingface.co/spaces/Abhish07/real-time-human-drowsiness-system)

</div>

