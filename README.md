# 📉 Attention Drop Detector — V7

> Know where viewers leave **before** you publish.

A Python-based video analysis tool that predicts attention drops in your content. Upload a video, get back a scored timeline, a list of drop zones, the reasons behind each drop, and actionable fixes — all inside a clean web UI.

---

## 🚀 What it does

The analyzer splits your video into time windows and scores each one across four signals:

| Signal | Method | Why it matters |
|---|---|---|
| **👤 Face Presence** | MediaPipe FaceMesh (468 landmarks) | Human faces keep viewers watching |
| **🏃 Motion** | OpenCV Farneback Optical Flow | Static frames lose attention fast |
| **🔊 Audio Energy** | Librosa spectral energy (300–3400 Hz) + speech probability | Dead audio = dead engagement |
| **🎬 Scene Cuts** | PySceneDetect ContentDetector | Pacing keeps the brain stimulated |

Each window is scored, labelled (`engaging` / `neutral` / `drop`), and explained in plain English — with a suggested fix for every drop zone.

---

## 🧠 Content Modes

Scoring weights are tuned per content type so the drop thresholds make sense for your format:

| Mode | Weights |
|---|---|
| `facecam` | Audio + Face dominant |
| `podcast` | Audio dominant |
| `gaming` | Motion + Audio |
| `vlog` | Balanced |
| `tutorial` | Audio + Cuts |
| `shortform` | Cuts dominant (TikTok / Reels pacing) |
| `cinematic` | Audio + Narration |
| `default` | General balanced |

---

## 📦 Tech Stack

- **Video:** OpenCV (motion via Farneback Optical Flow)
- **Face:** MediaPipe FaceMesh
- **Audio:** ffmpeg (extraction) + Librosa (spectral analysis)
- **Cuts:** PySceneDetect ContentDetector
- **UI:** Flask + vanilla HTML/CSS/JS
- **Packaging:** PyInstaller (Windows `.exe`)

---

## 💻 Installation & Setup

> **⚠️ Python version:** Use **Python 3.12 or older**. MediaPipe does not yet support Python 3.13 on Windows.

**1. Clone the repo**
```bash
git clone https://github.com/swaritsawarkar/attentiondropdetector.git
cd attentiondropdetector
```

**2. Install FFmpeg** (required for audio extraction)
```bash
winget install ffmpeg
```

**3. Create a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use

### Option 1: Web UI (recommended)
```bash
python app.py
```
Opens `http://localhost:7432` automatically. Drag-and-drop your video, pick a content mode and window size, hit **Analyze**.

### Option 2: Command Line
```bash
python analyzer.py your_video.mp4 --mode facecam --window 5 --out result.json --chart
```

### Option 3: Extract Highlights
Find the most engaging segments for Shorts / Reels:
```bash
python highlights.py result.json --top 3 --min-len 15
```

---

## 🖥️ Running the EXE (Windows)

Download the latest release from the [Releases page](../../releases). Extract and run `app.exe` — no Python required.

---

## 📋 Version History

### V7 — Face Detection Fix
- **Fixed:** Face detection was sampling all frames from the very start of each window instead of spreading samples across the full window duration. Root cause: `cap.read()` in the loop advanced sequentially but the loop variable controlling step size was never used to seek — so a 5-second window at 30fps only ever saw its first ~0.7 seconds. Fixed with explicit `cap.set()` seeks evenly distributed across the window.
- **Added:** All-black frame guard — some codecs return zero-value frames on bad seeks that silently fail MediaPipe. These are now skipped.
- **Improved:** Large frames (>1280px wide) are now downscaled before inference for speed. Previously only upscaling was handled.

### V6 — FaceMesh Migration
- Replaced `FaceDetection` with `FaceMesh` (468 landmarks) for robustness with glasses, angles, movement, and partial occlusion.
- Rewrote audio pipeline: ffmpeg extraction → Librosa spectral analysis in the 300–3400 Hz speech band.
- Added content mode system with per-mode scoring weights.
- New web UI with drag-and-drop upload, mode picker, window size selector, and score timeline chart.
- PyInstaller EXE build working end-to-end.

### V5 — Python 3.12 Migration
- Fixed Python 3.13 incompatibility with MediaPipe.

### V4 — Initial MediaPipe Integration
- Introduced MediaPipe FaceDetection (later replaced in V6).

---

## ⚠️ Limitations

- FaceMesh can struggle with extreme side profiles (>70° from camera).
- Speech probability is estimated (ZCR + spectral centroid heuristic), not a trained classifier.
- Scores are heuristic — not trained on real viewer retention data.
- PySceneDetect may need threshold tuning on heavily compressed or very dark footage.

---

*Built with Python 🐍 · Made by Swarit Sawarkar*
