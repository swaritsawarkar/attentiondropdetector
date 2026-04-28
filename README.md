# 📉 Attention Drop Detector V6

This thing automatically sniffs out "attention drops" in your videos – those moments where viewers are probably zoning out because nothing's happening.

Perfect for optimizing YouTube videos, Podcasts, Shorts/Reels, and Vlogs.

## 🚀 Features

The analyzer breaks down your video into sliding windows and scores them based on 4 key signals:
- **👤 Face Presence:** Uses **MediaPipe FaceMesh** (468 landmarks) to robustly detect faces, even with glasses, movement, or partial occlusion.
- **🏃 Motion Tracking:** Uses **OpenCV Farneback Optical Flow** to track pixel movement and ensure the frame isn't static.
- **🔊 Audio Energy:** Uses **Librosa** to analyze spectral speech-band energy and speech probability.
- **🎬 Scene Cuts:** Uses **PySceneDetect** to calculate the frequency of camera angle changes and B-roll.

---

## What was wrong with face detection before?

So, in V4, I was using `FaceDetection` from MediaPipe. The problem with `FaceDetection` is it just looks for a face-shaped bounding box in the image. The moment there was any of the following, it just fell apart:

-   **Glasses:** Totally messed it up because it occludes key facial regions it looks for.
-   **Arm movement:** Any motion blur on the frame when sampled would throw it off.
-   **Any head angle that isn't dead straight into the camera:** Forget about it.
-   **Slightly bad lighting:** Instant fail.

Lowering the confidence threshold to 0.25 helped a *little*, but it didn't fix the root cause. The model was still fundamentally looking for a clean, frontal face and just failing to find one if things weren't perfect.

## What V6 does instead (the proper fix!)

I switched to **MediaPipe FaceMesh**. This is a completely different approach, and it's way better.

FaceMesh fits 468 3D facial landmarks onto your face. It doesn't look for a face-shaped region; it actively tracks a mesh across your face and tries to fit it even when things are partially obscured. It's dramatically more robust because:

-   **Glasses:** Landmarks are still visible around the glasses, so the mesh still fits.
-   **Movement:** It tracks the mesh across frames, not a fresh detection each time. This means it's much more stable.
-   **Angles:** It's a 3D mesh, so it handles rotation naturally.
-   **Blur:** The tracking confidence threshold is set to a super low 0.1, so it keeps going even on blurry frames.

The confidence threshold is intentionally very low (0.1 detection, 0.1 tracking). We don't care about super clean, confident detections; we just want to know *if* a face is present or not. The `face_conf` value in the output is now the fraction of sampled frames where the mesh was found, not a model confidence score.

I also bumped the frame sampling from 16 to 20 per window, so there are more chances to catch your face.

---

## Other cool changes in V6

-   **Cleaner codebase:** Removed a bunch of redundant code and made things tighter.
-   **UI redesign:** The web interface got a facelift with tighter spacing and cleaner cards.
-   **Face percentage in summary:** Now you can actually see the face detection percentage right in the summary cards.
-   **Window size selector:** The broken slider is gone, replaced with nice, clickable buttons (this was already in V4 but cleaned up).

---

## Full Signal List

| Signal | Method |
|---|---|
| **Face** | MediaPipe FaceMesh, 468 landmarks, min confidence 0.1, 20 samples per window |
| **Motion** | Farneback Optical Flow, soft capped at 8px/frame |
| **Audio** | Librosa spectral energy in 300-3400Hz band + speech probability |
| **Cuts** | PySceneDetect ContentDetector, threshold 27.0 |

---

## Limitations (because nothing's perfect, right?)

-   FaceMesh still struggles with extreme side profiles (more than ~70 degrees from the camera).
-   Speech probability is estimated, not a definitive classification.
-   PySceneDetect may need threshold tuning on very dark or heavily compressed footage.
-   Scores are heuristic, meaning they're based on educated guesses, not trained on actual viewer retention data.

## 💻 Installation & Setup

> **⚠️ IMPORTANT: PYTHON VERSION**
> You MUST use **Python 3.12 or older**. Google's MediaPipe does not yet support Python 3.13 on Windows and will silently fail to load.

1. **Clone the repo**
2. **Install system dependencies:**
   - Install FFmpeg and ensure it is added to your system `PATH`.
     *(Windows users can simply run: `winget install ffmpeg`)*
3. **Create a Virtual Environment (Highly Recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Use

### Option 1: Web UI (Flask)
The easiest way to use the tool is via the web interface.

```bash
python app.py
```
*This will automatically open `http://localhost:7432` in your browser. Upload a video and watch the magic happen.*

### Option 2: Command Line (Standalone Analyzer)
You can run the analyzer directly from your terminal and output a JSON file.

```bash
python analyzer.py your_video.mp4 --mode default --chart
```

**Available Modes:**
You can adjust the scoring weights based on the content type:
- `default`
- `facecam` (Heavily weights face presence and audio)
- `podcast` (Heavily weights audio, ignores motion)
- `gaming` (Weights motion and audio)
- `vlog`
- `tutorial`
- `shortform` (Requires frequent cuts and high energy)
- `cinematic`

### Option 3: Extract Highlights
Want to find the most engaging parts of your video for TikTok or Shorts?

```bash
python highlights.py result.json --top 3 --min-len 15
```

## ⚠️ Known Issues / WIP
- **PyInstaller Executable Build:** Building this project into a standalone `.exe` using PyInstaller currently causes missing module warnings (specifically with `PySceneDetect` and `MediaPipe`). For now, run via raw Python.

---
*Built with Python 🐍*