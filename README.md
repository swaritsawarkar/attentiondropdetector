# 📉 Attention Drop Detector (v5)

An automated video analysis tool designed to detect "attention drops"—moments where viewer engagement is likely to fall due to a lack of motion, audio energy, facial presence, or scene cuts. 

Perfect for optimizing YouTube videos, Podcasts, Shorts/Reels, and Vlogs.

## 🚀 Features

The analyzer breaks down your video into sliding windows and scores them based on 4 key signals:
- **👤 Face Presence:** Uses **MediaPipe FaceMesh** (468 landmarks) to robustly detect faces, even with glasses, movement, or partial occlusion.
- **🏃 Motion Tracking:** Uses **OpenCV Farneback Optical Flow** to track pixel movement and ensure the frame isn't static.
- **🔊 Audio Energy:** Uses **Librosa** to analyze spectral speech-band energy and speech probability.
- **🎬 Scene Cuts:** Uses **PySceneDetect** to calculate the frequency of camera angle changes and B-roll.

## 🧠 Technical Challenges Overcome (Portfolio Notes)
Building a reliable, offline AI-analysis tool on Windows brought several unique engineering challenges:
1. **Audio Decoding & FFmpeg Resiliency:** Standard audio loaders (`librosa`/`audioread`) silently failed on Windows MP4s due to missing system bindings. I built a robust `subprocess` pipeline that safely interfaces with FFmpeg to extract temporary uncompressed `.wav` files, guaranteeing accurate spectral energy scoring.
2. **MediaPipe C++ Binding Crashes:** Discovered that Google's MediaPipe lacks pre-compiled C++ binaries for the newly released Python 3.13, causing silent module failures. I engineered custom traceback error handling to detect this environment mismatch and strictly isolated the app in a locked Python 3.12 Virtual Environment (`venv`).
3. **Temporal Tracking Limitations:** Seeking to specific video timestamps (`cap.set()`) broke the temporal continuity expected by MediaPipe's tracking algorithms (resulting in 0% face detection). Fixed by explicitly forcing `static_image_mode=True` to treat every sampled frame as a fresh inference.
4. **PyInstaller Executable Packaging:** Freezing heavy AI libraries into a standalone `.exe` caused missing dynamic DLLs and hidden imports. Solved by mapping comprehensive `--collect-all` hooks for `mediapipe`, `scenedetect`, and `librosa` directly into the build pipeline.

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