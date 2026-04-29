# 📉 Attention Drop Detector — V8

> Know where viewers leave **before** you publish.

A Python-based video analysis tool that predicts attention drops in your content. Upload a video, get back a scored timeline, drop zones, reasons, and actionable fixes — all inside a clean dark-mode web UI.

---

## 🚀 What it does

Splits your video into time windows and scores each one across four signals:

| Signal | Method | Why it matters |
|--------|--------|----------------|
| **👤 Face** | MediaPipe FaceMesh (468 landmarks) → Face Detection → Haar Cascade | Human faces keep viewers watching |
| **🏃 Motion** | OpenCV Farneback Optical Flow | Static frames lose attention fast |
| **🔊 Audio** | RMS energy via ffmpeg → moviepy → librosa (3-tier fallback) | Dead audio = dead engagement |
| **🎬 Cuts** | PySceneDetect ContentDetector | Pacing keeps the brain stimulated |

Each window is scored, labelled (`engaging` / `neutral` / `drop`), and explained in plain English — with a suggested fix for every drop zone.

> **Limit:** Videos must be under 10 minutes. Configurable in `analyzer.py`.

---

## 🧠 Content Modes

| Mode | Best for | Weights |
|------|----------|---------|
| `facecam` | Talking head, YouTube face cam | Audio + Face |
| `podcast` | Long interviews, discussions | Audio dominant |
| `gaming` | Gameplay + commentary | Motion + Audio |
| `vlog` | Lifestyle, day-in-the-life | Balanced |
| `tutorial` | Screenshare, demos | Audio + Cuts |
| `shortform` | TikTok, Reels, Shorts | Cuts dominant |
| `cinematic` | Film, essay, documentary | Audio + Narration |
| `default` | Unknown / general | Balanced |

---

## 📦 Tech Stack

- **Video:** OpenCV — Farneback Optical Flow
- **Face:** MediaPipe FaceMesh (fallback: Face Detection → Haar Cascade)
- **Audio:** ffmpeg + Librosa RMS energy (fallback: moviepy → librosa direct)
- **Cuts:** PySceneDetect ContentDetector
- **UI:** Flask + vanilla HTML/CSS/JS
- **Packaging:** PyInstaller `--onedir` (Windows)

---

## 💻 Installation

> ⚠️ **Python 3.12 required.** MediaPipe does not support Python 3.13 on Windows yet.

**1. Clone**
```bash
git clone https://github.com/swaritsawarkar/attentiondropdetector.git
cd attentiondropdetector
```

**2. Install ffmpeg** (required for audio)
```bash
winget install ffmpeg
```
Restart your terminal after install.

**3. Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use

### Option 1: Web UI
```bash
python app.py
# or double-click run_dev.bat
```
Opens `http://localhost:7432`. Drop your video, pick a mode, hit **ANALYZE**.

### Option 2: CLI
```bash
python analyzer.py your_video.mp4 --mode facecam --window 5 --out result.json --chart
```

### Option 3: Extract Highlights
```bash
python highlights.py result.json --top 3 --min-len 15
```

---

## 🖥️ Windows EXE

Download from [Releases](../../releases). Extract and run `AttentionDropDetector.exe`.

**To build your own EXE:**
```bash
# Activate venv first, then:
build_exe.bat
# Output: dist\AttentionDropDetector\
```
Zip the entire `dist\AttentionDropDetector\` folder for distribution.

---

## ⚠️ Known Limitations

- Max video: **10 minutes** (configurable in `analyzer.py`)
- FaceMesh struggles with extreme side profiles (>70° from camera)
- Speech probability is estimated via ZCR + spectral centroid, not a classifier
- Scores are heuristic — not trained on real viewer retention data
- PySceneDetect may need threshold tuning on heavily compressed footage
- ffmpeg is required for reliable audio; without it, audio analysis is skipped

---

## 📋 Version History

| Version | Date | Summary |
|---------|------|---------|
| **v8** | 2026-04-29 | Audio RMS+smoothing, 3-tier audio fallback, max video limit, chart sizing fix, UI improvements |
| v7 | 2026-04-28 | Fixed face detection frame sampling bug |
| v6 | 2026-04-27 | FaceMesh migration, audio pipeline, modes, web UI |
| v5 | 2026-04-25 | Python 3.12 venv fix |

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

*Built with Python 🐍 · Made by Swarit Sawarkar*
