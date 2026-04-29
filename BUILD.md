# Building the EXE — V8

> ⚠️ **Build inside Python 3.12 venv only.** MediaPipe does not support Python 3.13.

## Quick Build

```bash
# Activate venv first
.\venv\Scripts\activate

# Then run the build script
build_exe.bat
```

Output: `dist\AttentionDropDetector\`
Distribute by zipping the entire `dist\AttentionDropDetector\` folder.

---

## Manual Build Command

```bash
pyinstaller --noconfirm --onedir --windowed --name "AttentionDropDetector" ^
  --add-data "templates;templates" --add-data "static;static" ^
  --collect-all mediapipe --collect-all scenedetect --collect-all librosa ^
  --hidden-import=moviepy app.py
```

---

## Why `--onedir` and not `--onefile`?

`--onefile` extracts all files to a temp folder on every launch. With MediaPipe + OpenCV (500 MB+), this makes startup take 15–30 seconds and can break path resolution for model files. `--onedir` keeps everything in one folder and starts instantly.

---

## ffmpeg in the EXE

The EXE does **not** bundle ffmpeg. Users need ffmpeg installed separately:
```bash
winget install ffmpeg
```

If ffmpeg is missing, the app falls back to moviepy → librosa direct load → warns the user in the UI. The app will still work for video analysis; only audio analysis is affected.

---

## If PyInstaller fails

**Missing mediapipe models:**
```bash
--collect-all mediapipe
```
Already included in `build_exe.bat`.

**Missing hidden imports:**
```bash
--hidden-import=mediapipe.python.solutions.face_mesh
--hidden-import=moviepy.editor
```

**DLL load errors (OpenCV):**
Make sure you're building inside venv on the same Windows machine you'll run it on.
