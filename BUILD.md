# Building the exe — V5

> **⚠️ IMPORTANT:** You must build this using **Python 3.12 or older** inside a Virtual Environment (`venv`). Python 3.13 will break the MediaPipe integration!

## install dependencies

1. Create and activate a venv:
```cmd
python -m venv venv
.\venv\Scripts\activate
```
pip install flask pyinstaller opencv-python numpy librosa matplotlib mediapipe scenedetect
```

## build

```
python -m PyInstaller attentiondrop.spec
```

exe outputs to `dist/AttentionDropDetector.exe`

## run without exe (just python)

```
python app.py
```

opens at http://localhost:7432

## what each library does

| library | role |
|---|---|
| flask | web server inside the exe |
| opencv-python | video reading + Farneback optical flow |
| numpy | numeric processing |
| librosa | spectral audio analysis |
| matplotlib | chart generation |
| mediapipe | FaceMesh face detection (works with glasses, movement, angles) |
| scenedetect | accurate cut detection |
| pyinstaller | packages everything into the exe |
