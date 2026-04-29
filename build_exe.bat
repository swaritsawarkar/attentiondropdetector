@echo off
echo ==========================================
echo  Attention Drop Detector V8 — EXE Build
echo ==========================================
echo.

:: Activate venv
call venv\Scripts\activate.bat

:: Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building with PyInstaller (--onedir)...
pyinstaller ^
  --noconfirm ^
  --onedir ^
  --windowed ^
  --name "AttentionDropDetector" ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --collect-all mediapipe ^
  --collect-all scenedetect ^
  --collect-all librosa ^
  --hidden-import=moviepy ^
  app.py

echo.
echo Build complete! Output: dist\AttentionDropDetector\
echo Zip the dist\AttentionDropDetector folder for distribution.
pause
