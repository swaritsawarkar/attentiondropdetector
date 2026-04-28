# Building the exe — V5

> **⚠️ IMPORTANT:** You must build this using **Python 3.12 or older** inside a Virtual Environment (`venv`). Python 3.13 will break the MediaPipe integration!

## 1. Setup Environment

First, create and activate a Python virtual environment. This keeps your project's dependencies isolated.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (for Windows PowerShell)
.\venv\Scripts\activate
```

Next, install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 2. Build the `.exe`

Run the PyInstaller command. This bundles your `app.py`, all its dependencies, and the necessary data files (`templates`, `static`) into a single directory.

```bash
pyinstaller --noconfirm --onedir --windowed --add-data "templates;templates" --add-data "static;static" --collect-all mediapipe --collect-all scenedetect --collect-all librosa app.py
```

## 3. Run the Application

The final executable will be located inside the `dist/app` folder. You can run it by double-clicking `app.exe` or by running this command from your project root:

```bash
.\dist\app\app.exe
```

This will start the server and automatically open the user interface in your web browser.
