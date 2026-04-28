"""
Attention Drop Detector V6 - app.py
Flask backend. Handles uploads, runs analysis in background thread,
streams progress via polling. Opens browser automatically on launch.
"""

import base64
import json
import os
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from analyzer import analyze, MODES
from visualizer import plot

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _run(job_id: str, video_path: str, window_sec: float, mode: str) -> None:
    def prog(step: int, msg: str):
        with LOCK:
            JOBS[job_id]["progress"] = {"step": step, "msg": msg}

    try:
        import analyzer as _an

        prog(1, "Extracting motion...")
        windows, duration = _an.MotionExtractor(video_path, window_sec).extract()

        prog(2, "Detecting scene cuts...")
        cuts = _an.detect_cuts(video_path)
        _an.assign_cuts(cuts, windows)

        prog(3, "Analyzing audio...")
        _an.AudioExtractor(video_path).enrich(windows)

        prog(4, "Detecting faces (FaceMesh)...")
        _an.FaceExtractor(video_path, window_sec).enrich(windows)

        prog(5, "Scoring and explaining...")
        _an.Scorer(mode).score(windows)
        drops   = _an.Explainer(mode).explain(windows, window_sec)
        summary = _an.make_summary(windows, drops)

        result = _an.Result(
            video=video_path, duration=round(duration, 2),
            window_sec=window_sec, mode=mode,
            windows=windows, drops=drops, summary=summary,
        )

        prog(6, "Generating chart...")
        chart_path = video_path + "_chart.png"
        plot(result, save_path=chart_path, show=False)

        chart_b64 = ""
        if os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                chart_b64 = base64.b64encode(f.read()).decode()
            os.remove(chart_path)

        with LOCK:
            JOBS[job_id].update({
                "status":    "done",
                "result":    asdict(result),
                "chart_b64": chart_b64,
                "progress":  {"step": 6, "msg": "Done"},
            })

    except Exception as e:
        with LOCK:
            JOBS[job_id].update({"status": "error", "error": str(e)})
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def start_analysis():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file       = request.files["video"]
    window_sec = float(request.form.get("window", 5.0))
    mode       = request.form.get("mode", "default")
    if mode not in MODES:
        mode = "default"

    suffix = Path(file.filename).suffix or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()

    job_id = str(uuid.uuid4())
    with LOCK:
        JOBS[job_id] = {
            "status":    "running",
            "progress":  {"step": 0, "msg": "Starting..."},
            "result":    None,
            "chart_b64": "",
            "error":     None,
        }

    threading.Thread(target=_run, args=(job_id, tmp.name, window_sec, mode), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


if __name__ == "__main__":
    import webbrowser
    port = 7432
    print(f"Attention Drop Detector V6 — http://localhost:{port}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(port=port, debug=False, threaded=True)
