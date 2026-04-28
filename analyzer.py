"""
Attention Drop Detector V6 - analyzer.py
==========================================
Signal pipeline:
  - Face: MediaPipe FaceMesh (468 landmarks, works with glasses/movement/angles)
  - Motion: Farneback Optical Flow
  - Audio: Librosa spectral speech-band energy + speech probability
  - Cuts: PySceneDetect ContentDetector
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

# Suppress TensorFlow/MediaPipe C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
    import librosa.feature
except ImportError:
    librosa = None

# FaceMesh is way more robust than FaceDetection
# works with glasses, movement, angles, partial occlusion
HAS_MEDIAPIPE = False
_mp_face_mesh = None
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as _mp_face_mesh_module
    _mp_face_mesh = _mp_face_mesh_module
    HAS_MEDIAPIPE = True
except Exception as e:
    print("\n  [DEBUG] MediaPipe failed to load. Detailed error:")
    traceback.print_exc()
    print()
    HAS_MEDIAPIPE = False

HAS_SCENEDETECT = False
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    HAS_SCENEDETECT = True
except ImportError:
    pass


# ── Content Modes ─────────────────────────────────────────────────────────────

MODES = {
    "default": {
        "label": "Default",
        "weight_motion": 0.40, "weight_cuts": 0.25,
        "weight_audio": 0.25,  "weight_face": 0.10,
        "score_drop": 0.35,    "score_engaging": 0.65,
        "low_motion": 0.15,    "low_audio": 0.10,
        "static_secs": 8.0,
    },
    "facecam": {
        "label": "Face / Talking Head",
        "weight_motion": 0.10, "weight_cuts": 0.05,
        "weight_audio": 0.50,  "weight_face": 0.35,
        "score_drop": 0.25,    "score_engaging": 0.55,
        "low_motion": 0.03,    "low_audio": 0.08,
        "static_secs": 30.0,
    },
    "podcast": {
        "label": "Podcast / Interview",
        "weight_motion": 0.05, "weight_cuts": 0.05,
        "weight_audio": 0.65,  "weight_face": 0.25,
        "score_drop": 0.20,    "score_engaging": 0.50,
        "low_motion": 0.02,    "low_audio": 0.12,
        "static_secs": 60.0,
    },
    "gaming": {
        "label": "Gaming",
        "weight_motion": 0.45, "weight_cuts": 0.10,
        "weight_audio": 0.35,  "weight_face": 0.10,
        "score_drop": 0.30,    "score_engaging": 0.60,
        "low_motion": 0.20,    "low_audio": 0.10,
        "static_secs": 15.0,
    },
    "vlog": {
        "label": "Vlog / Lifestyle",
        "weight_motion": 0.25, "weight_cuts": 0.20,
        "weight_audio": 0.35,  "weight_face": 0.20,
        "score_drop": 0.28,    "score_engaging": 0.58,
        "low_motion": 0.08,    "low_audio": 0.08,
        "static_secs": 12.0,
    },
    "tutorial": {
        "label": "Tutorial / Screenshare",
        "weight_motion": 0.15, "weight_cuts": 0.20,
        "weight_audio": 0.55,  "weight_face": 0.10,
        "score_drop": 0.22,    "score_engaging": 0.52,
        "low_motion": 0.02,    "low_audio": 0.10,
        "static_secs": 20.0,
    },
    "shortform": {
        "label": "Short-form / Reels / TikTok",
        "weight_motion": 0.35, "weight_cuts": 0.35,
        "weight_audio": 0.20,  "weight_face": 0.10,
        "score_drop": 0.40,    "score_engaging": 0.70,
        "low_motion": 0.20,    "low_audio": 0.08,
        "static_secs": 4.0,
    },
    "cinematic": {
        "label": "Cinematic / Film / Essay",
        "weight_motion": 0.15, "weight_cuts": 0.10,
        "weight_audio": 0.55,  "weight_face": 0.20,
        "score_drop": 0.18,    "score_engaging": 0.48,
        "low_motion": 0.02,    "low_audio": 0.08,
        "static_secs": 45.0,
    },
}
DEFAULT_MODE = "default"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Window:
    start:       float
    end:         float
    motion:      float
    cuts:        int
    audio:       float
    face:        bool
    face_conf:   float = 0.0
    speech_prob: float = 0.0
    score:       float = 0.0
    label:       str   = "neutral"


@dataclass
class Drop:
    start:       float
    end:         float
    reasons:     list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Result:
    video:      str
    duration:   float
    window_sec: float
    mode:       str          = DEFAULT_MODE
    windows:    list[Window] = field(default_factory=list)
    drops:      list[Drop]   = field(default_factory=list)
    summary:    dict         = field(default_factory=dict)


# ── Motion (Farneback Optical Flow) ──────────────────────────────────────────

class MotionExtractor:
    def __init__(self, path: str, window_sec: float):
        if cv2 is None:
            raise ImportError("opencv-python not installed")
        self.path = path
        self.window_sec = window_sec

    def extract(self) -> tuple[list[Window], float]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps
        win_frames   = int(self.window_sec * fps)

        windows, win_idx = [], 0
        while True:
            start_sec = win_idx * self.window_sec
            end_sec   = min(start_sec + self.window_sec, duration)
            if start_sec >= duration:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
            magnitudes, prev_gray = [], None
            step = max(1, win_frames // 15)

            for _ in range(0, win_frames, step):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    magnitudes.append(float(np.mean(mag)))
                prev_gray = gray

            raw    = float(np.mean(magnitudes)) if magnitudes else 0.0
            motion = float(np.clip(raw / 8.0, 0.0, 1.0))
            windows.append(Window(
                start=round(start_sec, 2), end=round(end_sec, 2),
                motion=round(motion, 4), cuts=0, audio=0.0, face=False,
            ))
            win_idx += 1

        cap.release()
        return windows, duration


# ── Scene cuts (PySceneDetect) ────────────────────────────────────────────────

def detect_cuts(video_path: str) -> list[float]:
    if not HAS_SCENEDETECT:
        return []
    try:
        video   = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video, show_progress=False)
        return [s[0].get_seconds() for s in manager.get_scene_list()[1:]]
    except Exception as e:
        print(f"  [warn] SceneDetect failed: {e}")
        return []

def assign_cuts(cut_times: list[float], windows: list[Window]) -> None:
    for w in windows:
        w.cuts = sum(1 for t in cut_times if w.start <= t < w.end)


# ── Audio (Librosa spectral) ──────────────────────────────────────────────────

class AudioExtractor:
    def __init__(self, path: str):
        self.path = path

    def enrich(self, windows: list[Window]) -> None:
        if librosa is None:
            print("  [warn] librosa not installed.")
            return

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        audio_source = self.path

        try:
            # Use ffmpeg to extract audio robustly into a temp wav file
            cmd = ["ffmpeg", "-y", "-i", self.path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", tmp_wav.name]
            try:
                kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "check": True}
                if sys.platform == "win32":
                    # This flag is specific to Windows and prevents the console window.
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                subprocess.run(cmd, **kwargs)
                audio_source = tmp_wav.name
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  [warn] ffmpeg not found/failed. Trying direct load (may fail for MP4).")

            y, sr = librosa.load(audio_source, sr=22050, mono=True)
        except Exception as e:
            print(f"  [warn] Audio load failed: {e}")
            print("  [info] TIP: Install ffmpeg and add it to your system PATH to fix audio.")
            if os.path.exists(tmp_wav.name):
                os.remove(tmp_wav.name)
            return

        if os.path.exists(tmp_wav.name):
            try:
                os.remove(tmp_wav.name)
            except Exception:
                pass

        if y.size == 0:
            print("  [warn] Audio array is empty.")
            return

        hop         = 512
        stft        = np.abs(librosa.stft(y, hop_length=hop))
        freqs       = librosa.fft_frequencies(sr=sr)
        mask        = (freqs >= 300) & (freqs <= 3400)
        energy      = np.mean(stft[mask, :] ** 2, axis=0)
        e_max       = energy.max()
        energy_norm = energy / (e_max + 1e-9) if e_max > 0 else energy

        zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        speech_p = np.clip(zcr / 0.25, 0, 1) * 0.4 + np.clip(centroid / 3000.0, 0, 1) * 0.6
        times    = librosa.frames_to_time(np.arange(len(energy_norm)), sr=sr, hop_length=hop)

        for w in windows:
            m = (times >= w.start) & (times < w.end)
            if m.any():
                w.audio       = round(float(energy_norm[m].mean()), 4)
                w.speech_prob = round(float(speech_p[m].mean()), 4)


# ── Face (MediaPipe FaceMesh) ─────────────────────────────────────────────────
# FaceMesh uses 468 landmarks instead of a bounding box detector.
# It's dramatically more robust to glasses, movement, angles, and partial occlusion
# because it's fitting a full 3D mesh to your face rather than looking for a face-shaped region.
# Confidence threshold is very low (0.1) because we just want to know IF a face is present,
# not how clean the detection is.

class FaceExtractor:
    def __init__(self, path: str, window_sec: float):
        self.path       = path
        self.window_sec = window_sec

    def enrich(self, windows: list[Window]) -> None:
        if HAS_MEDIAPIPE and _mp_face_mesh is not None:
            self._enrich_facemesh(windows)
        else:
            print("  [warn] MediaPipe not available, using Haar Cascade fallback")
            self._enrich_haar(windows)

    def _enrich_facemesh(self, windows: list[Window]) -> None:
        if cv2 is None:
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return

        fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        win_frames    = int(self.window_sec * fps)
        n_samples     = 20  # how many frames to sample per window

        with _mp_face_mesh.FaceMesh(
            static_image_mode=True,         # True because we seek across the video, no temporal continuity
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.1,   # very low — we just want to know if face exists
        ) as mesh:
            for w in windows:
                start_frame = int(w.start * fps)
                end_frame   = min(int(w.end * fps), total_frames - 1)
                actual_frames = max(end_frame - start_frame, 1)

                # Spread sample positions evenly across the full window duration.
                # Previously cap.read() was called sequentially, so all 20 samples
                # landed on the first ~0.7s of each window — the rest was never seen.
                step        = max(1, actual_frames // n_samples)
                sample_positions = range(start_frame, end_frame, step)

                detections  = 0
                total       = 0

                for frame_idx in sample_positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Guard against silent all-black frames that some codecs return
                    # on bad seeks — MediaPipe always returns None for these.
                    if frame is None or frame.size == 0 or frame.max() == 0:
                        continue

                    total += 1

                    # Downscale very large frames to 720p-wide for speed;
                    # upscale tiny frames so landmarks have enough pixel detail.
                    h, ww = frame.shape[:2]
                    if ww > 1280:
                        frame = cv2.resize(frame, (1280, int(h * 1280 / ww)))
                    elif ww < 640:
                        frame = cv2.resize(frame, (640, int(h * 640 / ww)))

                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = mesh.process(rgb)
                    if results.multi_face_landmarks:
                        detections += 1

                if total > 0 and detections > 0:
                    w.face      = True
                    # confidence = fraction of sampled frames where face was found
                    w.face_conf = round(detections / total, 4)
                else:
                    w.face      = False
                    w.face_conf = 0.0

        cap.release()

    def _enrich_haar(self, windows: list[Window]) -> None:
        if cv2 is None:
            return
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(xml):
            return
        det = cv2.CascadeClassifier(xml)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        win_frames = int(self.window_sec * fps)

        for w in windows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(w.start * fps))
            found, total = 0, 0
            step = max(1, win_frames // 10)
            for _ in range(0, win_frames, step):
                ret, frame = cap.read()
                if not ret:
                    break
                total += 1
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = det.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                if len(faces) > 0:
                    found += 1
            w.face      = found > 0
            w.face_conf = round(found / max(total, 1), 4)

        cap.release()


# ── Scoring ───────────────────────────────────────────────────────────────────

class Scorer:
    def __init__(self, mode: str = DEFAULT_MODE):
        cfg           = MODES.get(mode, MODES[DEFAULT_MODE])
        self.wm       = cfg["weight_motion"]
        self.wc       = cfg["weight_cuts"]
        self.wa       = cfg["weight_audio"]
        self.wf       = cfg["weight_face"]
        self.drop     = cfg["score_drop"]
        self.engaging = cfg["score_engaging"]

    def score(self, windows: list[Window]) -> None:
        max_cuts = max((w.cuts for w in windows), default=1) or 1
        for w in windows:
            cut_norm  = min(w.cuts / max_cuts, 1.0)
            face_val  = w.face_conf if w.face else 0.0
            audio_val = w.audio * 0.6 + w.speech_prob * 0.4
            s = (
                self.wm * w.motion  +
                self.wc * cut_norm  +
                self.wa * audio_val +
                self.wf * face_val
            )
            w.score = round(min(float(s), 1.0), 4)
            w.label = (
                "drop"     if w.score < self.drop     else
                "engaging" if w.score >= self.engaging else
                "neutral"
            )


# ── Drop explanation ──────────────────────────────────────────────────────────

class Explainer:
    def __init__(self, mode: str = DEFAULT_MODE):
        cfg              = MODES.get(mode, MODES[DEFAULT_MODE])
        self.low_motion  = cfg["low_motion"]
        self.low_audio   = cfg["low_audio"]
        self.static_secs = cfg["static_secs"]
        self.mode        = mode

    def explain(self, windows: list[Window], window_sec: float) -> list[Drop]:
        drops = []
        for i, w in enumerate(windows):
            if w.label != "drop":
                continue
            reasons, suggestions = [], []

            if w.motion < self.low_motion:
                reasons.append(f"Very low motion ({w.motion:.2f}) — screen appears static")
                if self.mode in ("facecam", "podcast", "cinematic"):
                    suggestions.append("Try gesturing more or shifting position slightly")
                else:
                    suggestions.append("Add a zoom, pan, or B-roll cut to break static shots")

            if w.cuts == 0:
                dry = 0.0
                for j in range(i, -1, -1):
                    if windows[j].cuts == 0:
                        dry += window_sec
                    else:
                        break
                if dry >= self.static_secs:
                    reasons.append(f"No scene cuts for ~{dry:.0f}s — pacing too slow")
                    if self.mode == "shortform":
                        suggestions.append("Add a cut every 2-3 seconds for short-form pacing")
                    elif self.mode in ("facecam", "podcast"):
                        suggestions.append("Consider a jump cut or B-roll overlay to break the shot")
                    else:
                        suggestions.append("Insert a cut or reaction clip every 4-6 seconds")

            if w.audio < self.low_audio and w.speech_prob < 0.25:
                reasons.append(f"Audio energy very low ({w.audio:.2f}) and no speech detected")
                if self.mode == "podcast":
                    suggestions.append("Dead air in podcasts kills retention — check mic or fill silence")
                elif self.mode == "gaming":
                    suggestions.append("Keep commentary going — silence during gaming loses viewers fast")
                else:
                    suggestions.append("Add background music, voice presence, or sound effects")
            elif w.audio < self.low_audio:
                reasons.append(f"Audio energy low ({w.audio:.2f}) — content sounds quiet")
                suggestions.append("Raise your voice level or add music under the section")

            if not w.face:
                if self.mode in ("facecam", "podcast", "vlog"):
                    reasons.append("No face detected — human connection drops significantly")
                    suggestions.append("Make sure your face is clearly in frame and well-lit")
                elif self.mode not in ("tutorial", "gaming", "cinematic"):
                    if w.audio < 0.30:
                        reasons.append("No face and low audio — very low human connection")
                        suggestions.append("Show your face or add text overlays to maintain presence")
            elif w.face_conf < 0.4:
                reasons.append(f"Face only visible {w.face_conf:.0%} of the time in this window")
                suggestions.append("Try to keep your face consistently in frame")

            if not reasons:
                reasons.append(f"All signals weak (score {w.score:.2f}) — general energy drop")
                suggestions.append("Review pacing — add cuts, music, or motion to lift energy")

            drops.append(Drop(start=w.start, end=w.end, reasons=reasons, suggestions=suggestions))
        return drops


# ── Summary ───────────────────────────────────────────────────────────────────

def make_summary(windows: list[Window], drops: list[Drop]) -> dict:
    scores = [w.score for w in windows]
    return {
        "total_windows":  len(windows),
        "drop_count":     len(drops),
        "engaging_count": sum(1 for w in windows if w.label == "engaging"),
        "neutral_count":  sum(1 for w in windows if w.label == "neutral"),
        "avg_score":      round(float(np.mean(scores)), 4)  if scores else 0.0,
        "min_score":      round(float(np.min(scores)), 4)   if scores else 0.0,
        "max_score":      round(float(np.max(scores)), 4)   if scores else 0.0,
        "avg_motion":     round(float(np.mean([w.motion for w in windows])), 4),
        "avg_audio":      round(float(np.mean([w.audio  for w in windows])), 4),
        "avg_face_conf":  round(float(np.mean([w.face_conf for w in windows])), 4),
        "total_cuts":     sum(w.cuts for w in windows),
        "face_pct":       round(sum(1 for w in windows if w.face) / max(len(windows), 1), 4),
        "drop_time_pct":  round(len(drops) / max(len(windows), 1), 4),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def analyze(
    video_path: str,
    window_sec: float = 5.0,
    mode: str = DEFAULT_MODE,
    out_json: Optional[str] = None,
    verbose: bool = True,
) -> Result:
    def log(msg):
        if verbose:
            print(msg)

    log(f"\n  Video : {video_path}")
    log(f"  Window: {window_sec}s | Mode: {MODES.get(mode, MODES[DEFAULT_MODE])['label']}")
    log(f"  MediaPipe  : {'FaceMesh ready' if HAS_MEDIAPIPE else 'NOT installed — pip install mediapipe'}")
    log(f"  SceneDetect: {'ready' if HAS_SCENEDETECT else 'NOT installed — pip install scenedetect'}\n")

    log("[1/6] Extracting motion (Farneback Optical Flow)...")
    windows, duration = MotionExtractor(video_path, window_sec).extract()
    log(f"       {len(windows)} windows, {duration:.1f}s total")

    log("[2/6] Detecting cuts (PySceneDetect)...")
    cuts = detect_cuts(video_path)
    assign_cuts(cuts, windows)
    log(f"       {len(cuts)} cuts found")

    log("[3/6] Analyzing audio (spectral speech energy)...")
    AudioExtractor(video_path).enrich(windows)

    log("[4/6] Detecting faces (MediaPipe FaceMesh)...")
    FaceExtractor(video_path, window_sec).enrich(windows)
    face_count = sum(1 for w in windows if w.face)
    log(f"       Face found in {face_count}/{len(windows)} windows")

    log("[5/6] Scoring...")
    Scorer(mode).score(windows)

    log("[6/6] Explaining drops...")
    drops   = Explainer(mode).explain(windows, window_sec)
    summary = make_summary(windows, drops)

    result = Result(
        video=video_path, duration=round(duration, 2),
        window_sec=window_sec, mode=mode,
        windows=windows, drops=drops, summary=summary,
    )

    if out_json:
        with open(out_json, "w") as f:
            json.dump(asdict(result), f, indent=2)
        log(f"  Saved: {out_json}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--window", "-w", type=float, default=5.0)
    parser.add_argument("--mode",   "-m", default=DEFAULT_MODE, choices=list(MODES.keys()))
    parser.add_argument("--out",    "-o", default="result.json")
    parser.add_argument("--chart",  "-c", action="store_true")
    args = parser.parse_args()

    r = analyze(args.video, window_sec=args.window, mode=args.mode, out_json=args.out)
    s = r.summary
    print(f"\n  Duration : {r.duration:.1f}s | Mode: {MODES.get(r.mode,{}).get('label',r.mode)}")
    print(f"  Avg score: {s['avg_score']:.0%} | Drops: {s['drop_count']} | Engaging: {s['engaging_count']}")
    print(f"  Face detected: {s['face_pct']:.0%} of windows | Avg confidence: {s['avg_face_conf']:.0%}")
    for d in r.drops:
        print(f"\n  {d.start:.1f}s - {d.end:.1f}s")
        for x in d.reasons:    print(f"    WHY {x}")
        for x in d.suggestions: print(f"    FIX {x}")

    if args.chart:
        from visualizer import plot
        plot(r, save_path="attention_chart.png", show=True)
