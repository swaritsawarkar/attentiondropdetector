# Attention Drop Detector

A Python tool that analyzes a video file and predicts where viewer attention is likely to drop — before publishing.

Unlike platform analytics (which only show data *after* upload), this runs on raw video files and gives you actionable feedback immediately.

---

## What it does

1. Breaks the video into time windows (default: 5 seconds each)
2. Extracts four signals per window: **motion intensity**, **scene cut frequency**, **audio energy**, and **face presence**
3. Combines them into a weighted **attention score** (0–1) for each window
4. Labels each window: `engaging`, `neutral`, or `drop`
5. For each drop, explains **why** and suggests **specific fixes**
6. Extracts the best segments as **highlight clips**
7. Generates a full visualization chart

---

## Files

| File | What it does |
|---|---|
| `analyzer.py` | Core pipeline — feature extraction, scoring, drop explanation |
| `visualizer.py` | Generates the attention score chart (3-panel PNG) |
| `highlights.py` | Finds the most engaging segments for clips / Shorts |
| `demo.py` | Runs the full pipeline on synthetic data — no video needed |
| `requirements.txt` | Python dependencies |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** OpenCV, NumPy, librosa, matplotlib

---

## Usage

### Run on a real video
```bash
python analyzer.py myvideo.mp4
```

With options:
```bash
python analyzer.py myvideo.mp4 --window 8 --out results.json --chart
```

| Flag | Default | Description |
|---|---|---|
| `--window` / `-w` | `5.0` | Window size in seconds |
| `--out` / `-o` | `result.json` | Where to save the JSON output |
| `--chart` / `-c` | off | Generate the chart after analysis |

### Run the demo (no video needed)
```bash
python demo.py
```

This simulates a 5-minute video with realistic attention patterns (strong opening, mid-video lull, recovery, second drop near the end) and runs the full pipeline.

### Generate chart from existing results
```bash
python visualizer.py result.json --show
```

### Extract highlight segments
```bash
python highlights.py result.json --top 5 --min-len 15
```

---

## How the score is calculated

```
attention_score = 0.40 × motion
               + 0.25 × cut_frequency
               + 0.25 × audio_energy
               + 0.10 × face_presence
```

All signals are normalized to 0–1 before combining.

| Score | Label |
|---|---|
| ≥ 0.65 | Engaging |
| 0.35 – 0.65 | Neutral |
| < 0.35 | Drop |

---

## Example output

```
── Summary ──────────────────────────────────────────────
  Duration         : 300s
  Windows          : 60
  Avg score        : 43%
  Engaging windows : 13
  Drop windows     : 22  (37% of video)
  Total cuts       : 112

── Drop Zones (22) ──────────────────────────────────────

  105s – 110s
    WHY  Very low motion (0.09) — screen appears static
    WHY  Audio energy very low (0.04) — content sounds dead
    FIX  Add a zoom, pan, or B-roll cut to break static shots
    FIX  Add background music or increase voice presence

  115s – 120s
    WHY  No scene cuts for ~15s — pacing too slow
    FIX  Insert a cut or reaction clip every 4–6 seconds
```

---

## Output JSON format

`analyzer.py` saves a `result.json` file with:

```json
{
  "video": "myvideo.mp4",
  "duration": 300.0,
  "window_sec": 5.0,
  "windows": [
    {
      "start": 0.0,
      "end": 5.0,
      "motion": 0.72,
      "cuts": 3,
      "audio": 0.81,
      "face": true,
      "score": 0.74,
      "label": "engaging"
    }
  ],
  "drops": [
    {
      "start": 105.0,
      "end": 110.0,
      "reasons": ["Very low motion (0.09) ..."],
      "suggestions": ["Add a zoom, pan ..."]
    }
  ],
  "summary": { ... }
}
```

---

## Limitations

- Attention scores are **heuristic** — they are not validated against real platform retention data. They use editing signals as a proxy for engagement.
- Audio extraction requires the video to have an audio track.
- Face detection uses a basic Haar cascade — not robust to unusual camera angles.
- Very long videos (60+ min) may take several minutes to process.

---

## Future work

- Train a regression model on real YouTube retention curves to replace/calibrate the heuristic weights
- Subtitle / text overlay density as an additional signal
- Real-time feedback mode while editing in a timeline
- Browser-based drag-and-drop interface
- Compare multiple videos to track improvement over time

---

## Why this project

Most creator tools either require you to upload first (platform analytics) or are black-box AI systems that tell you *where* attention drops but not *why*. This tool uses editing signals — the same things a video editor thinks about — as measurable features, giving you an explainable and actionable output.

The goal is not a perfect ML model. It is: extract meaningful signals → present clear insights → help creators improve content.
