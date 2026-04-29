# Changelog

All notable changes to Attention Drop Detector are documented here.

---

## [v8] — 2026-04-29

### Fixed
- **Audio stability**: Replaced raw STFT spectral energy with RMS energy + moving-average smoothing (window=5). Audio scores no longer spike wildly.
- **Audio loading**: Replaced single-path ffmpeg load with a 3-tier fallback chain: ffmpeg → moviepy → librosa direct. If all fail, analysis continues without audio and the UI shows a clear warning.
- **Chart sizing**: Charts are no longer huge. Reduced from 14×9@140dpi to 12×7@100dpi. Added `max-height: 480px` in CSS so charts can never overflow the UI.
- **Max video limit**: Videos longer than 10 minutes are now rejected before processing with a clear error message. Limit is configurable via `MAX_VIDEO_DURATION_SEC` in `analyzer.py`.

### Improved
- **UI — Result cards**: Added `Face Conf` and `Audio` status cards. Cards now color-code based on value (green/yellow/red).
- **UI — Timeline**: Hover tooltips now show `start–end · score% · label` instead of just timestamps.
- **UI — Progress bar**: Smoother cubic-bezier animation. Step labels updated to match actual operations.
- **UI — Audio warning**: If audio could not be loaded, a yellow warning banner appears above the result cards.
- **UI — Drop zone text**: Now shows "max 10 minutes" instead of "up to 2 GB".
- **Codebase**: Added `version.txt`, `build_exe.bat`, `run_dev.bat`, expanded `.gitignore`.
- **Backend**: Chart generation errors are now caught gracefully — analysis still completes even if matplotlib fails.
- **Debug logging**: Audio pipeline now logs method used, sample rate, duration, RMS range, and smoothing window to `debug_log.txt`.

### Removed
- **Scrutinization length option**: Was already absent from UI. Confirmed removed from backend.
- `result.json`: Generated output file, not source code — removed from repo and added to `.gitignore`.

### Archived
- `FULL_PROJECT_JOURNEY.txt` → moved to `/archive` (historical doc, not needed for runtime)
- `PROJECT_RETROSPECTIVE.txt` → moved to `/archive` (historical doc, not needed for runtime)

---

## [v7] — 2026-04-28

- Fixed face detection frame sampling bug (windows were only sampling from their first ~0.7s).
- Added all-black frame guard for bad codec seeks.
- Added downscaling for frames wider than 1280px.

---

## [v6] — 2026-04-27

- Replaced FaceDetection with FaceMesh (468 landmarks).
- Rewrote audio pipeline: ffmpeg → Librosa spectral analysis (300–3400 Hz).
- Added content mode system with per-mode scoring weights.
- New Flask web UI with drag-and-drop, mode picker, and chart.
- PyInstaller EXE build working.

---

## [v5] — 2026-04-25

- Fixed Python 3.13 incompatibility (downgraded to Python 3.12 + venv).
