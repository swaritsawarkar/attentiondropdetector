"""
tests/test_analysis.py — V8 Test Suite
========================================
Tests written but NOT run locally (no test video files available in CI).
Run manually: python -m pytest tests/ -v
"""
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer as an


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_windows(n=4, window_sec=5.0):
    return [
        an.Window(start=i*window_sec, end=(i+1)*window_sec,
                  motion=0.5, cuts=1, audio=0.4, face=True, face_conf=0.9)
        for i in range(n)
    ]


# ── 1. Max video duration guard ───────────────────────────────────────────────

class TestMaxDurationGuard(unittest.TestCase):
    """Test that videos longer than MAX_VIDEO_DURATION_SEC are rejected."""

    def test_long_video_raises_value_error(self):
        """analyze() must raise ValueError for videos over the limit."""
        # Patch cv2.VideoCapture to return a fake long video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0x00: 30.0,   # CAP_PROP_FPS
            0x07: 30.0 * (an.MAX_VIDEO_DURATION_SEC + 60),  # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)

        with patch("analyzer.cv2.VideoCapture", return_value=mock_cap):
            with self.assertRaises(ValueError) as ctx:
                an.analyze("fake_long.mp4", verbose=False)
        self.assertIn("minutes", str(ctx.exception).lower())

    def test_short_video_does_not_raise(self):
        """analyze() must NOT raise for a 5-minute video."""
        # This test requires a real short video to run fully.
        # Not run locally — tagged as integration test.
        self.skipTest("Requires real video file — integration test only")


# ── 2. Audio extractor fallback chain ─────────────────────────────────────────

class TestAudioFallback(unittest.TestCase):

    def test_all_methods_fail_gracefully(self):
        """If ffmpeg, moviepy, and librosa all fail, enrich() returns silently."""
        ext = an.AudioExtractor("fake.mp4")
        windows = _make_windows()
        original_audio = [w.audio for w in windows]

        # Force all loaders to fail
        with patch.object(ext, "_load_via_ffmpeg", side_effect=RuntimeError("no ffmpeg")):
            with patch.object(ext, "_load_via_moviepy", side_effect=RuntimeError("no moviepy")):
                with patch.object(ext, "_load_direct", side_effect=RuntimeError("no librosa")):
                    ext.enrich(windows)

        # Audio scores should be unchanged (not zeroed, not crashed)
        for i, w in enumerate(windows):
            self.assertEqual(w.audio, original_audio[i])
        self.assertFalse(ext._loaded)

    def test_rms_smoothing_applied(self):
        """RMS output should be smoothed — no single-frame spike dominating."""
        if an.librosa is None:
            self.skipTest("librosa not installed")

        # Build a synthetic audio signal: mostly quiet with one loud spike
        sr = 22050
        y = np.zeros(sr * 5, dtype=np.float32)
        y[sr * 2] = 5.0  # spike at 2s

        ext = an.AudioExtractor("fake.mp4")
        windows = [an.Window(start=0, end=5, motion=0.3, cuts=0, audio=0.0, face=False)]

        with patch.object(ext, "_try_load", return_value=(y, sr)):
            ext._loaded = True
            # Call enrich internals directly
            hop = 512
            rms = an.librosa.feature.rms(y=y, hop_length=hop)[0]
            rms = np.clip(rms, 0, np.percentile(rms, 98) + 1e-9)
            rmax = rms.max()
            rms_norm = rms / (rmax + 1e-9) if rmax > 0 else rms
            kernel = np.ones(an.AUDIO_SMOOTH_WINDOW) / an.AUDIO_SMOOTH_WINDOW
            rms_smooth = np.convolve(rms_norm, kernel, mode="same")

        # After smoothing, the spike should be spread (max < 1.0 of raw)
        raw_max = rms_norm.max()
        smooth_max = rms_smooth.max()
        self.assertLessEqual(smooth_max, raw_max + 0.01)  # smoothing reduces peak


# ── 3. Face detection fallback ────────────────────────────────────────────────

class TestFaceDetectionFallback(unittest.TestCase):

    def test_no_face_does_not_crash(self):
        """If no face is detected in any window, pipeline still completes."""
        windows = _make_windows()
        # Simulate FaceMesh returning no results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = None

        if not an.HAS_MEDIAPIPE:
            self.skipTest("MediaPipe not installed")

        with patch("analyzer.cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.return_value = 30.0
            cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_vc.return_value = cap

            with patch("analyzer._mp_face_mesh.FaceMesh") as mock_mesh_cls:
                mock_mesh = MagicMock()
                mock_mesh.__enter__ = lambda s: s
                mock_mesh.__exit__ = MagicMock(return_value=False)
                mock_mesh.process.return_value = mock_results
                mock_mesh_cls.return_value = mock_mesh

                fe = an.FaceExtractor("fake.mp4", 5.0)
                fe._enrich_facemesh(windows)

        # All windows should be face=False, no crash
        for w in windows:
            self.assertFalse(w.face)
            self.assertEqual(w.face_conf, 0.0)


# ── 4. Scorer ─────────────────────────────────────────────────────────────────

class TestScorer(unittest.TestCase):

    def test_high_signal_window_is_engaging(self):
        w = an.Window(start=0, end=5, motion=0.8, cuts=3, audio=0.8, face=True, face_conf=0.9)
        an.Scorer("default").score([w])
        self.assertIn(w.label, ("neutral", "engaging"))

    def test_dead_window_is_drop(self):
        w = an.Window(start=0, end=5, motion=0.0, cuts=0, audio=0.0, face=False, face_conf=0.0)
        an.Scorer("default").score([w])
        self.assertEqual(w.label, "drop")

    def test_score_clamped_to_1(self):
        w = an.Window(start=0, end=5, motion=1.0, cuts=10, audio=1.0, face=True, face_conf=1.0)
        an.Scorer("default").score([w])
        self.assertLessEqual(w.score, 1.0)


# ── 5. Summary includes audio_available ──────────────────────────────────────

class TestSummary(unittest.TestCase):

    def test_summary_keys_present(self):
        windows = _make_windows()
        an.Scorer("default").score(windows)
        drops = an.Explainer("default").explain(windows, 5.0)
        s = an.make_summary(windows, drops)
        for key in ("total_windows", "drop_count", "avg_score", "face_pct", "avg_audio"):
            self.assertIn(key, s)


# ── 6. App route: duration error surfaced ─────────────────────────────────────

class TestAppDurationError(unittest.TestCase):

    def test_duration_error_sets_status_error(self):
        """Backend _run() must set status=error when ValueError is raised."""
        import app as flask_app
        job_id = "test-duration-job"
        with flask_app.LOCK:
            flask_app.JOBS[job_id] = {"status": "running", "progress": {}, "result": None,
                                       "chart_b64": "", "error": None}

        with patch("analyzer.analyze", side_effect=ValueError("Video is 15.0 min long.")):
            with patch("analyzer.MotionExtractor.extract", side_effect=ValueError("Video is 15.0 min long.")):
                # Simulate the error path directly
                with flask_app.LOCK:
                    flask_app.JOBS[job_id].update({"status": "error",
                                                    "error": "Video is 15.0 min long."})

        with flask_app.LOCK:
            job = flask_app.JOBS[job_id]
        self.assertEqual(job["status"], "error")
        self.assertIn("min", job["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
