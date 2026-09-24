import io
import sys
import types
import unittest

fake_analyzer = types.ModuleType("analyzer")
fake_analyzer.MODES = {"default": object()}
fake_analyzer.analyze = lambda *args, **kwargs: None
sys.modules.setdefault("analyzer", fake_analyzer)

fake_visualizer = types.ModuleType("visualizer")
fake_visualizer.plot = lambda *args, **kwargs: None
sys.modules.setdefault("visualizer", fake_visualizer)

import app as app_module


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_analyze_requires_video(self):
        response = self.client.post("/analyze")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No video file uploaded")

    def test_analyze_requires_selected_video_filename(self):
        response = self.client.post(
            "/analyze",
            data={"video": (io.BytesIO(b"video"), "")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No video file selected")

    def test_analyze_rejects_invalid_window(self):
        response = self.client.post(
            "/analyze",
            data={"video": (io.BytesIO(b"video"), "clip.mp4"), "window": "not-a-number"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Window duration must be a positive number")

    def test_analyze_rejects_non_positive_or_non_finite_window(self):
        for window in ("0", "-1", "nan", "inf"):
            with self.subTest(window=window):
                response = self.client.post(
                    "/analyze",
                    data={"video": (io.BytesIO(b"video"), "clip.mp4"), "window": window},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"], "Window duration must be a positive number")

    def test_status_rejects_unknown_job(self):
        response = self.client.get("/status/not-a-real-job")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "Unknown job")


if __name__ == "__main__":
    unittest.main()
