from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.separation_service import separate_demo_audio, separate_uploaded_audio  # noqa: E402


class SeparationServiceTest(unittest.TestCase):
    def test_demo_separation_returns_track_contract(self) -> None:
        result = separate_demo_audio("overlap_meeting", "/static/audio/overlap_meeting_enhanced.wav")

        self.assertEqual(result["track_count"], "1")
        self.assertEqual(result["tracks"][0]["audio_url"], "/static/audio/overlap_meeting_enhanced.wav")
        self.assertIn("track_id", result["tracks"][0])
        self.assertIn("label", result["tracks"][0])
        self.assertIn("description", result["tracks"][0])

    def test_upload_separation_returns_track_contract(self) -> None:
        result = separate_uploaded_audio("/static/uploads/sample.wav")

        self.assertEqual(result["method"], "Upload separation placeholder")
        self.assertEqual(result["tracks"][0]["audio_url"], "/static/uploads/sample.wav")


if __name__ == "__main__":
    unittest.main()
