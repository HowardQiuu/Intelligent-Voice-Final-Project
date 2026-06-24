from __future__ import annotations

import os
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import enhancement_service  # noqa: E402
from app.services.audio_service import UPLOAD_DIR  # noqa: E402


class EnhancementServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()

    def test_long_upload_uses_chunked_deepfilternet(self) -> None:
        path = UPLOAD_DIR / "long_upload.wav"
        enhanced = UPLOAD_DIR / "long_upload_deepfilter_chunked.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 8000 * 120)

        try:
            with patch.dict(os.environ, {"ENHANCEMENT_MAX_SECONDS": "60"}, clear=True):
                with patch(
                    "app.services.enhancement_service.denoise_audio_in_chunks",
                    return_value=(enhanced, "DeepFilterNet chunked denoise (2 chunks x 60s)"),
                ) as chunked_mock:
                    result = enhancement_service.enhance_uploaded_audio(path)
        finally:
            path.unlink(missing_ok=True)

        chunked_mock.assert_called_once()
        self.assertEqual(result["original_audio_url"], "/static/uploads/long_upload.wav")
        self.assertEqual(result["enhanced_audio_url"], "/static/uploads/long_upload_deepfilter_chunked.wav")
        self.assertIn("DeepFilterNet chunked denoise", result["method"])

    def test_should_skip_enhancement_is_false_for_long_audio(self) -> None:
        path = UPLOAD_DIR / "long_upload_skip_guard.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 8000 * 120)

        try:
            with patch.dict(os.environ, {"ENHANCEMENT_MAX_SECONDS": "60"}, clear=True):
                self.assertFalse(enhancement_service.should_skip_enhancement(path))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
