from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_service import UPLOAD_DIR, UPLOAD_LOUDNESS_FILTER, normalize_upload  # noqa: E402


class AudioServiceTest(unittest.TestCase):
    def test_normalize_upload_uses_stable_loudness_filter(self) -> None:
        input_path = UPLOAD_DIR / "audio_filter_input.wav"
        output_path = UPLOAD_DIR / "audio_filter_output.wav"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"input")
        commands: list[list[str]] = []

        def fake_run(cmd, **_kwargs):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b"filtered")
            return subprocess.CompletedProcess(cmd, 0)

        try:
            with patch("app.services.audio_service._ffmpeg_executable", return_value="ffmpeg"):
                with patch("app.services.audio_service.subprocess.run", side_effect=fake_run):
                    result = normalize_upload(input_path, "audio_filter_output")
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.assertEqual(result, output_path)
        self.assertEqual(commands[0][commands[0].index("-filter:a") + 1], UPLOAD_LOUDNESS_FILTER)
        self.assertEqual(commands[0][commands[0].index("-ac") + 1], "1")
        self.assertEqual(commands[0][commands[0].index("-ar") + 1], "48000")

    def test_normalize_upload_copies_input_when_filter_fails(self) -> None:
        input_path = UPLOAD_DIR / "audio_filter_fallback_input.wav"
        output_path = UPLOAD_DIR / "audio_filter_fallback_output.wav"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"source-audio")

        try:
            with patch("app.services.audio_service._ffmpeg_executable", return_value="ffmpeg"):
                with patch(
                    "app.services.audio_service.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
                ):
                    result = normalize_upload(input_path, "audio_filter_fallback_output")
                    self.assertEqual(result.read_bytes(), b"source-audio")
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
