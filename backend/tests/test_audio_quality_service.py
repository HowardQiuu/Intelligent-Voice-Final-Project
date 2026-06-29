from __future__ import annotations

import math
import os
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_quality_service import analyze_audio_quality, apply_audibility_pregain  # noqa: E402
from app.services.audio_service import UPLOAD_DIR, has_ffmpeg  # noqa: E402


class AudioQualityServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()

    @unittest.skipUnless(has_ffmpeg(), "ffmpeg is required for real pregain verification")
    def test_low_volume_audio_pregain_increases_rms_without_clipping(self) -> None:
        source = UPLOAD_DIR / "low_volume_quality.wav"
        output = UPLOAD_DIR / "low_volume_quality_pregain.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        _write_sine(source, amplitude=0.003)

        try:
            before = analyze_audio_quality(source)
            with patch.dict(os.environ, {"ENHANCEMENT_MAX_GAIN_DB": "24", "ENHANCEMENT_TARGET_LUFS": "-18"}, clear=True):
                output_path, status, metrics = apply_audibility_pregain(source)
            after = analyze_audio_quality(output_path)
        finally:
            source.unlink(missing_ok=True)
            output.unlink(missing_ok=True)

        self.assertEqual(status, "ok")
        self.assertGreater(after.rms_dbfs, before.rms_dbfs + 8.0)
        self.assertLess(after.clipping_ratio, 0.01)
        self.assertEqual(metrics["quality_pregain_status"], "ok")

    def test_normal_volume_audio_skips_pregain(self) -> None:
        source = UPLOAD_DIR / "normal_volume_quality.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        _write_sine(source, amplitude=0.2)

        try:
            output_path, status, _metrics = apply_audibility_pregain(source)
        finally:
            source.unlink(missing_ok=True)

        self.assertEqual(output_path, source)
        self.assertEqual(status, "skipped")


def _write_sine(path: Path, amplitude: float) -> None:
    sample_rate = 16000
    seconds = 1
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_rate * seconds):
            value = amplitude * math.sin(2 * math.pi * 440 * index / sample_rate)
            frames.extend(int(value * 32767).to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()
