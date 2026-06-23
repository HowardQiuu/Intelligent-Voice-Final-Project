from __future__ import annotations

import importlib
import os
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import separation_service  # noqa: E402
from app.services.separation_service import separate_demo_audio, separate_uploaded_audio  # noqa: E402


class FakeSpeakerAudio:
    shape = (8000,)

    def unsqueeze(self, dim):
        return self

    def to(self, dtype=None):
        return self


class FakeSources:
    shape = (1, 8000, 2)

    def detach(self):
        return self

    def cpu(self):
        return self

    def __getitem__(self, item):
        return FakeSpeakerAudio()


class FakeSeparator:
    @classmethod
    def from_hparams(cls, source, savedir, run_opts, **kwargs):
        return cls()

    def separate_file(self, path):
        return FakeSources()


class FakeSeparationModule:
    SepformerSeparation = FakeSeparator


class FakeLocalStrategy:
    COPY = "copy"


class FakeFetchingModule:
    LocalStrategy = FakeLocalStrategy


class FakeTorch:
    float32 = "float32"


class FakeTorchaudio:
    saved_paths: list[str] = []

    @staticmethod
    def save(path, audio, sample_rate):
        FakeTorchaudio.saved_paths.append(path)
        Path(path).write_bytes(b"fake wav bytes")


def fake_import_module(name: str):
    if name == "speechbrain.inference.separation":
        return FakeSeparationModule
    if name == "speechbrain.utils.fetching":
        return FakeFetchingModule
    if name == "torch":
        return FakeTorch
    if name == "torchaudio":
        return FakeTorchaudio
    return importlib.import_module(name)


class SeparationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTorchaudio.saved_paths = []
        separation_service._SEPARATOR_CACHE.clear()
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        for path in FakeTorchaudio.saved_paths:
            Path(path).unlink(missing_ok=True)

    def test_placeholder_demo_separation_returns_track_contract(self) -> None:
        result = separate_demo_audio("overlap_meeting", "/static/audio/overlap_meeting_enhanced.wav")

        self.assertEqual(result["method"], "Placeholder fallback")
        self.assertEqual(result["status"], "placeholder")
        self.assertEqual(result["track_count"], "1")
        self.assertEqual(result["tracks"][0]["audio_url"], "/static/audio/overlap_meeting_enhanced.wav")

    def test_upload_placeholder_separation_returns_track_contract(self) -> None:
        result = separate_uploaded_audio("/static/uploads/sample.wav")

        self.assertEqual(result["method"], "Placeholder fallback")
        self.assertEqual(result["status"], "placeholder")
        self.assertEqual(result["tracks"][0]["audio_url"], "/static/uploads/sample.wav")

    def test_speechbrain_mock_returns_two_tracks(self) -> None:
        source_path = separation_service.UPLOAD_DIR / "mock_enhanced.wav"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"fake source")
        try:
            with patch.dict(os.environ, {"SEPARATION_BACKEND": "speechbrain"}, clear=True):
                with patch("app.services.separation_service.importlib.import_module", side_effect=fake_import_module):
                    result = separate_uploaded_audio(f"/static/uploads/{source_path.name}")
        finally:
            source_path.unlink(missing_ok=True)

        self.assertIn("SpeechBrain SepFormer", result["method"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["track_count"], "2")
        self.assertEqual(len(result["tracks"]), 2)
        self.assertTrue(result["tracks"][0]["audio_url"].startswith("/static/uploads/"))

    def test_speechbrain_import_failure_falls_back(self) -> None:
        with patch.dict(os.environ, {"SEPARATION_BACKEND": "speechbrain"}, clear=True):
            with patch("app.services.separation_service.importlib.import_module", side_effect=ImportError):
                result = separate_demo_audio("noisy_meeting", "/static/audio/noisy_meeting_enhanced.wav")

        self.assertEqual(result["method"], "Placeholder fallback")
        self.assertIn("SpeechBrain failed", result["status"])

    def test_long_audio_skips_speechbrain_before_import(self) -> None:
        source_path = separation_service.UPLOAD_DIR / "long_meeting.wav"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(source_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 8000 * 120)

        try:
            with patch.dict(
                os.environ,
                {"SEPARATION_BACKEND": "speechbrain", "SEPARATION_MAX_SECONDS": "60"},
                clear=True,
            ):
                with patch("app.services.separation_service.importlib.import_module") as import_mock:
                    result = separate_uploaded_audio(f"/static/uploads/{source_path.name}")
        finally:
            source_path.unlink(missing_ok=True)

        import_mock.assert_not_called()
        self.assertEqual(result["method"], "Placeholder fallback")
        self.assertIn("Audio too long", result["status"])
        self.assertEqual(result["track_count"], "1")


if __name__ == "__main__":
    unittest.main()
