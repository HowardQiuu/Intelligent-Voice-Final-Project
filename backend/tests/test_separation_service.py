from __future__ import annotations

import importlib
import os
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

REAL_IMPORT_MODULE = importlib.import_module


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
    return REAL_IMPORT_MODULE(name)


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
                with patch.object(separation_service.importlib, "import_module", side_effect=fake_import_module):
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
            with patch.object(separation_service.importlib, "import_module", side_effect=ImportError):
                result = separate_demo_audio("noisy_meeting", "/static/audio/noisy_meeting_enhanced.wav")

        self.assertEqual(result["method"], "Placeholder fallback")
        self.assertIn("SpeechBrain failed", result["status"])

    def test_long_audio_uses_chunked_speechbrain(self) -> None:
        source_path = separation_service.UPLOAD_DIR / "long_meeting.wav"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(source_path, seconds=120)

        def fake_concat(chunk_paths: list[Path], output_path: Path) -> None:
            output_path.write_bytes(b"joined wav bytes")

        try:
            with patch.dict(
                os.environ,
                {
                    "SEPARATION_BACKEND": "speechbrain",
                    "SEPARATION_MAX_SECONDS": "60",
                    "SEPARATION_CHUNK_SECONDS": "60",
                },
                clear=True,
            ):
                with patch.object(separation_service.importlib, "import_module", side_effect=fake_import_module):
                    with patch(
                        "app.services.separation_service._split_audio_to_chunks",
                        return_value=[source_path, source_path],
                    ):
                        with patch("app.services.separation_service._concat_audio_chunks", side_effect=fake_concat):
                            result = separate_uploaded_audio(f"/static/uploads/{source_path.name}")
        finally:
            source_path.unlink(missing_ok=True)
            for path in separation_service.UPLOAD_DIR.glob("upload_*_speaker_*_chunked.wav"):
                path.unlink(missing_ok=True)

        self.assertIn("SpeechBrain SepFormer chunked", result["method"])
        self.assertEqual(result["status"], "ok-chunked")
        self.assertEqual(result["track_count"], "2")
        self.assertEqual(len(result["tracks"]), 2)


def _write_wav(path: Path, seconds: int) -> None:
    sample_rate = 8000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * sample_rate * seconds)


if __name__ == "__main__":
    unittest.main()
