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
from app.services.separation_service import separate_demo_audio, separate_uploaded_audio, separate_with_quality_router  # noqa: E402


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

    def test_gated_speaker_tracks_use_extreme_target_and_background_gain(self) -> None:
        numpy = REAL_IMPORT_MODULE("numpy")
        soundfile = REAL_IMPORT_MODULE("soundfile")
        source_path = separation_service.UPLOAD_DIR / "gated_gain_source.wav"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        samples = numpy.full((1000, 1), 0.2, dtype="float32")
        soundfile.write(str(source_path), samples, 1000)

        try:
            with patch.dict(
                os.environ,
                {
                    "SPEAKER_TRACK_BACKGROUND_GAIN": "0.01",
                    "SPEAKER_TRACK_TARGET_GAIN": "1.5",
                    "SPEAKER_TRACK_FADE_MS": "0",
                },
                clear=True,
            ):
                tracks = separation_service._write_gated_speaker_tracks(
                    source_path,
                    {"speaker A": [(0.2, 0.4)]},
                )
                output_path = separation_service.resolve_static_url(tracks[0]["audio_url"])
                output, _sample_rate = soundfile.read(str(output_path), always_2d=True, dtype="float32")
        finally:
            source_path.unlink(missing_ok=True)
            for path in separation_service.UPLOAD_DIR.glob("gated_gain_source_*_diarized.wav"):
                path.unlink(missing_ok=True)

        self.assertAlmostEqual(float(output[100, 0]), 0.002, places=3)
        self.assertAlmostEqual(float(output[250, 0]), 0.3, places=3)
        self.assertIn("强衰减", tracks[0]["description"])

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

    def test_quality_router_selects_best_scored_separation_candidate(self) -> None:
        def fake_candidate(candidate: str, enhanced_audio_url: str, transcript: list[dict]):
            if candidate == "gated":
                return {
                    "method": "FunASR speaker diarization gated tracks",
                    "status": "ok-diarization-gated",
                    "track_count": "1",
                    "tracks": [
                        {
                            "track_id": "gated",
                            "label": "gated",
                            "audio_url": enhanced_audio_url,
                            "description": "baseline",
                        }
                    ],
                }
            if candidate == "speechbrain":
                return {
                    "method": "SpeechBrain SepFormer",
                    "status": "ok",
                    "track_count": "2",
                    "tracks": [
                        {
                            "track_id": "speaker_1",
                            "label": "speaker 1",
                            "audio_url": enhanced_audio_url,
                            "description": "track 1",
                        },
                        {
                            "track_id": "speaker_2",
                            "label": "speaker 2",
                            "audio_url": enhanced_audio_url,
                            "description": "track 2",
                        },
                    ],
                }
            raise RuntimeError(candidate)

        with patch.dict(
            os.environ,
            {"QUALITY_ROUTER_ENABLED": "true", "SEPARATION_CANDIDATES": "gated,speechbrain"},
            clear=True,
        ):
            with patch("app.services.separation_service._run_separation_candidate", side_effect=fake_candidate):
                result = separate_with_quality_router("/static/uploads/router_enhanced.wav", transcript=[])

        self.assertEqual(result["track_count"], "2")
        self.assertEqual(result["metrics"]["quality_router_selected_separation"], "speechbrain")
        self.assertIn("Quality-aware separation candidate=speechbrain", result["tracks"][0]["description"])

    def test_mossformer2_candidate_uses_native_clearvoice_api_output(self) -> None:
        source = separation_service.UPLOAD_DIR / "mossformer_native_input.wav"
        output_dir = separation_service.UPLOAD_DIR / "router_mossformer2_test_tracks" / "MossFormer2_SS_16K"
        first = output_dir / "mossformer_native_input_s1.wav"
        second = output_dir / "mossformer_native_input_s2.wav"
        source.write_bytes(b"wav")

        class FakeClearVoice:
            def __init__(self, task, model_names):
                self.task = task
                self.model_names = model_names

            def __call__(self, input_path, online_write=False, output_path=None):
                output_dir.mkdir(parents=True, exist_ok=True)
                first.write_bytes(b"speaker1")
                second.write_bytes(b"speaker2")

        fake_module = type("FakeModule", (), {"ClearVoice": FakeClearVoice})
        try:
            with patch.object(separation_service.importlib, "import_module", return_value=fake_module):
                result = separation_service._run_clearvoice_mossformer2_separation(source, "router_mossformer2_test")
        finally:
            source.unlink(missing_ok=True)
            for path in separation_service.UPLOAD_DIR.glob("router_mossformer2_test_speaker_*.wav"):
                path.unlink(missing_ok=True)
            if first.exists():
                first.unlink()
            if second.exists():
                second.unlink()
            if output_dir.exists():
                output_dir.rmdir()
            parent = separation_service.UPLOAD_DIR / "router_mossformer2_test_tracks"
            if parent.exists():
                parent.rmdir()

        self.assertEqual(result["status"], "ok-mossformer2")
        self.assertEqual(result["track_count"], "2")
        self.assertIn("ClearVoice MossFormer2_SS_16K", result["method"])


def _write_wav(path: Path, seconds: int) -> None:
    sample_rate = 8000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * sample_rate * seconds)


if __name__ == "__main__":
    unittest.main()
