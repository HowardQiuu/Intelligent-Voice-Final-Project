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

    def test_speechbrain_savedir_is_model_specific(self) -> None:
        savedir = separation_service._speechbrain_savedir("speechbrain/custom-separator")

        self.assertEqual(savedir.name, "speechbrain_custom-separator")
        self.assertIn("speechbrain", str(savedir.parent))

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

    def test_default_router_candidates_include_libri2mix(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            candidates = separation_service._get_separation_candidates()

        self.assertEqual(candidates, ["libri2mix", "mossformer2", "gated"])

    def test_libri2mix_candidate_uses_librimix_model(self) -> None:
        source_path = separation_service.UPLOAD_DIR / "libri2mix_candidate.wav"
        source_path.write_bytes(b"wav")
        calls: list[dict] = []

        def fake_separate(source_path_arg, output_stem, max_seconds=None, model_name_override=None, refine_tracks=True):
            calls.append(
                {
                    "output_stem": output_stem,
                    "model_name_override": model_name_override,
                    "refine_tracks": refine_tracks,
                }
            )
            return {"method": "SpeechBrain SepFormer (speechbrain/sepformer-libri2mix, cuda)", "status": "ok", "tracks": []}

        try:
            with patch("app.services.separation_service.get_audio_duration_seconds", return_value=3.0):
                with patch("app.services.separation_service._separate_with_speechbrain", side_effect=fake_separate):
                    separation_service._run_separation_candidate(
                        "libri2mix",
                        separation_service.audio_url(source_path),
                        [],
                    )
        finally:
            source_path.unlink(missing_ok=True)

        self.assertEqual(calls[0]["model_name_override"], "speechbrain/sepformer-libri2mix")
        self.assertFalse(calls[0]["refine_tracks"])
        self.assertTrue(calls[0]["output_stem"].startswith("router_libri2mix_"))

    def test_libri2mix_default_bonus_breaks_speechbrain_near_tie(self) -> None:
        def fake_candidate(candidate: str, enhanced_audio_url: str, transcript: list[dict]):
            tracks = [
                {"track_id": f"{candidate}_1", "label": "speaker 1", "audio_url": enhanced_audio_url},
                {"track_id": f"{candidate}_2", "label": "speaker 2", "audio_url": enhanced_audio_url},
            ]
            if candidate == "speechbrain":
                return {"method": "SpeechBrain SepFormer direct candidate", "status": "ok", "tracks": tracks}
            if candidate == "libri2mix":
                return {"method": "SpeechBrain SepFormer (speechbrain/sepformer-libri2mix, cuda)", "status": "ok", "tracks": tracks}
            raise RuntimeError(candidate)

        with patch.dict(
            os.environ,
            {"QUALITY_ROUTER_ENABLED": "true", "SEPARATION_CANDIDATES": "libri2mix,speechbrain"},
            clear=True,
        ):
            with patch("app.services.separation_service._run_separation_candidate", side_effect=fake_candidate):
                result = separate_with_quality_router("/static/uploads/router_enhanced.wav", transcript=[])

        self.assertEqual(result["metrics"]["quality_router_selected_separation"], "libri2mix")
        self.assertIn("libri2mix=50.5", result["metrics"]["quality_router_separation_candidates"])
        self.assertIn("speechbrain=49.0", result["metrics"]["quality_router_separation_candidates"])

    def test_quality_router_penalizes_candidates_below_expected_speakers(self) -> None:
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
                            "description": "single gated track",
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
            {
                "QUALITY_ROUTER_ENABLED": "true",
                "SEPARATION_CANDIDATES": "gated,speechbrain",
                "SEPARATION_EXPECTED_SPEAKERS": "2",
            },
            clear=True,
        ):
            with patch("app.services.separation_service._run_separation_candidate", side_effect=fake_candidate):
                result = separate_with_quality_router("/static/uploads/router_enhanced.wav", transcript=[])

        self.assertEqual(result["track_count"], "2")
        self.assertEqual(result["metrics"]["quality_router_selected_separation"], "speechbrain")
        self.assertEqual(result["metrics"]["quality_router_expected_speakers"], "2")
        self.assertIn("gated=14.5", result["metrics"]["quality_router_separation_candidates"])

    def test_mixture_consistency_projection_restores_track_sum(self) -> None:
        numpy = importlib.import_module("numpy")
        soundfile = importlib.import_module("soundfile")
        sample_rate = 8000
        t = numpy.arange(sample_rate, dtype=numpy.float32) / sample_rate
        source_a = (0.2 * numpy.sin(2 * numpy.pi * 220 * t)).astype("float32")
        source_b = (0.2 * numpy.sin(2 * numpy.pi * 440 * t)).astype("float32")
        mixture = source_a + source_b
        source_path = separation_service.UPLOAD_DIR / "mixture_consistency_source.wav"
        track_1 = separation_service.UPLOAD_DIR / "mixture_consistency_track_1.wav"
        track_2 = separation_service.UPLOAD_DIR / "mixture_consistency_track_2.wav"
        soundfile.write(str(source_path), mixture, sample_rate)
        soundfile.write(str(track_1), source_a * 0.2, sample_rate)
        soundfile.write(str(track_2), source_b * 0.2, sample_rate)
        tracks = [
            {"audio_url": separation_service.audio_url(track_1), "description": "track 1"},
            {"audio_url": separation_service.audio_url(track_2), "description": "track 2"},
        ]
        try:
            with patch.dict(os.environ, {"SEPARATION_MIXTURE_CONSISTENCY": "true"}, clear=True):
                metrics = separation_service._apply_mixture_consistency_projection(source_path, tracks)
            projected_1 = soundfile.read(str(track_1), dtype="float32")[0]
            projected_2 = soundfile.read(str(track_2), dtype="float32")[0]
        finally:
            source_path.unlink(missing_ok=True)
            track_1.unlink(missing_ok=True)
            track_2.unlink(missing_ok=True)

        self.assertEqual(metrics["mixture_consistency_projection"], "applied")
        self.assertLess(float(numpy.max(numpy.abs((projected_1 + projected_2) - mixture))), 1e-4)
        self.assertIn("Mixture consistency", tracks[0]["description"])

    def test_quality_router_boosts_mossformer2_for_overlapped_transcript(self) -> None:
        def fake_candidate(candidate: str, enhanced_audio_url: str, transcript: list[dict]):
            if candidate == "speechbrain":
                return {
                    "method": "SpeechBrain SepFormer",
                    "status": "ok",
                    "track_count": "2",
                    "tracks": [
                        {"track_id": "speaker_1", "label": "speaker 1", "audio_url": enhanced_audio_url},
                        {"track_id": "speaker_2", "label": "speaker 2", "audio_url": enhanced_audio_url},
                    ],
                }
            if candidate == "mossformer2":
                return {
                    "method": "ClearVoice MossFormer2_SS_16K",
                    "status": "ok-mossformer2",
                    "track_count": "2",
                    "tracks": [
                        {"track_id": "moss_1", "label": "speaker 1", "audio_url": enhanced_audio_url},
                        {"track_id": "moss_2", "label": "speaker 2", "audio_url": enhanced_audio_url},
                    ],
                }
            raise RuntimeError(candidate)

        transcript = [
            {"speaker": "A", "start": 0.0, "end": 3.0, "text": "hello"},
            {"speaker": "B", "start": 1.0, "end": 4.0, "text": "world"},
        ]
        with patch.dict(
            os.environ,
            {
                "QUALITY_ROUTER_ENABLED": "true",
                "SEPARATION_CANDIDATES": "speechbrain,mossformer2",
                "MOSSFORMER2_OVERLAP_THRESHOLD": "0.05",
                "MOSSFORMER2_OVERLAP_BOOST": "8.0",
            },
            clear=True,
        ):
            with patch("app.services.separation_service._run_separation_candidate", side_effect=fake_candidate):
                result = separate_with_quality_router("/static/uploads/router_enhanced.wav", transcript=transcript)

        self.assertEqual(result["metrics"]["quality_router_selected_separation"], "mossformer2")
        self.assertEqual(result["metrics"]["quality_router_expected_speakers"], "2")
        self.assertIn("quality_router_transcript_overlap_ratio", result["metrics"])

    def test_quality_router_diagnostic_rerank_can_promote_mossformer2(self) -> None:
        def fake_candidate(candidate: str, enhanced_audio_url: str, transcript: list[dict]):
            if candidate == "speechbrain":
                return {
                    "method": "SpeechBrain SepFormer",
                    "status": "ok",
                    "track_count": "2",
                    "tracks": [
                        {"track_id": "speaker_1", "label": "speaker 1", "audio_url": enhanced_audio_url},
                        {"track_id": "speaker_2", "label": "speaker 2", "audio_url": enhanced_audio_url},
                    ],
                }
            if candidate == "mossformer2":
                return {
                    "method": "ClearVoice MossFormer2_SS_16K",
                    "status": "ok-mossformer2",
                    "track_count": "2",
                    "tracks": [
                        {"track_id": "moss_1", "label": "speaker 1", "audio_url": enhanced_audio_url},
                        {"track_id": "moss_2", "label": "speaker 2", "audio_url": enhanced_audio_url},
                    ],
                }
            raise RuntimeError(candidate)

        def fake_score(result, expected_speakers=None, overlap_ratio=0.0):
            return 77.0 if "SpeechBrain" in result["method"] else 73.0

        def fake_diagnostics(source_path, result):
            if "SpeechBrain" in result["method"]:
                return {
                    "sum_mix_correlation": -0.1,
                    "sum_mix_residual_ratio": 3.0,
                    "inter_track_correlation": -0.01,
                    "track_energy_balance": 0.7,
                    "track_overlap_ratio": 0.4,
                }
            return {
                "sum_mix_correlation": 0.99,
                "sum_mix_residual_ratio": 0.2,
                "inter_track_correlation": 0.01,
                "track_energy_balance": 1.0,
                "track_overlap_ratio": 0.5,
            }

        source = separation_service.UPLOAD_DIR / "diagnostic_rerank_source.wav"
        source.write_bytes(b"wav")
        try:
            with patch.dict(
                os.environ,
                {
                    "QUALITY_ROUTER_ENABLED": "true",
                    "SEPARATION_CANDIDATES": "speechbrain,mossformer2",
                    "MOSSFORMER2_DIAGNOSTIC_BONUS": "8.0",
                },
                clear=True,
            ):
                with patch("app.services.separation_service._run_separation_candidate", side_effect=fake_candidate):
                    with patch("app.services.separation_service._score_separation_result", side_effect=fake_score):
                        with patch(
                            "app.services.separation_service._candidate_output_diagnostics",
                            side_effect=fake_diagnostics,
                        ):
                            result = separate_with_quality_router(separation_service.audio_url(source), transcript=[])
        finally:
            source.unlink(missing_ok=True)

        self.assertEqual(result["metrics"]["quality_router_selected_separation"], "mossformer2")
        self.assertEqual(result["metrics"]["quality_router_selected_separation_score"], "81.0")
        self.assertEqual(
            result["metrics"]["quality_router_diagnostic_rerank"],
            "mossformer2_overlap_reconstruction_boost",
        )
        self.assertIn("mossformer2=81.0", result["metrics"]["quality_router_separation_candidates"])

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

    def test_stft_mask_refinement_rewrites_speechbrain_tracks(self) -> None:
        numpy = REAL_IMPORT_MODULE("numpy")
        soundfile = REAL_IMPORT_MODULE("soundfile")
        sample_rate = 16000
        times = numpy.arange(sample_rate, dtype="float32") / sample_rate
        first = 0.2 * numpy.sin(2 * numpy.pi * 440 * times)
        second = 0.2 * numpy.sin(2 * numpy.pi * 660 * times)
        first[int(sample_rate * 0.5) :] = 0.0
        second[: int(sample_rate * 0.5)] = 0.0
        source = separation_service.UPLOAD_DIR / "stft_refine_mix.wav"
        track_a = separation_service.UPLOAD_DIR / "stft_refine_a.wav"
        track_b = separation_service.UPLOAD_DIR / "stft_refine_b.wav"
        soundfile.write(str(source), first + second, sample_rate)
        soundfile.write(str(track_a), first, 8000)
        soundfile.write(str(track_b), second, 8000)
        tracks = [
            {"audio_url": separation_service.audio_url(track_a), "description": "a"},
            {"audio_url": separation_service.audio_url(track_b), "description": "b"},
        ]

        try:
            with patch.dict(
                os.environ,
                {
                    "SEPARATION_STFT_MASK_REFINEMENT": "true",
                    "SEPARATION_STFT_MASK_N_FFT": "256",
                    "SEPARATION_STFT_MASK_HOP": "64",
                    "SEPARATION_STFT_MASK_POWER": "1.5",
                },
                clear=True,
            ):
                metrics = separation_service._apply_stft_mask_refinement(source, tracks)
            refined, refined_rate = soundfile.read(str(track_a), always_2d=True, dtype="float32")
        finally:
            source.unlink(missing_ok=True)
            track_a.unlink(missing_ok=True)
            track_b.unlink(missing_ok=True)

        self.assertEqual(metrics["stft_mask_refinement"], "applied")
        self.assertEqual(refined_rate, sample_rate)
        self.assertEqual(len(refined), sample_rate)
        self.assertIn("STFT mask refinement applied", tracks[0]["description"])

    def test_low_overlap_leakage_suppression_applies_to_exclusive_tracks(self) -> None:
        numpy = REAL_IMPORT_MODULE("numpy")
        soundfile = REAL_IMPORT_MODULE("soundfile")
        sample_rate = 16000
        samples = numpy.zeros(sample_rate, dtype="float32")
        first = samples.copy()
        second = samples.copy()
        first[: sample_rate // 2] = 0.3
        first[sample_rate // 2 :] = 0.005
        second[: sample_rate // 2] = 0.005
        second[sample_rate // 2 :] = 0.3
        track_a = separation_service.UPLOAD_DIR / "low_overlap_a.wav"
        track_b = separation_service.UPLOAD_DIR / "low_overlap_b.wav"
        soundfile.write(str(track_a), first, sample_rate)
        soundfile.write(str(track_b), second, sample_rate)
        tracks = [
            {"audio_url": separation_service.audio_url(track_a), "description": "a"},
            {"audio_url": separation_service.audio_url(track_b), "description": "b"},
        ]

        try:
            with patch.dict(
                os.environ,
                {
                    "SEPARATION_LOW_OVERLAP_LEAKAGE_SUPPRESSION": "true",
                    "SEPARATION_LOW_OVERLAP_THRESHOLD": "0.2",
                    "SEPARATION_LOW_OVERLAP_DOMINANCE_DB": "3.0",
                    "SEPARATION_LOW_OVERLAP_LOSER_GAIN": "0.1",
                    "SEPARATION_LOW_OVERLAP_ACTIVE_FLOOR_DB": "-45",
                },
                clear=True,
            ):
                metrics = separation_service._apply_low_overlap_leakage_suppression(tracks)
            refined_a, _ = soundfile.read(str(track_a), always_2d=True, dtype="float32")
            refined_b, _ = soundfile.read(str(track_b), always_2d=True, dtype="float32")
        finally:
            track_a.unlink(missing_ok=True)
            track_b.unlink(missing_ok=True)

        self.assertEqual(metrics["low_overlap_leakage_suppression"], "applied")
        self.assertLess(float(refined_a[-100, 0]), 0.01)
        self.assertLess(float(refined_b[100, 0]), 0.01)
        self.assertIn("Low-overlap leakage suppression applied", tracks[0]["description"])

    def test_speechbrain_residual_projection_reconstructs_mixture_sum(self) -> None:
        numpy = REAL_IMPORT_MODULE("numpy")
        soundfile = REAL_IMPORT_MODULE("soundfile")
        sample_rate = 16000
        times = numpy.arange(sample_rate, dtype="float32") / sample_rate
        first = 0.1 * numpy.sin(2 * numpy.pi * 220 * times)
        second = 0.1 * numpy.sin(2 * numpy.pi * 330 * times)
        mixture = first + second
        track_a = separation_service.UPLOAD_DIR / "residual_projection_a.wav"
        track_b = separation_service.UPLOAD_DIR / "residual_projection_b.wav"
        source = separation_service.UPLOAD_DIR / "residual_projection_mix.wav"
        soundfile.write(str(source), mixture, sample_rate)
        soundfile.write(str(track_a), first * 0.6, sample_rate)
        soundfile.write(str(track_b), second * 0.6, sample_rate)
        tracks = [
            {"audio_url": separation_service.audio_url(track_a), "description": "a"},
            {"audio_url": separation_service.audio_url(track_b), "description": "b"},
        ]

        try:
            before_a, _ = soundfile.read(str(track_a), always_2d=True, dtype="float32")
            before_b, _ = soundfile.read(str(track_b), always_2d=True, dtype="float32")
            before_error = float(numpy.mean(((before_a + before_b)[:, 0] - mixture) ** 2))
            with patch.dict(
                os.environ,
                {
                    "SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION": "true",
                    "SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION_AMOUNT": "1.0",
                },
                clear=True,
            ):
                metrics = separation_service._apply_speechbrain_residual_projection(source, tracks)
            after_a, _ = soundfile.read(str(track_a), always_2d=True, dtype="float32")
            after_b, _ = soundfile.read(str(track_b), always_2d=True, dtype="float32")
            after_error = float(numpy.mean(((after_a + after_b)[:, 0] - mixture) ** 2))
        finally:
            source.unlink(missing_ok=True)
            track_a.unlink(missing_ok=True)
            track_b.unlink(missing_ok=True)

        self.assertEqual(metrics["speechbrain_residual_projection"], "applied")
        self.assertLess(after_error, before_error)
        self.assertIn("SpeechBrain residual projection applied", tracks[0]["description"])


def _write_wav(path: Path, seconds: int) -> None:
    sample_rate = 8000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * sample_rate * seconds)


if __name__ == "__main__":
    unittest.main()
