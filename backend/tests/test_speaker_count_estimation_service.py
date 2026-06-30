from __future__ import annotations

import math
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import speaker_count_estimation_service as estimator  # noqa: E402
from app.services.audio_service import UPLOAD_DIR, audio_url  # noqa: E402


class SpeakerCountEstimationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        estimator._BACKEND_CACHE.clear()
        estimator._BACKEND_ERROR_CACHE.clear()

    def test_estimates_count_from_clean_candidate_tracks_with_explicit_local_diagnostic(self) -> None:
        paths = [
            UPLOAD_DIR / "speaker_count_est_a.wav",
            UPLOAD_DIR / "speaker_count_est_b.wav",
            UPLOAD_DIR / "speaker_count_est_c.wav",
        ]
        for path, frequency in zip(paths, [220.0, 440.0, 880.0]):
            _write_tone(path, frequency)

        tracks = [
            {"track_id": f"track_{index}", "label": f"track {index}", "audio_url": audio_url(path)}
            for index, path in enumerate(paths, start=1)
        ]
        try:
            with patch.dict(
                "os.environ",
                {
                    "SPEAKER_EMBEDDING_BACKEND": "local_spectral",
                    "SPEAKER_EMBEDDING_STRONG_REQUIRED": "false",
                    "SPEAKER_COUNT_CLUSTER_SIMILARITY": "0.92",
                },
                clear=True,
            ):
                result = estimator.estimate_speaker_count_from_tracks(tracks)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

        self.assertEqual(result["status"], "degraded_local_embedding")
        self.assertEqual(result["embedding_backend"], "local_spectral")
        self.assertEqual(result["estimated_speaker_count"], 0)
        self.assertEqual(len(result["clusters"]), 3)
        self.assertGreaterEqual(result["min_track_quality"], 0.8)

    def test_strong_backend_is_called_and_normalized_before_clustering(self) -> None:
        backend = _FakeEmbeddingBackend(
            {
                "track_1": [3.0, 4.0],
                "track_2": [6.0, 8.0],
            }
        )
        paths = _write_track_set([220.0, 330.0], prefix="speaker_count_norm")
        tracks = _tracks_for_paths(paths)
        try:
            with patch.dict("os.environ", {"SPEAKER_EMBEDDING_BACKEND": "ecapa", "SPEAKER_CLUSTER_THRESHOLD": "0.95"}, clear=True):
                with patch.object(estimator, "_create_backend", return_value=backend):
                    with patch.object(estimator, "_score_track_quality", return_value=0.95):
                        result = estimator.estimate_speaker_count_from_tracks(tracks)
        finally:
            _unlink_all(paths)

        self.assertEqual(backend.calls, ["track_1", "track_2"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["embedding_backend"], "ecapa")
        self.assertEqual(result["embedding_backend_status"], "strong")
        self.assertEqual(result["raw_global_estimated_speaker_count"], 1)
        self.assertEqual(result["global_estimated_speaker_count"], 2)
        self.assertEqual(result["global_count_source"], "single_window_tracks")

    def test_same_speaker_tracks_across_windows_share_global_speaker_id(self) -> None:
        backend = _FakeEmbeddingBackend(
            {
                "w1_s1": [1.0, 0.0],
                "w2_s1": [0.98, 0.02],
                "w1_s2": [0.0, 1.0],
                "w2_s2": [0.03, 0.97],
            }
        )
        paths = _write_track_set([220.0, 225.0, 440.0, 445.0], prefix="speaker_count_cross")
        tracks = [
            {"track_id": track_id, "label": track_id, "audio_url": audio_url(path)}
            for track_id, path in zip(["w1_s1", "w2_s1", "w1_s2", "w2_s2"], paths)
        ]
        try:
            with patch.dict("os.environ", {"SPEAKER_EMBEDDING_BACKEND": "ecapa", "SPEAKER_CLUSTER_THRESHOLD": "0.90"}, clear=True):
                with patch.object(estimator, "_create_backend", return_value=backend):
                    with patch.object(estimator, "_score_track_quality", return_value=0.95):
                        result = estimator.estimate_speaker_count_from_tracks(tracks)
        finally:
            _unlink_all(paths)

        self.assertEqual(result["global_estimated_speaker_count"], 2)
        speaker_ids = {track["track_id"]: track["global_speaker_id"] for track in result["tracks"]}
        self.assertEqual(speaker_ids["w1_s1"], speaker_ids["w2_s1"])
        self.assertEqual(speaker_ids["w1_s2"], speaker_ids["w2_s2"])
        self.assertNotEqual(speaker_ids["w1_s1"], speaker_ids["w1_s2"])

    def test_different_speaker_tracks_do_not_merge_below_threshold(self) -> None:
        backend = _FakeEmbeddingBackend(
            {
                "track_1": [1.0, 0.0, 0.0],
                "track_2": [0.0, 1.0, 0.0],
                "track_3": [0.0, 0.0, 1.0],
            }
        )
        paths = _write_track_set([220.0, 440.0, 880.0], prefix="speaker_count_distinct")
        tracks = _tracks_for_paths(paths)
        try:
            with patch.dict("os.environ", {"SPEAKER_EMBEDDING_BACKEND": "ecapa", "SPEAKER_CLUSTER_THRESHOLD": "0.80"}, clear=True):
                with patch.object(estimator, "_create_backend", return_value=backend):
                    with patch.object(estimator, "_score_track_quality", return_value=0.95):
                        result = estimator.estimate_speaker_count_from_tracks(tracks)
        finally:
            _unlink_all(paths)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["global_estimated_speaker_count"], 3)
        self.assertEqual(len(result["clusters"]), 3)

    def test_unavailable_strong_backend_does_not_report_success(self) -> None:
        path = UPLOAD_DIR / "speaker_count_unavailable.wav"
        _write_tone(path, 330.0)
        tracks = [{"track_id": "track_1", "label": "track 1", "audio_url": audio_url(path)}]
        try:
            with patch.dict("os.environ", {"SPEAKER_EMBEDDING_BACKEND": "ecapa"}, clear=True):
                with patch.object(estimator, "_create_backend", side_effect=RuntimeError("missing model")):
                    result = estimator.estimate_speaker_count_from_tracks(tracks)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(result["status"], "embedding_backend_unavailable")
        self.assertEqual(result["embedding_backend_status"], "unavailable")
        self.assertEqual(result["global_estimated_speaker_count"], 0)
        self.assertIn("missing model", result["embedding_backend_error"])

    def test_adds_estimation_metrics_without_changing_tracks(self) -> None:
        backend = _FakeEmbeddingBackend({"track_1": [1.0, 0.0]})
        path = UPLOAD_DIR / "speaker_count_est_single.wav"
        _write_tone(path, 330.0)
        separation = {
            "method": "unit",
            "status": "ok",
            "track_count": "1",
            "tracks": [{"track_id": "track_1", "label": "track 1", "audio_url": audio_url(path)}],
            "metrics": {"quality_router_selected_separation": "unit"},
        }
        try:
            with patch.dict("os.environ", {"SPEAKER_EMBEDDING_BACKEND": "ecapa"}, clear=True):
                with patch.object(estimator, "_create_backend", return_value=backend):
                    annotated = estimator.add_speaker_count_estimation(separation)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(annotated["tracks"], separation["tracks"])
        self.assertEqual(annotated["speaker_count_estimation"]["estimated_speaker_count"], 1)
        self.assertEqual(annotated["metrics"]["estimated_speaker_count"], "1")
        self.assertEqual(annotated["metrics"]["speaker_count_embedding_backend_status"], "strong")
        self.assertEqual(annotated["metrics"]["quality_router_selected_separation"], "unit")


class _FakeEmbeddingBackend:
    name = "ecapa"
    status = "strong"
    strong = True

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[str] = []

    def embed(self, track: dict) -> list[float]:
        self.calls.append(track["track_id"])
        return self.embeddings[track["track_id"]]


def _write_track_set(frequencies: list[float], *, prefix: str) -> list[Path]:
    paths = [UPLOAD_DIR / f"{prefix}_{index}.wav" for index in range(1, len(frequencies) + 1)]
    for path, frequency in zip(paths, frequencies):
        _write_tone(path, frequency)
    return paths


def _tracks_for_paths(paths: list[Path]) -> list[dict]:
    return [
        {"track_id": f"track_{index}", "label": f"track {index}", "audio_url": audio_url(path)}
        for index, path in enumerate(paths, start=1)
    ]


def _unlink_all(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _write_tone(path: Path, frequency: float, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    frame_count = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(12000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            frames.extend(value.to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()
