from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.audio_service import UPLOAD_DIR, resolve_static_url  # noqa: E402
from app.services.pipeline_service import (  # noqa: E402
    _demo_clean_source_separation,
    _select_separation_audio_url,
    _track_transcript_segments,
    process_audio_path,
)


class PipelineServiceTest(unittest.TestCase):
    def test_separation_input_defaults_to_normalized_audio_when_raw_is_available(self) -> None:
        raw_path = UPLOAD_DIR / "pipeline_raw_input.wav"
        raw_path.write_bytes(b"raw")
        with patch.dict(os.environ, {}, clear=True):
            try:
                url, source = _select_separation_audio_url(
                    {"enhanced_audio_url": "/static/uploads/enhanced.wav"},
                    Path("backend/app/static/uploads/normalized.wav"),
                    raw_path=raw_path,
                )
            finally:
                raw_path.unlink(missing_ok=True)

        self.assertEqual(source, "normalized")
        self.assertEqual(url, "/static/uploads/normalized.wav")

    def test_separation_input_defaults_to_normalized_audio_without_raw_path(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            url, source = _select_separation_audio_url(
                {"enhanced_audio_url": "/static/uploads/enhanced.wav"},
                Path("backend/app/static/uploads/normalized.wav"),
            )

        self.assertEqual(source, "normalized")
        self.assertEqual(url, "/static/uploads/normalized.wav")

    def test_separation_input_can_use_normalized_audio(self) -> None:
        with patch.dict(os.environ, {"SEPARATION_INPUT_SOURCE": "normalized"}, clear=True):
            url, source = _select_separation_audio_url(
                {"enhanced_audio_url": "/static/uploads/enhanced.wav"},
                Path("backend/app/static/uploads/normalized.wav"),
            )

        self.assertEqual(source, "normalized")
        self.assertEqual(url, "/static/uploads/normalized.wav")

    def test_demo_clean_source_separation_uses_manifest_tracks(self) -> None:
        created_paths: list[Path] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_a = temp_path / "source_a.wav"
            source_b = temp_path / "source_b.wav"
            manifest_path = temp_path / "manifest.jsonl"
            source_a.write_bytes(b"source-a")
            source_b.write_bytes(b"source-b")
            manifest_path.write_text(
                json.dumps(
                    {
                        "meeting": "UNIT_MIX",
                        "mix_path": "unit_mix.wav",
                        "source_paths": [str(source_a), str(source_b)],
                        "speakers": ["S1", "S2"],
                    }
                ),
                encoding="utf-8",
            )

            try:
                with patch.dict(os.environ, {"SEPARATION_DEMO_CLEAN_SOURCES": "true"}, clear=True), patch(
                    "app.services.pipeline_service.NEAR_MIX_MANIFEST_PATH",
                    manifest_path,
                ):
                    separation = _demo_clean_source_separation("unit_mix.wav")
                self.assertIsNotNone(separation)
                assert separation is not None
                created_paths = [resolve_static_url(track["audio_url"]) for track in separation["tracks"]]
                created_paths = [path for path in created_paths if path is not None]
                self.assertEqual(separation["track_count"], "2")
                self.assertEqual(separation["metrics"]["demo_clean_source_mode"], "on")
                self.assertEqual([path.read_bytes() for path in created_paths], [b"source-a", b"source-b"])
            finally:
                for path in created_paths:
                    path.unlink(missing_ok=True)

    def test_pipeline_separates_processed_audio_then_transcribes_each_track(self) -> None:
        raw_path = UPLOAD_DIR / "pipeline_flow_raw.wav"
        normalized_path = UPLOAD_DIR / "pipeline_flow_normalized.wav"
        enhanced_path = UPLOAD_DIR / "pipeline_flow_enhanced.wav"
        track_path = UPLOAD_DIR / "pipeline_flow_track_a.wav"
        raw_path.write_bytes(b"raw")
        normalized_path.write_bytes(b"normalized")
        enhanced_path.write_bytes(b"enhanced")
        track_path.write_bytes(b"track")
        asr_paths: list[Path] = []
        separation_urls: list[str] = []
        separation_transcripts: list[list[dict]] = []
        separation_reference_paths: list[Path | None] = []
        separation_display_names: list[str] = []

        def fake_resolve_static_url(url: str):
            if url == "/static/uploads/pipeline_flow_normalized.wav":
                return normalized_path
            if url == "/static/uploads/pipeline_flow_enhanced.wav":
                return enhanced_path
            if url == "/static/uploads/pipeline_flow_track_a.wav":
                return track_path
            return resolve_static_url(url)

        def fake_transcribe(audio_path, display_name, fallback=None):
            asr_paths.append(audio_path)
            return {
                "direct_asr_text": "",
                "enhanced_asr_text": "track transcript",
                "transcript": [{"speaker": "A", "start": "00:00", "end": "00:01", "text": "hello"}],
                "signal_metrics": {"ASR 后端": "unit-test", "ASR 状态": "success"},
            }

        def fake_separate(
            audio_url,
            transcript,
            reference_audio_path=None,
            display_name="",
            expected_speakers=None,
        ):
            separation_urls.append(audio_url)
            separation_transcripts.append(transcript)
            separation_reference_paths.append(reference_audio_path)
            separation_display_names.append(display_name)
            return {
                "method": "SpeechBrain SepFormer",
                "status": "ok",
                "track_count": "1",
                "tracks": [
                    {
                        "track_id": "track_a",
                        "label": "speaker A",
                        "audio_url": "/static/uploads/pipeline_flow_track_a.wav",
                        "description": "separated track",
                    }
                ],
                "metrics": {},
            }

        def fake_enhance(audio_path):
            self.assertEqual(audio_path, track_path)
            return {
                "original_audio_url": "/static/uploads/pipeline_flow_track_a.wav",
                "enhanced_audio_url": "/static/uploads/pipeline_flow_enhanced.wav",
                "method": "unit track enhancement",
                "metrics": {},
            }

        try:
            with patch.dict(os.environ, {"SEPARATION_DEMO_CLEAN_SOURCES": "false"}, clear=True):
                with patch("app.services.pipeline_service.normalize_upload", return_value=normalized_path), patch(
                    "app.services.pipeline_service.enhance_uploaded_audio",
                    side_effect=fake_enhance,
                ), patch("app.services.pipeline_service.resolve_static_url", side_effect=fake_resolve_static_url), patch(
                    "app.services.pipeline_service.transcribe_audio", side_effect=fake_transcribe
                ), patch(
                    "app.services.pipeline_service.separate_with_quality_router", side_effect=fake_separate
                ), patch(
                    "app.services.pipeline_service.build_meeting_analysis_metrics", return_value={}
                ), patch(
                    "app.services.pipeline_service.generate_summary",
                    return_value=SimpleNamespace(
                        summary={
                            "title": "unit",
                            "keywords": [],
                            "abstract": "unit",
                            "decisions": [],
                            "action_items": [],
                        },
                        metrics={},
                        used_llm=False,
                    ),
                ), patch(
                    "app.services.pipeline_service.classify_transcript_topics",
                    return_value=SimpleNamespace(topics=[], metrics={}, used_llm=False),
                ):
                    result = process_audio_path(raw_path, raw_path.name, case_id="unit", processing_mode="full")
        finally:
            raw_path.unlink(missing_ok=True)
            normalized_path.unlink(missing_ok=True)
            enhanced_path.unlink(missing_ok=True)
            track_path.unlink(missing_ok=True)

        self.assertEqual(asr_paths, [enhanced_path])
        self.assertEqual(separation_transcripts, [[]])
        self.assertEqual(result.signal_metrics["separation_input_source"], "normalized")
        self.assertEqual(len(separation_urls), 1)
        self.assertEqual(separation_urls[0], "/static/uploads/pipeline_flow_normalized.wav")
        self.assertEqual(separation_reference_paths, [raw_path])
        self.assertEqual(separation_display_names, [raw_path.name])
        self.assertEqual(result.enhanced_asr_text, "hello")
        self.assertEqual(result.separated_tracks[0].audio_url, "/static/uploads/pipeline_flow_enhanced.wav")
        self.assertEqual(result.separated_tracks[0].separated_audio_url, "/static/uploads/pipeline_flow_track_a.wav")
        self.assertEqual(result.separated_tracks[0].track_enhancement_status, "success")
        self.assertEqual(result.separated_tracks[0].asr_text, "track transcript")
        self.assertEqual(result.transcript[0].primary_track_id, "track_a")

    def test_fast_path_uses_quality_router_without_reference_sources(self) -> None:
        raw_path = UPLOAD_DIR / "near_mix_fast_path.wav"
        raw_path.write_bytes(b"near mix")
        separation = {
            "method": "SpeechBrain SepFormer",
            "status": "ok",
            "track_count": "2",
            "tracks": [
                {
                    "track_id": "model_1",
                    "label": "model speaker 1",
                    "audio_url": "/static/uploads/model_a.wav",
                    "description": "model output",
                }
            ],
            "metrics": {
                "quality_router_selected_separation": "libri2mix",
            },
        }
        router_calls = []

        def fake_router(
            audio_url,
            transcript,
            reference_audio_path=None,
            display_name="",
            expected_speakers=None,
        ):
            router_calls.append(
                {
                    "audio_url": audio_url,
                    "transcript": transcript,
                    "reference_audio_path": reference_audio_path,
                    "display_name": display_name,
                    "expected_speakers": expected_speakers,
                }
            )
            return separation

        try:
            with patch.dict(os.environ, {"SEPARATION_DEMO_CLEAN_SOURCES": "false"}, clear=True), patch(
                "app.services.pipeline_service.separate_with_quality_router",
                side_effect=fake_router,
            ):
                with patch("app.services.pipeline_service.normalize_upload") as normalize:
                    result = process_audio_path(
                        raw_path,
                        "R8001_M8004_near_all_speakers_mix.wav",
                        case_id="upload",
                        processing_mode="fast",
                    )
        finally:
            raw_path.unlink(missing_ok=True)

        normalize.assert_not_called()
        self.assertEqual(len(router_calls), 1)
        self.assertEqual(router_calls[0]["transcript"], [])
        self.assertEqual(router_calls[0]["reference_audio_path"], raw_path)
        self.assertIsNone(router_calls[0]["expected_speakers"])
        self.assertEqual(result.signal_metrics["quality_router_selected_separation"], "libri2mix")
        self.assertEqual(result.signal_metrics["fast_path_mode"], "quality-router-separation-only")
        self.assertEqual(result.separated_tracks[0].label, "model speaker 1")

    def test_near_mix_full_mode_runs_pipeline_then_uses_quality_router(self) -> None:
        raw_path = UPLOAD_DIR / "near_mix_full_raw.wav"
        normalized_path = UPLOAD_DIR / "near_mix_full_normalized.wav"
        enhanced_path = UPLOAD_DIR / "near_mix_full_enhanced.wav"
        raw_path.write_bytes(b"raw")
        normalized_path.write_bytes(b"normalized")
        enhanced_path.write_bytes(b"enhanced")
        separation = {
            "method": "ClearVoice MossFormer2_SS_16K",
            "status": "ok-mossformer2",
            "track_count": "1",
            "tracks": [
                {
                    "track_id": "model_1",
                    "label": "model speaker 1",
                    "audio_url": "/static/uploads/model_a.wav",
                    "description": "model output",
                }
            ],
            "metrics": {
                "quality_router_selected_separation": "mossformer2",
            },
        }

        def fake_resolve_static_url(url: str):
            if url == "/static/uploads/near_mix_full_normalized.wav":
                return normalized_path
            if url == "/static/uploads/near_mix_full_enhanced.wav":
                return enhanced_path
            return resolve_static_url(url)

        try:
            with patch.dict(os.environ, {"SEPARATION_DEMO_CLEAN_SOURCES": "false"}, clear=True), patch(
                "app.services.pipeline_service.normalize_upload",
                return_value=normalized_path,
            ) as normalize:
                with patch(
                    "app.services.pipeline_service.enhance_uploaded_audio",
                    return_value={
                        "original_audio_url": "/static/uploads/near_mix_full_normalized.wav",
                        "enhanced_audio_url": "/static/uploads/near_mix_full_enhanced.wav",
                        "method": "unit enhancement",
                        "metrics": {},
                    },
                ), patch("app.services.pipeline_service.resolve_static_url", side_effect=fake_resolve_static_url), patch(
                    "app.services.pipeline_service.transcribe_audio",
                    return_value={
                        "direct_asr_text": "",
                        "enhanced_asr_text": "enhanced transcript",
                        "transcript": [{"speaker": "A", "start": "00:00", "end": "00:01", "text": "hello"}],
                        "signal_metrics": {"ASR 后端": "unit-test", "ASR 状态": "success"},
                    },
                ), patch(
                    "app.services.pipeline_service.separate_with_quality_router",
                    return_value=separation,
                ) as router, patch(
                    "app.services.pipeline_service.build_meeting_analysis_metrics", return_value={}
                ), patch(
                    "app.services.pipeline_service.generate_summary",
                    return_value=SimpleNamespace(
                        summary={
                            "title": "unit",
                            "keywords": [],
                            "abstract": "unit",
                            "decisions": [],
                            "action_items": [],
                        },
                        metrics={},
                        used_llm=False,
                    ),
                ), patch(
                    "app.services.pipeline_service.classify_transcript_topics",
                    return_value=SimpleNamespace(topics=[], metrics={}, used_llm=False),
                ):
                    result = process_audio_path(
                        raw_path,
                        "R8001_M8004_near_all_speakers_mix.wav",
                        case_id="upload",
                        processing_mode="full",
                    )
        finally:
            raw_path.unlink(missing_ok=True)
            normalized_path.unlink(missing_ok=True)
            enhanced_path.unlink(missing_ok=True)

        normalize.assert_called_once()
        router.assert_called_once()
        self.assertNotIn("expected_speakers", router.call_args.kwargs)
        self.assertEqual(result.signal_metrics["processing_mode"], "full")
        self.assertEqual(result.signal_metrics["quality_router_selected_separation"], "mossformer2")

    def test_track_asr_segments_are_merged_for_readable_alignment(self) -> None:
        segments = _track_transcript_segments(
            [
                {"speaker": "raw", "start": "00:00", "end": "00:01", "text": "我"},
                {"speaker": "raw", "start": "00:01", "end": "00:02", "text": "今天"},
                {"speaker": "raw", "start": "00:02", "end": "00:03", "text": "讨论方案"},
            ],
            track_id="track_a",
            track_label="说话人 A",
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "我 今天 讨论方案")
        self.assertEqual(segments[0]["start"], "00:00")
        self.assertEqual(segments[0]["end"], "00:03")
        self.assertEqual(segments[0]["primary_track_id"], "track_a")


if __name__ == "__main__":
    unittest.main()
