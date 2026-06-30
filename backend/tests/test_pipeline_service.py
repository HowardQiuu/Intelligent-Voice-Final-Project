from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.audio_service import UPLOAD_DIR, resolve_static_url  # noqa: E402
from app.services.pipeline_service import _select_separation_audio_url, process_audio_path  # noqa: E402


class PipelineServiceTest(unittest.TestCase):
    def test_separation_input_defaults_to_raw_audio_when_available(self) -> None:
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
                staged_path = resolve_static_url(locals().get("url", ""))
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)

        self.assertEqual(source, "raw")
        self.assertTrue(url.startswith("/static/uploads/"))

    def test_separation_input_defaults_to_normalized_audio_without_raw_path(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            url, source = _select_separation_audio_url(
                {"enhanced_audio_url": "/static/uploads/enhanced.wav"},
                Path("backend/app/static/uploads/normalized.wav"),
            )

        self.assertEqual(source, "normalized")
        self.assertEqual(url, "/static/uploads/normalized.wav")

    def test_separation_input_can_use_enhanced_audio(self) -> None:
        with patch.dict(os.environ, {"SEPARATION_INPUT_SOURCE": "enhanced"}, clear=True):
            url, source = _select_separation_audio_url(
                {"enhanced_audio_url": "/static/uploads/enhanced.wav"},
                Path("backend/app/static/uploads/normalized.wav"),
            )

        self.assertEqual(source, "enhanced")
        self.assertEqual(url, "/static/uploads/enhanced.wav")

    def test_pipeline_uses_enhanced_audio_for_asr_and_raw_audio_for_separation(self) -> None:
        raw_path = UPLOAD_DIR / "pipeline_flow_raw.wav"
        normalized_path = UPLOAD_DIR / "pipeline_flow_normalized.wav"
        enhanced_path = UPLOAD_DIR / "pipeline_flow_enhanced.wav"
        raw_path.write_bytes(b"raw")
        normalized_path.write_bytes(b"normalized")
        enhanced_path.write_bytes(b"enhanced")
        asr_paths: list[Path] = []
        separation_urls: list[str] = []
        separation_reference_paths: list[Path | None] = []
        separation_display_names: list[str] = []

        def fake_resolve_static_url(url: str):
            if url == "/static/uploads/pipeline_flow_normalized.wav":
                return normalized_path
            if url == "/static/uploads/pipeline_flow_enhanced.wav":
                return enhanced_path
            return resolve_static_url(url)

        def fake_transcribe(audio_path, display_name, fallback=None):
            asr_paths.append(audio_path)
            return {
                "direct_asr_text": "",
                "enhanced_asr_text": "enhanced transcript",
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
            separation_reference_paths.append(reference_audio_path)
            separation_display_names.append(display_name)
            return {
                "method": "SpeechBrain SepFormer",
                "status": "ok",
                "track_count": "2",
                "tracks": [],
                "metrics": {},
            }

        try:
            with patch.dict(os.environ, {}, clear=True):
                with patch("app.services.pipeline_service.normalize_upload", return_value=normalized_path), patch(
                    "app.services.pipeline_service.enhance_uploaded_audio",
                    return_value={
                        "original_audio_url": "/static/uploads/pipeline_flow_normalized.wav",
                        "enhanced_audio_url": "/static/uploads/pipeline_flow_enhanced.wav",
                        "method": "unit enhancement",
                        "metrics": {},
                    },
                ), patch("app.services.pipeline_service.resolve_static_url", side_effect=fake_resolve_static_url), patch(
                    "app.services.pipeline_service.generate_enhancement_visual", return_value=("", {})
                ), patch("app.services.pipeline_service.transcribe_audio", side_effect=fake_transcribe), patch(
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
            for url in separation_urls:
                staged_path = resolve_static_url(url)
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)

        self.assertEqual(asr_paths, [enhanced_path])
        self.assertEqual(result.signal_metrics["separation_input_source"], "raw")
        self.assertEqual(len(separation_urls), 1)
        self.assertNotEqual(separation_urls[0], "/static/uploads/pipeline_flow_enhanced.wav")
        self.assertTrue(separation_urls[0].startswith("/static/uploads/"))
        self.assertEqual(separation_reference_paths, [raw_path])
        self.assertEqual(separation_display_names, [raw_path.name])

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
            with patch("app.services.pipeline_service.separate_with_quality_router", side_effect=fake_router):
                with patch("app.services.pipeline_service.normalize_upload") as normalize:
                    result = process_audio_path(raw_path, "R8001_M8004_near_all_speakers_mix.wav", case_id="upload")
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
            with patch("app.services.pipeline_service.normalize_upload", return_value=normalized_path) as normalize:
                with patch(
                    "app.services.pipeline_service.enhance_uploaded_audio",
                    return_value={
                        "original_audio_url": "/static/uploads/near_mix_full_normalized.wav",
                        "enhanced_audio_url": "/static/uploads/near_mix_full_enhanced.wav",
                        "method": "unit enhancement",
                        "metrics": {},
                    },
                ), patch("app.services.pipeline_service.resolve_static_url", side_effect=fake_resolve_static_url), patch(
                    "app.services.pipeline_service.generate_enhancement_visual", return_value=("", {})
                ), patch(
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


if __name__ == "__main__":
    unittest.main()
