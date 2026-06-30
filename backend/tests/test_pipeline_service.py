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

        def fake_separate(audio_url, transcript):
            separation_urls.append(audio_url)
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
                    result = process_audio_path(raw_path, raw_path.name, case_id="unit")
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


if __name__ == "__main__":
    unittest.main()
