from __future__ import annotations

import os
import sys
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import asr_service  # noqa: E402
from app.services.asr_service import fallback_upload_result, transcribe_audio  # noqa: E402
from app.services.audio_service import UPLOAD_DIR  # noqa: E402


class FakeWhisperModel:
    calls: list[str] = []
    init_count = 0

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        type(self).init_count += 1
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path: str, language: str, vad_filter: bool, word_timestamps: bool):
        self.calls.append(path)
        offset = 4.2 * (len(self.calls) - 1)
        segments = [
            SimpleNamespace(start=0.0, end=4.2, text=f"今天讨论自动转写模块 {offset:.0f}。"),
            SimpleNamespace(start=4.2, end=9.0, text="后续接入会议摘要生成。"),
        ]
        return segments, SimpleNamespace(language=language)


class FailingCudaWhisperModel(FakeWhisperModel):
    cpu_init_count = 0
    cuda_init_count = 0

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        if device == "cuda":
            type(self).cuda_init_count += 1
            raise RuntimeError("cuda unavailable")
        type(self).cpu_init_count += 1
        super().__init__(model_name, device, compute_type)


class AsrServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeWhisperModel.calls = []
        FakeWhisperModel.init_count = 0
        FailingCudaWhisperModel.calls = []
        FailingCudaWhisperModel.init_count = 0
        FailingCudaWhisperModel.cpu_init_count = 0
        FailingCudaWhisperModel.cuda_init_count = 0
        asr_service._WHISPER_MODEL_CACHE.clear()
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.audio_path = UPLOAD_DIR / "asr_test.wav"
        _write_wav(self.audio_path, seconds=1)

    def tearDown(self) -> None:
        self.audio_path.unlink(missing_ok=True)
        self.env_patch.stop()

    def test_placeholder_backend_returns_fallback_with_metrics(self) -> None:
        with patch.dict(os.environ, {"ASR_BACKEND": "placeholder"}, clear=True):
            result = transcribe_audio(self.audio_path, "meeting.wav")

        self.assertEqual(result["enhanced_asr_text"], fallback_upload_result("meeting.wav")["enhanced_asr_text"])
        self.assertEqual(result["signal_metrics"]["ASR 后端"], "placeholder")
        self.assertEqual(result["signal_metrics"]["ASR 状态"], "placeholder")

    def test_faster_whisper_success_builds_transcript(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASR_BACKEND": "faster-whisper",
                "ASR_MODEL": "tiny",
                "ASR_DEVICE": "cpu",
                "ASR_COMPUTE_TYPE": "int8",
                "ASR_LANGUAGE": "zh",
            },
            clear=True,
        ):
            with patch("app.services.asr_service._load_whisper_model_class", return_value=FakeWhisperModel):
                result = transcribe_audio(self.audio_path, "meeting.wav")

        self.assertEqual(result["signal_metrics"]["ASR 状态"], "success")
        self.assertEqual(result["signal_metrics"]["ASR 模型"], "tiny")
        self.assertIn("今天讨论自动转写模块", result["enhanced_asr_text"])
        self.assertEqual(len(result["transcript"]), 2)
        self.assertEqual(result["transcript"][0]["speaker"], "说话人")
        self.assertEqual(result["transcript"][0]["start"], "00:00")

    def test_faster_whisper_unavailable_falls_back(self) -> None:
        with patch.dict(os.environ, {"ASR_BACKEND": "faster-whisper"}, clear=True):
            with patch("app.services.asr_service._load_whisper_model_class", side_effect=ImportError("missing")):
                result = transcribe_audio(self.audio_path, "meeting.wav")

        self.assertTrue(result["signal_metrics"]["ASR 状态"].startswith("unavailable"))
        self.assertEqual(result["transcript"][0]["speaker"], "说话人A")

    def test_long_audio_uses_chunked_asr(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASR_BACKEND": "faster-whisper",
                "ASR_MAX_SECONDS": "10",
                "ASR_CHUNK_SECONDS": "60",
            },
            clear=True,
        ):
            with patch("app.services.asr_service.get_audio_duration_seconds", return_value=120.0):
                with patch("app.services.asr_service._load_whisper_model_class", return_value=FakeWhisperModel):
                    with patch(
                        "app.services.asr_service._split_audio_to_chunks",
                        return_value=[(self.audio_path, 0.0), (self.audio_path, 60.0)],
                    ):
                        result = transcribe_audio(self.audio_path, "meeting.wav")

        self.assertEqual(result["signal_metrics"]["ASR 状态"], "success-chunked")
        self.assertEqual(result["signal_metrics"]["ASR 后端"], "faster-whisper")
        self.assertEqual(len(FakeWhisperModel.calls), 2)
        self.assertEqual(result["transcript"][2]["start"], "01:00")

    def test_faster_whisper_reuses_cached_model(self) -> None:
        env = {
            "ASR_BACKEND": "faster-whisper",
            "ASR_MODEL": "tiny",
            "ASR_DEVICE": "cpu",
            "ASR_COMPUTE_TYPE": "int8",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("app.services.asr_service._load_whisper_model_class", return_value=FakeWhisperModel):
                first = transcribe_audio(self.audio_path, "meeting.wav")
                second = transcribe_audio(self.audio_path, "meeting.wav")

        self.assertEqual(FakeWhisperModel.init_count, 1)
        self.assertEqual(first["signal_metrics"]["ASR模型缓存"], "miss")
        self.assertEqual(second["signal_metrics"]["ASR模型缓存"], "hit")

    def test_cuda_auto_fallback_reuses_cached_cpu_model(self) -> None:
        env = {
            "ASR_BACKEND": "faster-whisper",
            "ASR_MODEL": "tiny",
            "ASR_DEVICE": "auto",
            "ASR_COMPUTE_TYPE": "auto",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("app.services.asr_service._ctranslate2_cuda_available", return_value=True):
                with patch("app.services.asr_service._load_whisper_model_class", return_value=FailingCudaWhisperModel):
                    first = transcribe_audio(self.audio_path, "meeting.wav")
                    second = transcribe_audio(self.audio_path, "meeting.wav")

        self.assertEqual(FailingCudaWhisperModel.cpu_init_count, 1)
        self.assertEqual(FailingCudaWhisperModel.cuda_init_count, 2)
        self.assertEqual(first["signal_metrics"]["ASR 状态"], "success-cpu-fallback")
        self.assertEqual(second["signal_metrics"]["ASR模型缓存"], "hit")

    def test_placeholder_and_unsupported_backend_do_not_load_model(self) -> None:
        for backend in ["placeholder", "unknown"]:
            with patch.dict(os.environ, {"ASR_BACKEND": backend}, clear=True):
                with patch("app.services.asr_service._load_whisper_model_class") as load_mock:
                    result = transcribe_audio(self.audio_path, "meeting.wav")
            load_mock.assert_not_called()
            self.assertIn("signal_metrics", result)


def _write_wav(path: Path, seconds: int) -> None:
    sample_rate = 8000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * sample_rate * seconds)


if __name__ == "__main__":
    unittest.main()
