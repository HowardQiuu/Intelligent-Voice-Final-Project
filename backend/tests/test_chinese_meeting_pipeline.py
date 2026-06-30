from __future__ import annotations

import os
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_service import UPLOAD_DIR  # noqa: E402
from app.services.funasr_service import parse_funasr_sentence_info  # noqa: E402
from app.services.pipeline_analysis_service import build_meeting_analysis_metrics  # noqa: E402


class ChineseMeetingPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.env_patch.stop()
        for path in UPLOAD_DIR.glob("meeting_pipeline_test*"):
            path.unlink(missing_ok=True)

    def test_funasr_sentence_info_maps_stable_speakers(self) -> None:
        raw = [
            {
                "text": "项目讨论",
                "sentence_info": [
                    {"start": 0, "end": 1200, "speaker": "spk0", "text": "< | zh | >我们先看转写模块。"},
                    {"start": 1300, "end": 2600, "speaker": "spk1", "text": "这里需要说话人分段。"},
                    {"start": 3000, "end": 4200, "speaker": "spk0", "text": "然后生成会议纪要。"},
                ],
            }
        ]

        transcript = parse_funasr_sentence_info(raw, duration_seconds=5)

        self.assertEqual([item["speaker"] for item in transcript], ["说话人 A", "说话人 B", "说话人 A"])
        self.assertEqual(transcript[0]["start"], "00:00")
        self.assertEqual(transcript[1]["end"], "00:03")
        self.assertNotIn("<", transcript[0]["text"])
        self.assertNotIn("|", transcript[0]["text"])

    def test_funasr_vad_segment_sentence_key_maps_spk_ids(self) -> None:
        raw = [
            {
                "text": "full text fallback should not be used",
                "sentence_info": [
                    {"start": 0, "end": 6940, "spk": 0, "sentence": "<|zh|><|Speech|>first speaker"},
                    {"start": 6940, "end": 59970, "spk": 1, "sentence": "<|zh|><|Speech|>second speaker"},
                    {"start": 59970, "end": 73630, "spk": 3, "sentence": "<|zh|><|Speech|>third speaker"},
                ],
            }
        ]

        transcript = parse_funasr_sentence_info(raw, duration_seconds=74)

        self.assertEqual(len(transcript), 3)
        self.assertEqual(len({item["speaker"] for item in transcript}), 3)
        self.assertEqual(transcript[0]["start"], "00:00")
        self.assertEqual(transcript[2]["end"], "01:14")
        self.assertEqual(transcript[0]["text"], "first speaker")
        self.assertNotIn("full text fallback", " ".join(item["text"] for item in transcript))

    def test_funasr_without_sentence_info_falls_back_to_single_speaker(self) -> None:
        transcript = parse_funasr_sentence_info([{"text": "整段中文会议转写"}], duration_seconds=8)

        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["speaker"], "说话人 A")
        self.assertEqual(transcript[0]["end"], "00:08")

    def test_meeting_analysis_metrics_include_quality_and_route(self) -> None:
        source = _write_wav("meeting_pipeline_test_analysis.wav", seconds=6)
        transcript = [
            {"start": "00:00", "end": "00:03", "speaker": "说话人 A", "text": "讨论模型选择"},
            {"start": "00:02", "end": "00:05", "speaker": "说话人 B", "text": "补充创新方向"},
        ]

        metrics = build_meeting_analysis_metrics(
            audio_path=source,
            transcript=transcript,
            asr_metrics={"ASR 后端": "funasr", "ASR 状态": "success", "主处理后端": "FunASR中文会议转写"},
            separation={"status": "ok", "method": "SpeechBrain SepFormer"},
        )

        self.assertEqual(metrics["检测说话人数"], "2")
        self.assertIn("会议提取质量评分", metrics)
        self.assertIn("FunASR", metrics["自适应路由说明"])
        self.assertIn("说话人 A", metrics["说话人会议画像"])


def _write_wav(name: str, seconds: int) -> Path:
    path = UPLOAD_DIR / name
    sample_rate = 16000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frame = int(6000).to_bytes(2, "little", signed=True)
        wav.writeframes(frame * sample_rate * seconds)
    return path


if __name__ == "__main__":
    unittest.main()
