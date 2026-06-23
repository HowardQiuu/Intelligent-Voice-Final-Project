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
from app.services.chunking_service import build_chunk_plan  # noqa: E402
from app.services.visualization_service import generate_enhancement_visual  # noqa: E402


class ChunkingVisualizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()

    def test_chunk_plan_for_long_audio(self) -> None:
        path = _write_wav("chunk_plan.wav", seconds=120)
        try:
            with patch.dict(os.environ, {"CHUNK_SECONDS": "60", "CHUNK_OVERLAP_SECONDS": "5"}, clear=True):
                plan = build_chunk_plan(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(plan["chunk_count"], "3")
        self.assertEqual(plan["chunks"][1]["start"], "00:55")

    def test_enhancement_visual_generates_svg(self) -> None:
        original = _write_wav("visual_original.wav", seconds=2, amplitude=6000)
        enhanced = _write_wav("visual_enhanced.wav", seconds=2, amplitude=3000)
        try:
            url, metrics = generate_enhancement_visual(original, enhanced, "test")
            output_path = UPLOAD_DIR / Path(url).name
            self.assertTrue(output_path.exists())
            self.assertEqual(metrics["增强可视化"], "已生成波形/能量对比图")
            self.assertIn("平均能量变化", metrics)
        finally:
            original.unlink(missing_ok=True)
            enhanced.unlink(missing_ok=True)
            if url:
                (UPLOAD_DIR / Path(url).name).unlink(missing_ok=True)


def _write_wav(name: str, seconds: int, amplitude: int = 1000) -> Path:
    path = UPLOAD_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frame = int(amplitude).to_bytes(2, "little", signed=True)
        wav.writeframes(frame * sample_rate * seconds)
    return path


if __name__ == "__main__":
    unittest.main()
