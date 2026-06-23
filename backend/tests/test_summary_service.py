from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.summary_service import generate_summary  # noqa: E402


SAMPLE_TRANSCRIPT = [
    {"start": "00:00", "end": "00:10", "speaker": "说话人A", "text": "今天讨论摘要模块。"},
    {"start": "00:10", "end": "00:20", "speaker": "说话人B", "text": "需要接入大语言模型 API。"},
]

FALLBACK_SUMMARY = {
    "title": "缓存摘要",
    "keywords": ["缓存", "兜底"],
    "abstract": "这是缓存摘要。",
    "decisions": ["保留兜底逻辑"],
    "action_items": ["继续测试摘要模块"],
}


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


class FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.last_payload = None

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url, headers, json):
        self.last_payload = json
        content = {
            "title": "LLM 会议纪要",
            "keywords": ["摘要生成", "大语言模型"],
            "abstract": "会议讨论了摘要模块接入大语言模型 API，并确认使用结构化 JSON 输出。",
            "decisions": ["使用 OpenAI-Compatible 接口"],
            "action_items": ["验证前端展示效果"],
        }
        return FakeResponse(json_module.dumps(content, ensure_ascii=False))


json_module = json


class SummaryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()

    def test_fallback_without_api_key(self) -> None:
        result = generate_summary(
            transcript=SAMPLE_TRANSCRIPT,
            case_name="测试会议",
            enhanced_asr_text="今天讨论摘要模块，需要接入大语言模型 API。",
            fallback=FALLBACK_SUMMARY,
        )

        self.assertFalse(result.used_llm)
        self.assertEqual(result.summary["title"], "缓存摘要")
        self.assertEqual(result.metrics["摘要生成"], "缓存兜底")
        self.assertEqual(result.metrics["摘要状态"], "未配置 LLM_API_KEY")

    def test_llm_disabled_uses_fallback_even_with_api_key(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_ENABLED": "false"}, clear=True):
            result = generate_summary(
                transcript=SAMPLE_TRANSCRIPT,
                case_name="测试会议",
                enhanced_asr_text="今天讨论摘要模块。",
                fallback=FALLBACK_SUMMARY,
            )

        self.assertFalse(result.used_llm)
        self.assertEqual(result.summary["title"], "缓存摘要")
        self.assertEqual(result.metrics["摘要生成"], "缓存兜底")
        self.assertEqual(result.metrics["摘要状态"], "LLM_ENABLED=false")

    @patch("app.services.summary_service.httpx.Client", FakeClient)
    def test_llm_json_summary_success(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "test-key",
                "LLM_BASE_URL": "https://api.deepseek.com/v1",
                "LLM_MODEL": "deepseek-chat",
            },
            clear=True,
        ):
            result = generate_summary(
                transcript=SAMPLE_TRANSCRIPT,
                case_name="测试会议",
                enhanced_asr_text="今天讨论摘要模块，需要接入大语言模型 API。",
                fallback=FALLBACK_SUMMARY,
            )

        self.assertTrue(result.used_llm)
        self.assertEqual(result.summary["title"], "LLM 会议纪要")
        self.assertEqual(result.metrics["摘要生成"], "LLM API")
        self.assertEqual(result.metrics["摘要状态"], "结构化 JSON 生成成功")


if __name__ == "__main__":
    unittest.main()
