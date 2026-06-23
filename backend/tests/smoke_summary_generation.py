from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.summary_service import generate_summary  # noqa: E402


def main() -> None:
    transcript = [
        {
            "start": "00:00",
            "end": "00:15",
            "speaker": "说话人A",
            "text": "今天主要讨论智能会议系统的摘要生成模块。",
        },
        {
            "start": "00:15",
            "end": "00:35",
            "speaker": "说话人B",
            "text": "我们已经接入 DeepSeek 兼容接口，希望输出主题、关键词、决策和待办事项。",
        },
        {
            "start": "00:35",
            "end": "00:50",
            "speaker": "说话人A",
            "text": "后续需要在前端展示模型调用状态，并保留缓存兜底机制。",
        },
    ]
    fallback = {
        "title": "摘要生成模块测试",
        "keywords": ["摘要生成", "兜底"],
        "abstract": "这是摘要生成模块的兜底测试结果。",
        "decisions": ["保留缓存兜底"],
        "action_items": ["继续验证 LLM API"],
    }

    result = generate_summary(
        transcript=transcript,
        case_name="摘要生成模块测试会议",
        enhanced_asr_text="今天讨论智能会议系统摘要生成模块，接入 DeepSeek 兼容接口，并展示模型调用状态。",
        fallback=fallback,
    )

    print("摘要来源:", result.metrics.get("摘要生成"))
    print("摘要模型:", result.metrics.get("摘要模型"))
    print("摘要状态:", result.metrics.get("摘要状态"))
    print("会议标题:", result.summary["title"])
    print("关键词:", "、".join(result.summary["keywords"]))
    print("摘要:", result.summary["abstract"])
    print("关键决策:", "；".join(result.summary["decisions"]))
    print("待办事项:", "；".join(result.summary["action_items"]))


if __name__ == "__main__":
    main()
