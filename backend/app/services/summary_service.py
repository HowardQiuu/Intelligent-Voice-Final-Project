from __future__ import annotations


def fallback_summary() -> dict:
    return {
        "title": "上传会议音频演示结果",
        "keywords": ["音频上传", "语音增强", "ASR 接入", "会议纪要"],
        "abstract": "系统已完成上传音频的接收和预处理，并通过兜底数据展示从音频到会议纪要的完整链路。",
        "decisions": ["当前版本优先保证课堂演示稳定", "后续可接入真实 ASR 与摘要 API"],
        "action_items": ["准备真实会议样例", "接入 faster-whisper", "根据课程展示需要优化前端效果"],
    }
