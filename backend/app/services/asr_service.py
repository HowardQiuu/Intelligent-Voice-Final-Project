from __future__ import annotations


def build_pipeline_steps(cache_mode: bool = True) -> list[dict]:
    source = "缓存结果" if cache_mode else "上传音频"
    return [
        {"key": "input", "name": "会议音频输入", "status": "done", "detail": f"已读取{source}"},
        {"key": "enhance", "name": "语音增强", "status": "done", "detail": "完成降噪、归一化与增强音频输出"},
        {"key": "diarization", "name": "说话人处理", "status": "done", "detail": "使用演示分段结果展示说话人标签"},
        {"key": "asr", "name": "自动语音识别", "status": "done", "detail": "生成带时间戳的会议转写文本"},
        {"key": "summary", "name": "概要生成", "status": "done", "detail": "提取主题、关键词、决策与待办事项"},
    ]


def fallback_upload_result(filename: str) -> dict:
    return {
        "direct_asr_text": f"已接收上传文件：{filename}。当前演示模式未强制调用本地 ASR 模型。",
        "enhanced_asr_text": "系统已完成音频预处理，并使用演示兜底结果展示转写与摘要流程。",
        "signal_metrics": {
            "处理模式": "上传演示模式",
            "增强策略": "格式转换 / 音量归一化 / 缓存兜底",
            "模型状态": "可后续接入 faster-whisper 或云端 ASR",
        },
        "transcript": [
            {"start": "00:00", "end": "00:15", "speaker": "说话人A", "text": "这里展示上传音频后的转写结果占位。"},
            {"start": "00:15", "end": "00:30", "speaker": "说话人B", "text": "后续可以接入 Whisper 或其他中文会议转写模型。"},
        ],
    }
