from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .audio_service import get_audio_duration_seconds


def build_meeting_analysis_metrics(
    *,
    audio_path: Path | None,
    transcript: list[dict],
    asr_metrics: dict[str, str] | None = None,
    separation: dict | None = None,
) -> dict[str, str]:
    duration = get_audio_duration_seconds(audio_path) if audio_path else None
    intervals = _transcript_intervals(transcript)
    speaker_stats = _speaker_stats(intervals, duration)
    speech_coverage = _coverage(intervals, duration)
    overlap_ratio = _overlap_ratio(intervals, duration)
    silent_ratio = _silent_ratio(audio_path)
    speaker_count = len(speaker_stats)
    quality_score = _quality_score(
        speech_coverage=speech_coverage,
        overlap_ratio=overlap_ratio,
        silent_ratio=silent_ratio,
        segment_count=len(transcript),
        speaker_count=speaker_count,
        asr_status=(asr_metrics or {}).get("ASR 状态") or (asr_metrics or {}).get("ASR 鐘舵€?") or "",
        separation_status=(separation or {}).get("status", ""),
    )

    route = _route_explanation(asr_metrics or {}, separation or {}, overlap_ratio)
    metrics = {
        "主处理后端": (asr_metrics or {}).get("主处理后端", _infer_backend(asr_metrics or {})),
        "检测说话人数": str(speaker_count or 1),
        "语音覆盖率": _percent(speech_coverage),
        "静音比例": _percent(silent_ratio),
        "疑似重叠比例": _percent(overlap_ratio),
        "会议提取质量评分": str(quality_score),
        "自适应路由说明": route,
        "说话人会议画像": _speaker_profile_text(speaker_stats, duration),
        "按说话人摘要": _speaker_summary_text(transcript),
    }
    if "中文ASR模型" not in metrics and (asr_metrics or {}).get("ASR 模型"):
        metrics["中文ASR模型"] = str((asr_metrics or {}).get("ASR 模型"))
    if "说话人分段模型" not in metrics and (asr_metrics or {}).get("说话人分段模型"):
        metrics["说话人分段模型"] = str((asr_metrics or {}).get("说话人分段模型"))
    return metrics


def _transcript_intervals(transcript: list[dict]) -> list[dict]:
    intervals = []
    for item in transcript or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _parse_timestamp(item.get("start"))
        end = _parse_timestamp(item.get("end"))
        if math.isnan(start):
            start = 0.0
        if math.isnan(end) or end < start:
            end = start
        intervals.append(
            {
                "start": start,
                "end": end,
                "speaker": str(item.get("speaker") or "说话人 A").strip() or "说话人 A",
                "text": text,
            }
        )
    return intervals


def _speaker_stats(intervals: list[dict], duration: float | None) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for item in intervals:
        speaker = item["speaker"]
        length = max(0.0, item["end"] - item["start"])
        current = stats.setdefault(speaker, {"duration": 0.0, "turns": 0.0, "longest": 0.0, "share": 0.0})
        current["duration"] += length
        current["turns"] += 1
        current["longest"] = max(current["longest"], length)
    total_duration = duration or sum(item["duration"] for item in stats.values()) or 1.0
    for item in stats.values():
        item["share"] = min(1.0, item["duration"] / total_duration)
    return stats


def _coverage(intervals: list[dict], duration: float | None) -> float:
    if not intervals:
        return 0.0
    merged = _merge_intervals([(item["start"], item["end"]) for item in intervals])
    speech_seconds = sum(max(0.0, end - start) for start, end in merged)
    if duration and duration > 0:
        return min(1.0, speech_seconds / duration)
    span = max((item["end"] for item in intervals), default=0.0)
    return min(1.0, speech_seconds / span) if span > 0 else 0.0


def _overlap_ratio(intervals: list[dict], duration: float | None) -> float:
    if len(intervals) < 2:
        return 0.0
    overlap = 0.0
    sorted_items = sorted(intervals, key=lambda item: (item["start"], item["end"]))
    for index, current in enumerate(sorted_items):
        for other in sorted_items[index + 1 :]:
            if other["start"] >= current["end"]:
                break
            if other["speaker"] == current["speaker"]:
                continue
            overlap += max(0.0, min(current["end"], other["end"]) - max(current["start"], other["start"]))
    denominator = duration or max((item["end"] for item in intervals), default=0.0)
    return min(1.0, overlap / denominator) if denominator > 0 else 0.0


def _silent_ratio(audio_path: Path | None) -> float:
    if audio_path is None or not audio_path.exists():
        return 0.0
    try:
        soundfile = __import__("soundfile")
        rms_values = []
        for block in soundfile.blocks(str(audio_path), blocksize=16000, always_2d=True, dtype="float32"):
            if len(block) == 0:
                continue
            channel = block[:, 0]
            rms = float(math.sqrt(float((channel * channel).mean())))
            rms_values.append(rms)
        if not rms_values:
            return 0.0
        threshold = max(0.004, sorted(rms_values)[max(0, int(len(rms_values) * 0.2) - 1)] * 1.4)
        return sum(1 for value in rms_values if value <= threshold) / len(rms_values)
    except Exception:
        return 0.0


def _quality_score(
    *,
    speech_coverage: float,
    overlap_ratio: float,
    silent_ratio: float,
    segment_count: int,
    speaker_count: int,
    asr_status: str,
    separation_status: str,
) -> int:
    score = 40
    score += int(speech_coverage * 25)
    score += 15 if segment_count >= 2 else 5 if segment_count == 1 else 0
    score += 10 if speaker_count >= 2 else 4 if speaker_count == 1 else 0
    score -= int(overlap_ratio * 12)
    score -= int(max(0.0, silent_ratio - 0.55) * 20)
    if "success" in asr_status:
        score += 10
    if "fallback" in separation_status or "placeholder" in separation_status:
        score -= 6
    return max(0, min(100, score))


def _route_explanation(asr_metrics: dict[str, str], separation: dict, overlap_ratio: float) -> str:
    backend = asr_metrics.get("ASR 后端") or asr_metrics.get("ASR 鍚庣") or "fallback"
    separation_status = separation.get("status", "unknown")
    separation_method = str(separation.get("method", "")).lower()
    if backend == "funasr":
        base = "中文会议主路径：DeepFilterNet增强后使用FunASR/SenseVoice+VAD+CAM++完成转写和说话人分段"
    elif backend == "faster-whisper":
        base = "FunASR不可用时自动回退到faster-whisper，保证演示链路不中断"
    else:
        base = "真实ASR不可用时使用演示兜底，保证页面和摘要流程稳定"
    if "speechbrain" in separation_method or "mossformer2" in separation_method or "clearvoice" in separation_method:
        separation_note = "当前使用真实盲源分离候选输出多轨音频"
    else:
        separation_note = "当前分离输出由质量路由或兜底策略决定"
    if overlap_ratio >= 0.08:
        return f"{base}；检测到较高重叠，{separation_note}；分离状态：{separation_status}"
    return f"{base}；{separation_note}；分离状态：{separation_status}"


def _speaker_profile_text(stats: dict[str, dict[str, float]], duration: float | None) -> str:
    if not stats:
        return "暂无有效说话人画像"
    parts = []
    for speaker, item in sorted(stats.items()):
        parts.append(
            f"{speaker}: {item['duration']:.0f}s/{int(item['turns'])}次/最长{item['longest']:.0f}s/占比{item['share']:.0%}"
        )
    return "；".join(parts[:6])


def _speaker_summary_text(transcript: list[dict]) -> str:
    grouped: dict[str, list[str]] = {}
    for item in transcript or []:
        speaker = str(item.get("speaker") or "说话人 A")
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if text:
            grouped.setdefault(speaker, []).append(text)
    if not grouped:
        return "暂无按说话人摘要"
    parts = []
    for speaker, texts in sorted(grouped.items()):
        joined = " ".join(texts)
        parts.append(f"{speaker}: {joined[:80]}{'...' if len(joined) > 80 else ''}")
    return "；".join(parts[:5])


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return math.nan
    try:
        parts = [float(part) for part in str(value).split(":")]
    except ValueError:
        return math.nan
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return math.nan


def _percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.0%}"


def _infer_backend(metrics: dict[str, str]) -> str:
    backend = metrics.get("ASR 后端") or metrics.get("ASR 鍚庣")
    if backend == "funasr":
        return "FunASR中文会议转写"
    if backend == "faster-whisper":
        return "faster-whisper回退转写"
    return "演示兜底"
