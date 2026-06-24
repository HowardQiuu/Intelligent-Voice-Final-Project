from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from ..models import ProcessResult
from .asr_service import build_pipeline_steps, fallback_upload_result, transcribe_audio
from .audio_service import UPLOAD_DIR, audio_url, normalize_upload, resolve_static_url
from .chunking_service import build_chunk_plan
from .demo_cache import get_case, get_result
from .enhancement_service import enhance_demo_audio, enhance_uploaded_audio
from .separation_service import separate_demo_audio, separate_uploaded_audio
from .summary_service import fallback_summary, generate_summary
from .transcript_topic_service import classify_transcript_topics
from .visualization_service import generate_enhancement_visual


UPLOAD_CHUNK_BYTES = 1024 * 1024


def _compact_error(exc: Exception, limit: int = 120) -> str:
    message = re.sub(r"\s+", " ", str(exc)).strip()
    if "DeepFilterNet" in message or "deepFilter" in message:
        return "DeepFilterNet 处理失败，已使用归一化音频继续流程"
    if "returned non-zero exit status" in message:
        return "外部命令执行失败，已使用兜底音频继续流程"
    message = re.sub(r"[A-Za-z]:\\[^'\" ]+", "<path>", message)
    if len(message) > limit:
        return f"{message[: limit - 3]}..."
    return message


def process_demo_case(case_id: str) -> ProcessResult:
    case = get_case(case_id)
    cached = get_result(case_id)

    audio = enhance_demo_audio(case_id)
    original_path = resolve_static_url(audio["original_audio_url"])
    enhanced_path = resolve_static_url(audio["enhanced_audio_url"])
    chunk_plan = build_chunk_plan(enhanced_path)
    visual_url, visual_metrics = generate_enhancement_visual(original_path, enhanced_path, case_id)
    separation = separate_demo_audio(case_id, audio["enhanced_audio_url"])
    summary_result = generate_summary(
        transcript=cached["transcript"],
        case_name=case["name"],
        enhanced_asr_text=cached["enhanced_asr_text"],
        fallback=cached["summary"],
    )
    topic_result = classify_transcript_topics(cached["transcript"], case["name"])
    signal_metrics = {
        **cached["signal_metrics"],
        "分离算法": separation["method"],
        "分离状态": separation["status"],
        "分离轨道数": separation["track_count"],
        "分块处理": chunk_plan["summary"],
        "分块数量": chunk_plan["chunk_count"],
        **visual_metrics,
        **topic_result.metrics,
        **summary_result.metrics,
    }
    return ProcessResult(
        case_id=case_id,
        case_name=case["name"],
        original_audio_url=audio["original_audio_url"],
        enhanced_audio_url=audio["enhanced_audio_url"],
        enhancement_visual_url=visual_url,
        processing_chunks=chunk_plan["chunks"],
        separated_tracks=separation["tracks"],
        direct_asr_text=cached["direct_asr_text"],
        enhanced_asr_text=cached["enhanced_asr_text"],
        signal_metrics=signal_metrics,
        steps=build_pipeline_steps(cache_mode=True),
        transcript=cached["transcript"],
        transcript_topics=topic_result.topics,
        summary=summary_result.summary,
    )


async def save_upload_stream(file: UploadFile, destination: Path) -> None:
    with destination.open("wb") as f:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            f.write(chunk)


def stage_local_file(source_path: Path) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    staged_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{source_path.suffix.lower()}"
    try:
        os.link(source_path, staged_path)
    except OSError:
        shutil.copyfile(source_path, staged_path)
    return staged_path


def process_audio_path(raw_path: Path, display_name: str, case_id: str) -> ProcessResult:
    normalized = normalize_upload(raw_path, raw_path.stem)
    chunk_plan = build_chunk_plan(normalized)
    try:
        audio = enhance_uploaded_audio(normalized)
    except RuntimeError as exc:
        audio = {
            "original_audio_url": audio_url(normalized),
            "enhanced_audio_url": audio_url(normalized),
            "metrics": _fallback_loudness_metrics(),
            "method": f"上传增强兜底：{_compact_error(exc)}",
        }
    original_path = resolve_static_url(audio["original_audio_url"])
    enhanced_path = resolve_static_url(audio["enhanced_audio_url"])
    visual_url, visual_metrics = generate_enhancement_visual(original_path, enhanced_path, raw_path.stem)
    separation = separate_uploaded_audio(audio["enhanced_audio_url"])

    fallback = fallback_upload_result(display_name)
    asr_result = transcribe_audio(enhanced_path, display_name, fallback=fallback)
    signal_metrics = {
        **asr_result["signal_metrics"],
        "增强算法": audio["method"],
        "分离算法": separation["method"],
        "分离状态": separation["status"],
        "分离轨道数": separation["track_count"],
        "分块处理": chunk_plan["summary"],
        "分块数量": chunk_plan["chunk_count"],
        **audio.get("metrics", {}),
        **visual_metrics,
    }
    summary_result = generate_summary(
        transcript=asr_result["transcript"],
        case_name=display_name or "上传会议音频",
        enhanced_asr_text=asr_result["enhanced_asr_text"],
        fallback=fallback_summary(),
    )
    topic_result = classify_transcript_topics(asr_result["transcript"], display_name or "上传会议音频")
    signal_metrics = {
        **signal_metrics,
        **topic_result.metrics,
        **summary_result.metrics,
    }
    return ProcessResult(
        case_id=case_id,
        case_name=display_name or "上传会议音频",
        original_audio_url=audio["original_audio_url"],
        enhanced_audio_url=audio["enhanced_audio_url"],
        enhancement_visual_url=visual_url,
        processing_chunks=chunk_plan["chunks"],
        separated_tracks=separation["tracks"],
        direct_asr_text=asr_result["direct_asr_text"],
        enhanced_asr_text=asr_result["enhanced_asr_text"],
        signal_metrics=signal_metrics,
        steps=build_pipeline_steps(cache_mode=False),
        transcript=asr_result["transcript"],
        transcript_topics=topic_result.topics,
        summary=summary_result.summary,
    )


def separate_uploaded_path(raw_path: Path, display_name: str) -> dict:
    normalized = normalize_upload(raw_path, raw_path.stem)
    try:
        audio = enhance_uploaded_audio(normalized)
    except RuntimeError as exc:
        audio = {
            "original_audio_url": audio_url(normalized),
            "enhanced_audio_url": audio_url(normalized),
            "metrics": _fallback_loudness_metrics(),
            "method": f"上传增强兜底：{_compact_error(exc)}",
        }
    separation = separate_uploaded_audio(audio["enhanced_audio_url"])
    return {
        "file_name": display_name,
        "original_audio_url": audio["original_audio_url"],
        "enhanced_audio_url": audio["enhanced_audio_url"],
        "enhancement_method": audio["method"],
        "separation": separation,
    }


def _fallback_loudness_metrics() -> dict[str, str]:
    return {
        "响度预处理": "highpass + loudnorm(-20 LUFS) + limiter",
        "增强后响度处理": "skipped",
        "响度处理状态": "fallback",
    }
