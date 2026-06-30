from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

from ..models import ProcessResult
from .asr_service import build_pipeline_steps, fallback_upload_result, transcribe_audio
from .audio_service import UPLOAD_DIR, audio_url, normalize_upload, resolve_static_url
from .chunking_service import build_chunk_plan
from .demo_cache import get_case, get_result
from .enhancement_service import enhance_demo_audio, enhance_uploaded_audio
from .pipeline_analysis_service import build_meeting_analysis_metrics
from .separation_alignment_service import (
    align_transcript_to_separation_tracks,
    alignment_metrics,
    build_textgrid_separation_evaluation,
    evaluation_metrics,
    should_transcribe_evaluation_tracks,
)
from .separation_service import separate_with_quality_router
from .summary_service import fallback_summary, generate_summary
from .transcript_topic_service import classify_transcript_topics
from .visualization_service import generate_enhancement_visual


UPLOAD_CHUNK_BYTES = 1024 * 1024
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


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


def process_demo_case(case_id: str, *, processing_mode: str = "fast") -> ProcessResult:
    case = get_case(case_id)
    if case.get("audio_path"):
        source_path = _resolve_repo_audio_path(case["audio_path"])
        if not source_path.exists():
            raise KeyError(case_id)
        staged_path = stage_local_file(source_path)
        return process_audio_path(
            staged_path,
            source_path.name,
            case_id=case_id,
            reference_audio_path=source_path,
            processing_mode=processing_mode,
        )

    cached = get_result(case_id)

    audio = enhance_demo_audio(case_id)
    original_path = resolve_static_url(audio["original_audio_url"])
    enhanced_path = resolve_static_url(audio["enhanced_audio_url"])
    chunk_plan = build_chunk_plan(enhanced_path)
    visual_url, visual_metrics = generate_enhancement_visual(original_path, enhanced_path, case_id)
    separation = separate_with_quality_router(audio["enhanced_audio_url"], cached["transcript"])
    aligned_transcript, separation_alignment = align_transcript_to_separation_tracks(
        cached["transcript"],
        separation["tracks"],
    )
    summary_result = generate_summary(
        transcript=aligned_transcript,
        case_name=case["name"],
        enhanced_asr_text=cached["enhanced_asr_text"],
        fallback=cached["summary"],
    )
    topic_result = classify_transcript_topics(aligned_transcript, case["name"])
    separation_evaluation = build_textgrid_separation_evaluation(
        separated_tracks=separation["tracks"],
        display_name=case["name"],
        transcribe_track=_evaluation_transcriber(display_name=case["name"]),
    )
    analysis_metrics = build_meeting_analysis_metrics(
        audio_path=enhanced_path,
        transcript=aligned_transcript,
        asr_metrics=cached.get("signal_metrics", {}),
        separation=separation,
    )
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
        **analysis_metrics,
        **alignment_metrics(separation_alignment),
        **evaluation_metrics(separation_evaluation),
        **separation.get("metrics", {}),
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
        speaker_count_estimation=separation.get("speaker_count_estimation", {}),
        steps=build_pipeline_steps(cache_mode=True),
        transcript=aligned_transcript,
        transcript_topics=topic_result.topics,
        separation_alignment=separation_alignment,
        separation_evaluation=separation_evaluation,
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


def process_audio_path(
    raw_path: Path,
    display_name: str,
    case_id: str,
    *,
    reference_audio_path: Path | None = None,
    processing_mode: str = "fast",
) -> ProcessResult:
    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()
    if _use_fast_separation_path(processing_mode):
        return _process_fast_quality_router(raw_path, display_name, case_id, pipeline_start)

    stage_start = time.perf_counter()
    normalized = normalize_upload(raw_path, raw_path.stem)
    timings["runtime_normalize_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    chunk_plan = build_chunk_plan(normalized)
    timings["runtime_chunk_plan_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    try:
        audio = enhance_uploaded_audio(normalized)
    except RuntimeError as exc:
        audio = {
            "original_audio_url": audio_url(normalized),
            "enhanced_audio_url": audio_url(normalized),
            "metrics": _fallback_loudness_metrics(),
            "method": f"上传增强兜底：{_compact_error(exc)}",
        }
    timings["runtime_enhancement_seconds"] = time.perf_counter() - stage_start
    original_path = resolve_static_url(audio["original_audio_url"])
    enhanced_path = resolve_static_url(audio["enhanced_audio_url"])

    stage_start = time.perf_counter()
    visual_url, visual_metrics = generate_enhancement_visual(original_path, enhanced_path, raw_path.stem)
    timings["runtime_visual_seconds"] = time.perf_counter() - stage_start

    fallback = fallback_upload_result(display_name)
    stage_start = time.perf_counter()
    asr_result = transcribe_audio(enhanced_path, display_name, fallback=fallback)
    timings["runtime_asr_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    separation_audio_url, separation_input_source = _select_separation_audio_url(audio, normalized, raw_path=raw_path)
    separation_reference_path = reference_audio_path or raw_path
    separation = separate_with_quality_router(
        separation_audio_url,
        asr_result["transcript"],
        reference_audio_path=separation_reference_path,
        display_name=display_name,
    )
    timings["runtime_separation_seconds"] = time.perf_counter() - stage_start
    aligned_transcript, separation_alignment = align_transcript_to_separation_tracks(
        asr_result["transcript"],
        separation["tracks"],
    )
    separation_evaluation = build_textgrid_separation_evaluation(
        separated_tracks=separation["tracks"],
        display_name=display_name,
        reference_audio_path=separation_reference_path,
        transcribe_track=_evaluation_transcriber(reference_audio_path=separation_reference_path, display_name=display_name),
    )

    analysis_metrics = build_meeting_analysis_metrics(
        audio_path=enhanced_path,
        transcript=aligned_transcript,
        asr_metrics=asr_result.get("signal_metrics", {}),
        separation=separation,
    )
    signal_metrics = {
        **asr_result["signal_metrics"],
        "增强算法": audio["method"],
        "分离算法": separation["method"],
        "分离状态": separation["status"],
        "分离轨道数": separation["track_count"],
        "分块处理": chunk_plan["summary"],
        "分块数量": chunk_plan["chunk_count"],
        "separation_input_source": separation_input_source,
        "processing_mode": _normalize_processing_mode(processing_mode),
        **audio.get("metrics", {}),
        **visual_metrics,
        **analysis_metrics,
        **alignment_metrics(separation_alignment),
        **evaluation_metrics(separation_evaluation),
        **separation.get("metrics", {}),
    }
    stage_start = time.perf_counter()
    summary_result = generate_summary(
        transcript=aligned_transcript,
        case_name=display_name or "上传会议音频",
        enhanced_asr_text=asr_result["enhanced_asr_text"],
        fallback=fallback_summary(),
    )
    timings["runtime_summary_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    topic_result = classify_transcript_topics(aligned_transcript, display_name or "上传会议音频")
    timings["runtime_topic_seconds"] = time.perf_counter() - stage_start
    timings["runtime_total_seconds"] = time.perf_counter() - pipeline_start
    signal_metrics = {
        **signal_metrics,
        **topic_result.metrics,
        **summary_result.metrics,
        **_format_runtime_metrics(timings),
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
        speaker_count_estimation=separation.get("speaker_count_estimation", {}),
        steps=build_pipeline_steps(cache_mode=False),
        transcript=aligned_transcript,
        transcript_topics=topic_result.topics,
        separation_alignment=separation_alignment,
        separation_evaluation=separation_evaluation,
        summary=summary_result.summary,
    )


def _process_fast_quality_router(
    raw_path: Path,
    display_name: str,
    case_id: str,
    pipeline_start: float,
) -> ProcessResult:
    source_url = audio_url(raw_path)
    separation = separate_with_quality_router(
        source_url,
        [],
        reference_audio_path=raw_path,
        display_name=display_name,
    )

    chunk_plan = build_chunk_plan(raw_path)
    signal_metrics = {
        "分离算法": separation["method"],
        "分离状态": separation["status"],
        "分离轨道数": separation["track_count"],
        "分块处理": chunk_plan["summary"],
        "分块数量": chunk_plan["chunk_count"],
        "separation_input_source": "raw",
        "fast_path_mode": "quality-router-separation-only",
        "processing_mode": "fast",
        "runtime_total_seconds": f"{time.perf_counter() - pipeline_start:.2f}",
        **separation.get("metrics", {}),
    }
    return ProcessResult(
        case_id=case_id,
        case_name=display_name or raw_path.name,
        original_audio_url=source_url,
        enhanced_audio_url=source_url,
        enhancement_visual_url=None,
        processing_chunks=chunk_plan["chunks"],
        separated_tracks=separation["tracks"],
        direct_asr_text="Fast separation path: ASR skipped; quality router model separation only.",
        enhanced_asr_text="Fast separation path: ASR skipped; quality router model separation only.",
        signal_metrics=signal_metrics,
        speaker_count_estimation=separation.get("speaker_count_estimation", {}),
        steps=build_pipeline_steps(cache_mode=False),
        transcript=[],
        transcript_topics=[],
        separation_alignment={},
        separation_evaluation={},
        summary=fallback_summary(),
    )


def _resolve_repo_audio_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_DIR / path


def _normalize_processing_mode(value: str | None) -> str:
    mode = (value or "fast").strip().lower()
    if mode in {"full", "complete", "pipeline"}:
        return "full"
    return "fast"


def _use_fast_separation_path(value: str | None) -> bool:
    return _normalize_processing_mode(value) == "fast"


def separate_uploaded_path(raw_path: Path, display_name: str) -> dict:
    normalized = normalize_upload(raw_path, raw_path.stem)
    if _skip_separate_upload_enhancement():
        audio = {
            "original_audio_url": audio_url(normalized),
            "enhanced_audio_url": audio_url(normalized),
            "metrics": _fallback_loudness_metrics(),
            "method": "separate-upload enhancement skipped",
        }
        separation_audio_url, separation_input_source = _select_separation_audio_url(audio, normalized, raw_path=raw_path)
        separation = separate_with_quality_router(
            separation_audio_url,
            [],
            reference_audio_path=raw_path,
            display_name=display_name,
        )
        return {
            "file_name": display_name,
            "original_audio_url": audio["original_audio_url"],
            "enhanced_audio_url": audio["enhanced_audio_url"],
            "separation_input_source": separation_input_source,
            "enhancement_method": audio["method"],
            "separation": separation,
        }
    try:
        audio = enhance_uploaded_audio(normalized)
    except RuntimeError as exc:
        audio = {
            "original_audio_url": audio_url(normalized),
            "enhanced_audio_url": audio_url(normalized),
            "metrics": _fallback_loudness_metrics(),
            "method": f"上传增强兜底：{_compact_error(exc)}",
        }
    separation_audio_url, separation_input_source = _select_separation_audio_url(audio, normalized, raw_path=raw_path)
    separation = separate_with_quality_router(
        separation_audio_url,
        [],
        reference_audio_path=raw_path,
        display_name=display_name,
    )
    return {
        "file_name": display_name,
        "original_audio_url": audio["original_audio_url"],
        "enhanced_audio_url": audio["enhanced_audio_url"],
        "separation_input_source": separation_input_source,
        "enhancement_method": audio["method"],
        "separation": separation,
    }


def _skip_separate_upload_enhancement() -> bool:
    return os.getenv("SEPARATE_UPLOAD_SKIP_ENHANCEMENT", "false").strip().lower() in {"1", "true", "yes", "on"}


def _select_separation_audio_url(audio: dict, normalized_path: Path, raw_path: Path | None = None) -> tuple[str, str]:
    default_source = "raw" if raw_path is not None else "normalized"
    source = os.getenv("SEPARATION_INPUT_SOURCE", default_source).strip().lower() or default_source
    if source in {"enhanced", "enhanced_audio", "denoised"}:
        return audio["enhanced_audio_url"], "enhanced"
    if source in {"raw", "original", "mix"} and raw_path is not None and raw_path.exists():
        return audio_url(stage_local_file(raw_path)), "raw"
    return audio_url(normalized_path), "normalized"


def _evaluation_transcriber(reference_audio_path: Path | None = None, display_name: str = ""):
    if not should_transcribe_evaluation_tracks(reference_audio_path, display_name):
        return None
    return lambda path, display_name: transcribe_audio(path, display_name, fallback=fallback_upload_result(display_name))


def _fallback_loudness_metrics() -> dict[str, str]:
    return {
        "响度预处理": "highpass + loudnorm(-20 LUFS) + limiter",
        "增强后响度处理": "skipped",
        "响度处理状态": "fallback",
    }


def _format_runtime_metrics(timings: dict[str, float]) -> dict[str, str]:
    return {key: f"{value:.2f}" for key, value in timings.items()}
