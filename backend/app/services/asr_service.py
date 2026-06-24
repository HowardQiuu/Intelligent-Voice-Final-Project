from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from .audio_service import UPLOAD_DIR, ffmpeg_executable, get_audio_duration_seconds


DEFAULT_ASR_BACKEND = "faster-whisper"
DEFAULT_ASR_MODEL = "small"
DEFAULT_ASR_DEVICE = "auto"
DEFAULT_ASR_COMPUTE_TYPE = "auto"
DEFAULT_ASR_LANGUAGE = "zh"
DEFAULT_ASR_MAX_SECONDS = 600.0
DEFAULT_ASR_CHUNK_SECONDS = 60.0
DEFAULT_ASR_MAX_CHUNKS = 240
_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_WHISPER_MODEL_CACHE_LOCK = threading.Lock()


def build_pipeline_steps(cache_mode: bool = True) -> list[dict]:
    source = "缓存结果" if cache_mode else "上传音频"
    return [
        {"key": "input", "name": "会议音频输入", "status": "done", "detail": f"已读取{source}"},
        {"key": "enhance", "name": "语音增强", "status": "done", "detail": "完成降噪、归一化与增强音频输出"},
        {"key": "chunking", "name": "分块处理", "status": "done", "detail": "按固定窗口规划长会议音频，避免一次性占满内存"},
        {"key": "separation", "name": "语音分离", "status": "done", "detail": "输出可替换的说话人语音轨道结果"},
        {"key": "asr", "name": "自动语音识别", "status": "done", "detail": "基于增强音频生成带时间戳的会议转写文本"},
        {"key": "summary", "name": "概要生成", "status": "done", "detail": "提取主题、关键词、决策与待办事项"},
    ]


def fallback_upload_result(filename: str) -> dict:
    return {
        "direct_asr_text": f"已接收上传文件：{filename}。当前使用演示兜底转写。",
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


def transcribe_audio(audio_path: Path | None, display_name: str, fallback: dict | None = None) -> dict:
    fallback_data = fallback or fallback_upload_result(display_name)
    backend = os.getenv("ASR_BACKEND", DEFAULT_ASR_BACKEND).strip().lower() or DEFAULT_ASR_BACKEND
    model_name = os.getenv("ASR_MODEL", DEFAULT_ASR_MODEL).strip() or DEFAULT_ASR_MODEL
    requested_device = os.getenv("ASR_DEVICE", DEFAULT_ASR_DEVICE).strip().lower() or DEFAULT_ASR_DEVICE
    requested_compute_type = (
        os.getenv("ASR_COMPUTE_TYPE", DEFAULT_ASR_COMPUTE_TYPE).strip().lower() or DEFAULT_ASR_COMPUTE_TYPE
    )
    device, compute_type = _resolve_asr_device_and_compute_type(requested_device, requested_compute_type)
    language = os.getenv("ASR_LANGUAGE", DEFAULT_ASR_LANGUAGE).strip() or DEFAULT_ASR_LANGUAGE
    max_seconds = _get_float_env("ASR_MAX_SECONDS", DEFAULT_ASR_MAX_SECONDS)
    chunk_seconds = _get_float_env("ASR_CHUNK_SECONDS", DEFAULT_ASR_CHUNK_SECONDS)
    vad_filter = _get_bool_env("ASR_VAD_FILTER", True)

    if backend in {"placeholder", "fallback", "disabled"}:
        return _fallback_with_metrics(
            fallback_data,
            backend="placeholder",
            model=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            status="placeholder",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
        )

    if backend != "faster-whisper":
        return _fallback_with_metrics(
            fallback_data,
            backend=backend,
            model=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            status=f"unsupported-backend:{backend}",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
        )

    if audio_path is None or not audio_path.exists():
        return _fallback_with_metrics(
            fallback_data,
            backend="faster-whisper",
            model=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            status="audio-not-found",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
        )

    duration = get_audio_duration_seconds(audio_path)
    if duration is None or duration <= 0:
        return _fallback_with_metrics(
            fallback_data,
            backend="faster-whisper",
            model=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            status="duration-unavailable",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
        )

    try:
        WhisperModel = _load_whisper_model_class()
    except Exception as exc:
        return _fallback_with_metrics(
            fallback_data,
            backend="faster-whisper",
            model=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            status=f"unavailable:{exc.__class__.__name__}",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
        )

    model_device = device
    model_compute_type = compute_type
    cache_status = "miss"
    try:
        model, model_device, model_compute_type, device_status_suffix, cache_status = _get_cached_whisper_model(
            WhisperModel=WhisperModel,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            requested_device=requested_device,
            requested_compute_type=requested_compute_type,
        )
        if duration > max_seconds:
            transcript, detected_language = _transcribe_audio_in_chunks(
                model=model,
                audio_path=audio_path,
                duration=duration,
                chunk_seconds=chunk_seconds,
                language=language,
                vad_filter=vad_filter,
            )
            status = f"success-chunked{device_status_suffix}"
        else:
            segments, info = model.transcribe(
                str(audio_path),
                language=language or None,
                vad_filter=vad_filter,
                word_timestamps=False,
            )
            transcript = [_segment_to_transcript(item) for item in segments]
            detected_language = getattr(info, "language", None) or language
            status = f"success{device_status_suffix}"
        transcript = [item for item in transcript if item["text"]]
    except Exception as exc:
        return _fallback_with_metrics(
            fallback_data,
            backend="faster-whisper",
            model=model_name,
            device=model_device,
            compute_type=model_compute_type,
            language=language,
            status=f"failed:{exc.__class__.__name__}",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
            cache_status=cache_status,
        )

    if not transcript:
        return _fallback_with_metrics(
            fallback_data,
            backend="faster-whisper",
            model=model_name,
            device=model_device,
            compute_type=model_compute_type,
            language=language,
            status="empty-result",
            segment_count=len(fallback_data.get("transcript", [])),
            chunk_seconds=chunk_seconds,
            cache_status=cache_status,
        )

    text = " ".join(item["text"] for item in transcript)
    return {
        "direct_asr_text": "原始音频未执行真实 ASR，当前展示增强后音频的本地 ASR 转写结果。",
        "enhanced_asr_text": text,
        "transcript": transcript,
        "signal_metrics": _asr_metrics(
            backend="faster-whisper",
            model=model_name,
            device=model_device,
            compute_type=model_compute_type,
            language=detected_language,
            status=status,
            segment_count=len(transcript),
            chunk_seconds=chunk_seconds,
            cache_status=cache_status,
        ),
    }


def _load_whisper_model_class():
    from faster_whisper import WhisperModel

    return WhisperModel


def _get_cached_whisper_model(
    *,
    WhisperModel: Any,
    model_name: str,
    device: str,
    compute_type: str,
    requested_device: str,
    requested_compute_type: str,
) -> tuple[Any, str, str, str, str]:
    try:
        model, cache_status = _load_cached_whisper_model(WhisperModel, model_name, device, compute_type)
        return model, device, compute_type, "", cache_status
    except Exception:
        if requested_device != "auto" or device == "cpu":
            raise
        fallback_device = "cpu"
        fallback_compute_type = "int8" if requested_compute_type == "auto" else compute_type
        model, cache_status = _load_cached_whisper_model(
            WhisperModel,
            model_name,
            fallback_device,
            fallback_compute_type,
        )
        return model, fallback_device, fallback_compute_type, "-cpu-fallback", cache_status


def _load_cached_whisper_model(
    WhisperModel: Any,
    model_name: str,
    device: str,
    compute_type: str,
) -> tuple[Any, str]:
    cache_key = (model_name, device, compute_type)
    with _WHISPER_MODEL_CACHE_LOCK:
        cached = _WHISPER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, "hit"
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _WHISPER_MODEL_CACHE[cache_key] = model
        return model, "miss"


def _resolve_asr_device_and_compute_type(requested_device: str, requested_compute_type: str) -> tuple[str, str]:
    device = requested_device
    if requested_device == "auto":
        device = "cuda" if _ctranslate2_cuda_available() else "cpu"

    compute_type = requested_compute_type
    if requested_compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def _ctranslate2_cuda_available() -> bool:
    try:
        ctranslate2 = __import__("ctranslate2")
        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


def _transcribe_audio_in_chunks(
    *,
    model: Any,
    audio_path: Path,
    duration: float,
    chunk_seconds: float,
    language: str,
    vad_filter: bool,
) -> tuple[list[dict], str]:
    chunk_count = int((duration + chunk_seconds - 0.001) // chunk_seconds)
    max_chunks = _get_int_env("ASR_MAX_CHUNKS", DEFAULT_ASR_MAX_CHUNKS)
    if chunk_count > max_chunks:
        raise RuntimeError(f"Audio requires {chunk_count} ASR chunks, over limit {max_chunks}")

    work_dir = UPLOAD_DIR / f"asr_chunks_{uuid.uuid4().hex[:10]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        chunks = _split_audio_to_chunks(audio_path, work_dir, chunk_seconds, duration)
        transcript: list[dict] = []
        detected_language = language
        for chunk_path, offset in chunks:
            segments, info = model.transcribe(
                str(chunk_path),
                language=language or None,
                vad_filter=vad_filter,
                word_timestamps=False,
            )
            detected_language = getattr(info, "language", None) or detected_language
            transcript.extend(_segment_to_transcript(item, offset_seconds=offset) for item in segments)
        return transcript, detected_language
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _split_audio_to_chunks(path: Path, work_dir: Path, chunk_seconds: float, duration: float) -> list[tuple[Path, float]]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for chunked ASR")

    chunks: list[tuple[Path, float]] = []
    start = 0.0
    index = 1
    while start < duration:
        chunk_duration = min(chunk_seconds, duration - start)
        chunk_path = work_dir / f"chunk_{index:04d}.wav"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{chunk_duration:.3f}",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to create ASR chunk {index}")
        chunks.append((chunk_path, start))
        start += chunk_seconds
        index += 1
    return chunks


def _segment_to_transcript(segment: Any, offset_seconds: float = 0.0) -> dict:
    start = offset_seconds + float(getattr(segment, "start", 0.0) or 0.0)
    end = offset_seconds + float(getattr(segment, "end", start - offset_seconds) or start - offset_seconds)
    text = str(getattr(segment, "text", "") or "").strip()
    return {
        "start": _format_seconds(start),
        "end": _format_seconds(end),
        "speaker": "说话人",
        "text": text,
    }


def _fallback_with_metrics(
    fallback: dict,
    *,
    backend: str,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    status: str,
    segment_count: int,
    chunk_seconds: float,
    cache_status: str = "miss",
) -> dict:
    return {
        **fallback,
        "signal_metrics": {
            **fallback.get("signal_metrics", {}),
            **_asr_metrics(
                backend=backend,
                model=model,
                device=device,
                compute_type=compute_type,
                language=language,
                status=status,
                segment_count=segment_count,
                chunk_seconds=chunk_seconds,
                cache_status=cache_status,
            ),
        },
    }


def _asr_metrics(
    *,
    backend: str,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    status: str,
    segment_count: int,
    chunk_seconds: float,
    cache_status: str = "miss",
) -> dict[str, str]:
    return {
        "ASR 后端": backend,
        "ASR 模型": model,
        "ASR 设备": f"{device}/{compute_type}",
        "ASR 状态": status,
        "ASR 语言": language or "auto",
        "ASR 分段数": str(segment_count),
        "ASR 分块窗口": f"{chunk_seconds:.0f}s",
        "ASR模型缓存": cache_status,
    }


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, value)


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _format_seconds(value: float) -> str:
    total = int(round(max(0.0, value)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
