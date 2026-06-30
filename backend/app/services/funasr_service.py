from __future__ import annotations

import importlib
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from .audio_service import get_audio_duration_seconds


DEFAULT_FUNASR_MODEL = "iic/SenseVoiceSmall"
DEFAULT_FUNASR_VAD_MODEL = "fsmn-vad"
DEFAULT_FUNASR_PUNC_MODEL = "ct-punc"
DEFAULT_FUNASR_SPK_MODEL = "cam++"
DEFAULT_FUNASR_SPK_MODE = "vad_segment"
DEFAULT_FUNASR_DEVICE = "cuda"
DEFAULT_FUNASR_BATCH_SIZE_S = 60
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELSCOPE_MODEL_ALIASES = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
}
_FUNASR_MODEL_CACHE: dict[tuple[str, str, str, str, str, str], Any] = {}
_FUNASR_MODEL_CACHE_LOCK = threading.Lock()


def transcribe_audio_with_funasr(audio_path: Path, display_name: str) -> dict:
    if audio_path is None or not audio_path.exists():
        raise FileNotFoundError("audio file not found for FunASR")

    model_name = _env("FUNASR_MODEL", _env("ASR_MODEL", DEFAULT_FUNASR_MODEL))
    vad_model = _env("FUNASR_VAD_MODEL", DEFAULT_FUNASR_VAD_MODEL)
    punc_model = _env("FUNASR_PUNC_MODEL", DEFAULT_FUNASR_PUNC_MODEL)
    spk_model = _env("FUNASR_SPK_MODEL", DEFAULT_FUNASR_SPK_MODEL)
    spk_mode = _env("FUNASR_SPK_MODE", DEFAULT_FUNASR_SPK_MODE)
    requested_device = _env("FUNASR_DEVICE", _env("ASR_DEVICE", DEFAULT_FUNASR_DEVICE)).lower()
    language = _env("ASR_LANGUAGE", "zh")
    batch_size_s = _int_env("FUNASR_BATCH_SIZE_S", DEFAULT_FUNASR_BATCH_SIZE_S)

    automodel_class = _load_automodel_class()
    model, device, cache_status = _get_cached_funasr_model(
        automodel_class=automodel_class,
        model_name=model_name,
        vad_model=vad_model,
        punc_model=punc_model,
        spk_model=spk_model,
        spk_mode=spk_mode,
        requested_device=requested_device,
    )

    try:
        raw_result = _generate(
            model,
            audio_path,
            language=language,
            batch_size_s=batch_size_s,
        )
    except Exception:
        if device == "cpu":
            raise
        model, device, cache_status = _get_cached_funasr_model(
            automodel_class=automodel_class,
            model_name=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            spk_model=spk_model,
            spk_mode=spk_mode,
            requested_device="cpu",
        )
        raw_result = _generate(
            model,
            audio_path,
            language=language,
            batch_size_s=batch_size_s,
        )
        cache_status = f"{cache_status}-cpu-fallback"

    duration = get_audio_duration_seconds(audio_path) or 0.0
    parsed = _parse_funasr_result(raw_result, duration)
    transcript = parsed["transcript"]
    text = " ".join(item["text"] for item in transcript if item.get("text")).strip()
    if not transcript or not text:
        raise RuntimeError("FunASR returned no transcript")

    return {
        "direct_asr_text": "原始音频未单独执行 ASR；当前展示增强后音频的中文会议转写结果。",
        "enhanced_asr_text": text,
        "transcript": transcript,
        "signal_metrics": _funasr_metrics(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            spk_model=spk_model,
            spk_mode=spk_mode,
            device=device,
            language=language,
            status="success",
            segment_count=len(transcript),
            speaker_count=len({item["speaker"] for item in transcript}),
            cache_status=cache_status,
            events=parsed["events"],
        ),
    }


def parse_funasr_sentence_info(raw_result: Any, duration_seconds: float = 0.0) -> list[dict]:
    """Small public helper used by tests and by future adapters."""
    return _parse_funasr_result(raw_result, duration_seconds)["transcript"]


def _load_automodel_class() -> Any:
    _ensure_modelscope_runtime()
    _ensure_ffmpeg_on_path()
    module = importlib.import_module("funasr")
    if hasattr(module, "AutoModel"):
        return getattr(module, "AutoModel")
    auto_model_module = importlib.import_module("funasr.auto.auto_model")
    return getattr(auto_model_module, "AutoModel")


def _ensure_modelscope_runtime() -> None:
    os.environ.setdefault("MODELSCOPE_HUB_FILE_LOCK", _env("FUNASR_MODELSCOPE_FILE_LOCK", "false"))
    cache_dir = os.getenv("FUNASR_MODELSCOPE_CACHE", "").strip() or str(BACKEND_DIR / ".runtime" / "modelscope")
    resolved = Path(cache_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(resolved))


def _ensure_ffmpeg_on_path() -> None:
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        source_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    except Exception:
        return
    if not source_path.exists():
        return
    shim_dir = BACKEND_DIR / ".runtime" / "bin"
    shim_path = shim_dir / "ffmpeg.exe"
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
        if not shim_path.exists() or shim_path.stat().st_size != source_path.stat().st_size:
            shutil.copyfile(source_path, shim_path)
        ffmpeg_dir = str(shim_dir)
    except OSError:
        ffmpeg_dir = str(source_path.parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if path_parts and path_parts[0].lower() == ffmpeg_dir.lower():
        return
    os.environ["PATH"] = os.pathsep.join([ffmpeg_dir, *path_parts])


def _get_cached_funasr_model(
    *,
    automodel_class: Any,
    model_name: str,
    vad_model: str,
    punc_model: str,
    spk_model: str,
    spk_mode: str,
    requested_device: str,
) -> tuple[Any, str, str]:
    device = _resolve_device(requested_device)
    cache_key = (model_name, vad_model, punc_model, spk_model, spk_mode, device)
    with _FUNASR_MODEL_CACHE_LOCK:
        cached = _FUNASR_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, device, "hit"
        try:
            model = _build_automodel(automodel_class, model_name, vad_model, punc_model, spk_model, spk_mode, device)
        except Exception:
            if device == "cpu":
                raise
            device = "cpu"
            cache_key = (model_name, vad_model, punc_model, spk_model, spk_mode, device)
            cached = _FUNASR_MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached, device, "hit-cpu-fallback"
            model = _build_automodel(automodel_class, model_name, vad_model, punc_model, spk_model, spk_mode, device)
            _FUNASR_MODEL_CACHE[cache_key] = model
            return model, device, "miss-cpu-fallback"
        _FUNASR_MODEL_CACHE[cache_key] = model
        return model, device, "miss"


def _build_automodel(
    automodel_class: Any,
    model_name: str,
    vad_model: str,
    punc_model: str,
    spk_model: str,
    spk_mode: str,
    device: str,
) -> Any:
    kwargs = {
        "model": _resolve_local_model_reference(model_name),
        "vad_model": _resolve_local_model_reference(vad_model),
        "punc_model": _resolve_local_model_reference(punc_model),
        "spk_model": _resolve_local_model_reference(spk_model),
        "spk_mode": spk_mode,
        "device": device,
        "disable_update": True,
        "disable_pbar": True,
    }
    return automodel_class(**{key: value for key, value in kwargs.items() if value})


def _resolve_local_model_reference(model_ref: str) -> str:
    normalized = MODELSCOPE_MODEL_ALIASES.get(model_ref, model_ref)
    if os.path.exists(normalized):
        return normalized
    if "/" not in normalized:
        return model_ref
    owner, name = normalized.split("/", 1)
    cache_root = Path(os.environ.get("MODELSCOPE_CACHE", BACKEND_DIR / ".runtime" / "modelscope"))
    candidate = cache_root / "models" / owner / name
    if candidate.exists():
        return str(candidate)
    legacy_candidate = cache_root / owner / name
    if legacy_candidate.exists():
        return str(legacy_candidate)
    return model_ref


def _generate(model: Any, audio_path: Path, *, language: str, batch_size_s: int) -> Any:
    kwargs = {
        "input": str(audio_path),
        "cache": {},
        "language": language,
        "use_itn": True,
        "batch_size_s": batch_size_s,
        "merge_vad": True,
        "merge_length_s": 15,
    }
    try:
        return model.generate(**kwargs)
    except TypeError:
        compact_kwargs = {"input": str(audio_path), "language": language, "use_itn": True}
        return model.generate(**compact_kwargs)


def _parse_funasr_result(raw_result: Any, duration_seconds: float) -> dict:
    item = _first_result(raw_result)
    sentence_info = item.get("sentence_info") if isinstance(item, dict) else None
    speaker_map: dict[str, str] = {}
    events: list[str] = []
    transcript: list[dict] = []

    if isinstance(sentence_info, list) and sentence_info:
        for sentence in sentence_info:
            if not isinstance(sentence, dict):
                continue
            text, text_events = _clean_text(_sentence_text(sentence))
            events.extend(text_events)
            if not text:
                continue
            start = _timestamp_to_seconds(sentence.get("start", sentence.get("start_time", 0)))
            end = _timestamp_to_seconds(sentence.get("end", sentence.get("end_time", start)))
            speaker = _stable_speaker_label(
                sentence.get("speaker", sentence.get("spk", sentence.get("spk_id", "0"))),
                speaker_map,
            )
            transcript.append(
                {
                    "start": _format_seconds(start),
                    "end": _format_seconds(max(end, start)),
                    "speaker": speaker,
                    "text": text,
                }
            )

    if not transcript and isinstance(item, dict):
        text, text_events = _clean_text(str(item.get("text") or ""))
        events.extend(text_events)
        if text:
            transcript.append(
                {
                    "start": "00:00",
                    "end": _format_seconds(duration_seconds),
                    "speaker": "说话人 A",
                    "text": text,
                }
            )

    return {"transcript": transcript, "events": sorted(set(events))}


def _sentence_text(sentence: dict) -> str:
    return str(sentence.get("text") or sentence.get("sentence") or sentence.get("raw_text") or "")


def _first_result(raw_result: Any) -> dict:
    if isinstance(raw_result, list) and raw_result:
        first = raw_result[0]
        return first if isinstance(first, dict) else {}
    if isinstance(raw_result, dict):
        return raw_result
    return {}


def _clean_text(text: str) -> tuple[str, list[str]]:
    tags = re.findall(r"<\s*\|\s*([^|]+?)\s*\|\s*>", text)
    cleaned = re.sub(r"<\s*\|\s*[^|]+?\s*\|\s*>", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    events = [
        _normalize_tag(tag)
        for tag in tags
        if _normalize_tag(tag).lower() not in {"zh", "en", "yue", "ja", "ko", "nospeech", "withitn", "woitn"}
    ]
    return cleaned, events


def _normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", "", tag).strip()


def _stable_speaker_label(raw_speaker: Any, speaker_map: dict[str, str]) -> str:
    raw = str(raw_speaker if raw_speaker is not None else "0").strip() or "0"
    if raw.startswith("说话人 "):
        return raw
    if raw not in speaker_map:
        index = len(speaker_map)
        suffix = chr(ord("A") + index) if index < 26 else str(index + 1)
        speaker_map[raw] = f"说话人 {suffix}"
    return speaker_map[raw]


def _timestamp_to_seconds(value: Any) -> float:
    if isinstance(value, (list, tuple)) and value:
        return _timestamp_to_seconds(value[0])
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1000:
        return number / 1000.0
    return max(0.0, number)


def _format_seconds(value: float) -> str:
    total = int(round(max(0.0, value)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _funasr_metrics(
    *,
    model: str,
    vad_model: str,
    punc_model: str,
    spk_model: str,
    spk_mode: str,
    device: str,
    language: str,
    status: str,
    segment_count: int,
    speaker_count: int,
    cache_status: str,
    events: list[str],
) -> dict[str, str]:
    metrics = {
        "ASR 后端": "funasr",
        "ASR 模型": model,
        "ASR 设备": device,
        "ASR 状态": status,
        "ASR 语言": language,
        "ASR 分段数": str(segment_count),
        "ASR模型缓存": cache_status,
        "主处理后端": "FunASR中文会议转写",
        "中文ASR模型": model,
        "说话人分段模型": spk_model,
        "说话人分段模式": spk_mode,
        "VAD模型": vad_model,
        "标点模型": punc_model,
        "检测说话人数": str(speaker_count),
    }
    if events:
        metrics["SenseVoice事件标签"] = "、".join(events[:8])
    return metrics


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if _torch_cuda_available() else "cpu"
    return requested_device or DEFAULT_FUNASR_DEVICE


def _torch_cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return max(1, value)
