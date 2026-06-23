from __future__ import annotations

import importlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .audio_service import UPLOAD_DIR, audio_url, get_audio_duration_seconds, resolve_static_url


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_SEPARATION_MODEL = "speechbrain/sepformer-wsj02mix"
DEFAULT_MAX_SECONDS = 60.0
SPEECHBRAIN_SAVEDIR = BACKEND_DIR / "models" / "speechbrain" / "sepformer-wsj02mix"
_SEPARATOR_CACHE: dict[tuple[str, str], Any] = {}


def separate_demo_audio(case_id: str, enhanced_audio_url: str) -> dict:
    source_path = _resolve_static_url(enhanced_audio_url)
    fallback = _placeholder_demo_result(case_id, enhanced_audio_url)
    return _separate_audio(
        source_path=source_path,
        source_url=enhanced_audio_url,
        output_stem=f"{case_id}_{uuid.uuid4().hex[:8]}",
        fallback=fallback,
    )


def separate_uploaded_audio(enhanced_audio_url: str) -> dict:
    source_path = _resolve_static_url(enhanced_audio_url)
    fallback = _placeholder_upload_result(enhanced_audio_url)
    return _separate_audio(
        source_path=source_path,
        source_url=enhanced_audio_url,
        output_stem=f"upload_{uuid.uuid4().hex[:8]}",
        fallback=fallback,
    )


def _separate_audio(source_path: Path | None, source_url: str, output_stem: str, fallback: dict) -> dict:
    backend = os.getenv("SEPARATION_BACKEND", "placeholder").strip().lower() or "placeholder"
    if backend in {"placeholder", "demo", "off"}:
        return fallback
    if backend != "speechbrain":
        return _with_fallback_status(fallback, f"Unsupported backend: {backend}")

    if source_path is None or not source_path.exists():
        return _with_fallback_status(fallback, "Enhanced audio file not found")

    max_seconds = _get_max_seconds()
    duration = get_audio_duration_seconds(source_path)
    if duration is not None and duration > max_seconds:
        return _with_fallback_status(
            fallback,
            f"Audio too long for SpeechBrain ({duration / 60:.1f} min > {max_seconds / 60:.1f} min)",
        )

    try:
        return _separate_with_speechbrain(source_path, output_stem, max_seconds=max_seconds)
    except Exception as exc:
        return _with_fallback_status(fallback, f"SpeechBrain failed: {_short_error(exc)}")


def _separate_with_speechbrain(source_path: Path, output_stem: str, max_seconds: float | None = None) -> dict:
    speechbrain = importlib.import_module("speechbrain.inference.separation")
    fetching = importlib.import_module("speechbrain.utils.fetching")
    torch = importlib.import_module("torch")
    torchaudio = importlib.import_module("torchaudio")

    separator_class = getattr(speechbrain, "SepformerSeparation")
    local_strategy = getattr(getattr(fetching, "LocalStrategy"), "COPY")
    model_name = os.getenv("SEPARATION_MODEL", DEFAULT_SEPARATION_MODEL).strip() or DEFAULT_SEPARATION_MODEL
    device = os.getenv("SEPARATION_DEVICE", "cpu").strip() or "cpu"
    max_seconds = max_seconds if max_seconds is not None else _get_max_seconds()

    separator = _get_speechbrain_separator(separator_class, model_name, device, local_strategy)
    _disable_speechbrain_optional_lazy_modules()
    sources = separator.separate_file(path=_speechbrain_audio_path(source_path))
    sources = sources.detach().cpu()
    sources = _trim_sources(sources, max_seconds)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    source_count = _source_count(sources)
    tracks = []
    for index in range(source_count):
        speaker_audio = _speaker_tensor(torch, sources, index)
        output_path = UPLOAD_DIR / f"{output_stem}_speaker_{index + 1}.wav"
        _save_speaker_audio(torchaudio, speaker_audio, output_path, sample_rate=8000)
        tracks.append(
            {
                "track_id": f"{output_stem}_speaker_{index + 1}",
                "label": f"分离说话人 {index + 1}",
                "audio_url": audio_url(output_path),
                "description": f"SpeechBrain SepFormer 输出的第 {index + 1} 条说话人音轨。",
            }
        )

    if not tracks:
        raise RuntimeError("SpeechBrain returned no separated sources")

    return {
        "method": f"SpeechBrain SepFormer ({model_name})",
        "status": "ok",
        "track_count": str(len(tracks)),
        "tracks": tracks,
    }


def _placeholder_demo_result(case_id: str, enhanced_audio_url: str) -> dict:
    label = {
        "clear_meeting": "主说话人轨道",
        "noisy_meeting": "降噪后会议语音轨道",
        "overlap_meeting": "多人讨论分离轨道",
    }.get(case_id, "会议语音轨道")
    return {
        "method": "Placeholder fallback",
        "status": "placeholder",
        "track_count": "1",
        "tracks": [
            {
                "track_id": f"{case_id}_speaker_mix",
                "label": label,
                "audio_url": enhanced_audio_url,
                "description": "演示模式复用增强后音频，后续可替换为真实说话人分离模型输出。",
            }
        ],
    }


def _placeholder_upload_result(enhanced_audio_url: str) -> dict:
    return {
        "method": "Placeholder fallback",
        "status": "placeholder",
        "track_count": "1",
        "tracks": [
            {
                "track_id": "upload_speaker_mix",
                "label": "上传音频语音轨道",
                "audio_url": enhanced_audio_url,
                "description": "上传演示模式保留接口形状，后续可输出多个说话人独立音轨。",
            }
        ],
    }


def _with_fallback_status(fallback: dict, reason: str) -> dict:
    result = {
        **fallback,
        "method": "Placeholder fallback",
        "status": reason,
    }
    tracks = []
    for track in result.get("tracks", []):
        tracks.append(
            {
                **track,
                "description": f"{track.get('description', '')} 当前回退原因：{reason}",
            }
        )
    result["tracks"] = tracks
    return result


def _short_error(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 120:
        message = f"{message[:117]}..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _disable_speechbrain_optional_lazy_modules() -> None:
    # SpeechBrain 1.x registers optional lazy integration redirects.
    # Python inspect can touch them while audio_io.load resolves warnings,
    # causing optional dependency errors unrelated to SepFormer separation.
    optional_modules = {
        "speechbrain.pretrained",
        "speechbrain.k2_integration",
        "speechbrain.wordemb",
        "speechbrain.lobes.models.huggingface_transformers",
        "speechbrain.lobes.models.spacy",
        "speechbrain.lobes.models.flair",
        "speechbrain.nnet.loss.transducer_loss",
    }
    for module_name in optional_modules:
        sys.modules.pop(module_name, None)


def _get_speechbrain_separator(separator_class: Any, model_name: str, device: str, local_strategy: Any) -> Any:
    cache_key = (model_name, device)
    separator = _SEPARATOR_CACHE.get(cache_key)
    if separator is None:
        separator = separator_class.from_hparams(
            source=model_name,
            savedir=str(SPEECHBRAIN_SAVEDIR),
            run_opts={"device": device},
            local_strategy=local_strategy,
        )
        _SEPARATOR_CACHE[cache_key] = separator
    return separator


def _resolve_static_url(url: str) -> Path | None:
    return resolve_static_url(url)


def _speechbrain_audio_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _get_max_seconds() -> float:
    raw = os.getenv("SEPARATION_MAX_SECONDS", str(DEFAULT_MAX_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MAX_SECONDS
    return max(1.0, value)


def _trim_sources(sources: Any, max_seconds: float) -> Any:
    max_samples = int(8000 * max_seconds)
    shape = tuple(getattr(sources, "shape", ()))
    if not shape:
        return sources
    time_dim = _time_dimension(shape)
    if shape[time_dim] <= max_samples:
        return sources
    if time_dim == 0:
        return sources[:max_samples, ...]
    if time_dim == 1:
        return sources[:, :max_samples, ...]
    return sources[..., :max_samples]


def _source_count(sources: Any) -> int:
    shape = tuple(getattr(sources, "shape", ()))
    if len(shape) >= 3:
        return int(shape[-1])
    if len(shape) == 2:
        if shape[0] > shape[1] and shape[1] <= 8:
            return int(shape[1])
        return int(shape[0])
    return 0


def _speaker_tensor(torch: Any, sources: Any, index: int) -> Any:
    shape = tuple(getattr(sources, "shape", ()))
    if len(shape) >= 3:
        audio = sources[0, :, index].unsqueeze(0)
    elif len(shape) == 2:
        if shape[0] > shape[1] and shape[1] <= 8:
            audio = sources[:, index].unsqueeze(0)
        else:
            audio = sources[index].unsqueeze(0)
    else:
        raise RuntimeError("Unsupported SpeechBrain source tensor shape")
    return audio.to(dtype=torch.float32)


def _save_speaker_audio(torchaudio: Any, speaker_audio: Any, output_path: Path, sample_rate: int) -> None:
    try:
        torchaudio.save(str(output_path), speaker_audio, sample_rate=sample_rate)
        return
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise

    soundfile = importlib.import_module("soundfile")
    waveform = speaker_audio.squeeze(0).detach().cpu().numpy()
    soundfile.write(str(output_path), waveform, sample_rate)


def _time_dimension(shape: tuple[int, ...]) -> int:
    if len(shape) >= 3:
        return 1
    if len(shape) == 2 and shape[0] > shape[1] and shape[1] <= 8:
        return 0
    if len(shape) == 2:
        return 1
    return 0
