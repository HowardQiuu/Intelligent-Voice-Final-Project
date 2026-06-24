from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .audio_service import UPLOAD_DIR, audio_url, ffmpeg_executable, get_audio_duration_seconds, resolve_static_url


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_SEPARATION_MODEL = "speechbrain/sepformer-wsj02mix"
DEFAULT_SEPARATION_DEVICE = "auto"
DEFAULT_MAX_SECONDS = 60.0
DEFAULT_CHUNK_SECONDS = 60.0
DEFAULT_MAX_CHUNKS = 120
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
        try:
            return _separate_with_speechbrain_chunks(source_path, output_stem, duration)
        except Exception as exc:
            return _with_fallback_status(fallback, f"Chunked SpeechBrain failed: {_short_error(exc)}")

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
    requested_device = os.getenv("SEPARATION_DEVICE", DEFAULT_SEPARATION_DEVICE).strip().lower() or DEFAULT_SEPARATION_DEVICE
    device = _resolve_torch_device(torch, requested_device)
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
        "method": f"SpeechBrain SepFormer ({model_name}, {device})",
        "status": "ok",
        "track_count": str(len(tracks)),
        "tracks": tracks,
    }


def _separate_with_speechbrain_chunks(source_path: Path, output_stem: str, duration: float) -> dict:
    chunk_seconds = _get_chunk_seconds()
    chunk_count = int((duration + chunk_seconds - 0.001) // chunk_seconds)
    max_chunks = _get_max_chunks()
    if chunk_count > max_chunks:
        raise RuntimeError(f"Audio requires {chunk_count} separation chunks, over limit {max_chunks}")

    work_dir = UPLOAD_DIR / f"separation_chunks_{uuid.uuid4().hex[:10]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    intermediate_tracks: list[Path] = []
    try:
        chunk_paths = _split_audio_to_chunks(source_path, work_dir, chunk_seconds, duration)
        grouped: dict[int, list[Path]] = {}
        for chunk_index, chunk_path in enumerate(chunk_paths, start=1):
            result = _separate_with_speechbrain(
                chunk_path,
                f"{output_stem}_chunk_{chunk_index:04d}",
                max_seconds=chunk_seconds,
            )
            for speaker_index, track in enumerate(result["tracks"], start=1):
                track_path = resolve_static_url(track["audio_url"])
                if track_path is None or not track_path.exists():
                    raise RuntimeError(f"Separated chunk track missing: {track['audio_url']}")
                grouped.setdefault(speaker_index, []).append(track_path)
                intermediate_tracks.append(track_path)

        if not grouped:
            raise RuntimeError("SpeechBrain returned no separated chunk tracks")

        tracks = []
        for speaker_index in sorted(grouped):
            output_path = UPLOAD_DIR / f"{output_stem}_speaker_{speaker_index}_chunked.wav"
            _concat_audio_chunks(grouped[speaker_index], output_path)
            tracks.append(
                {
                    "track_id": f"{output_stem}_speaker_{speaker_index}",
                    "label": f"分离说话人 {speaker_index}",
                    "audio_url": audio_url(output_path),
                    "description": (
                        "SpeechBrain SepFormer 分块分离后拼接输出，"
                        f"共 {len(chunk_paths)} 个音频块。"
                    ),
                }
            )

        return {
            "method": f"SpeechBrain SepFormer chunked ({len(chunk_paths)} chunks x {chunk_seconds:g}s)",
            "status": "ok-chunked",
            "track_count": str(len(tracks)),
            "tracks": tracks,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        for track_path in intermediate_tracks:
            track_path.unlink(missing_ok=True)


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


def _resolve_torch_device(torch: Any, requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if _torch_cuda_usable(torch) else "cpu"
    if requested_device == "cuda":
        return "cuda" if _torch_cuda_usable(torch) else "cpu"
    return requested_device


def _torch_cuda_usable(torch: Any) -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        probe = torch.ones(1, device="cuda")
        _ = (probe + 1).cpu()
        return True
    except Exception:
        return False


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


def _get_chunk_seconds() -> float:
    raw = os.getenv("SEPARATION_CHUNK_SECONDS", str(DEFAULT_CHUNK_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CHUNK_SECONDS
    return max(5.0, value)


def _get_max_chunks() -> int:
    raw = os.getenv("SEPARATION_MAX_CHUNKS", str(DEFAULT_MAX_CHUNKS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CHUNKS
    return max(1, value)


def _split_audio_to_chunks(path: Path, work_dir: Path, chunk_seconds: float, duration: float) -> list[Path]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for chunked SpeechBrain separation")

    chunk_paths: list[Path] = []
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
            "8000",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to create separation chunk {index}")
        chunk_paths.append(chunk_path)
        start += chunk_seconds
        index += 1
    return chunk_paths


def _concat_audio_chunks(chunk_paths: list[Path], output_path: Path) -> None:
    if not chunk_paths:
        raise RuntimeError("No separated chunks to concatenate")

    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to concatenate separated chunks")

    list_path = output_path.with_suffix(".concat.txt")
    list_lines = []
    for chunk_path in chunk_paths:
        normalized = chunk_path.resolve().as_posix().replace("'", "'\\''")
        list_lines.append(f"file '{normalized}'")
    list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")

    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-ac",
            "1",
            "-ar",
            "8000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    finally:
        list_path.unlink(missing_ok=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Chunked separation concatenation produced an empty output")


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
