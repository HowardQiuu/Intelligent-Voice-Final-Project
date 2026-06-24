from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .audio_service import (
    UPLOAD_DIR,
    apply_audio_filter,
    audio_url,
    ffmpeg_executable,
    generate_demo_audio,
    get_audio_duration_seconds,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_VENDOR_DIR = BACKEND_DIR / "vendor" / "DeepFilterNet"
DEFAULT_SOURCE_DIR = PROJECT_VENDOR_DIR / "DeepFilterNet"
DEFAULT_MODEL_DIR = PROJECT_VENDOR_DIR / "models" / "DeepFilterNet3"
DEFAULT_ENHANCEMENT_MAX_SECONDS = 300.0
DEFAULT_ENHANCEMENT_CHUNK_SECONDS = 60.0
DEFAULT_ENHANCEMENT_MAX_CHUNKS = 120
DEFAULT_ENHANCEMENT_WORKERS = 2
POST_ENHANCEMENT_LOUDNESS_FILTER = "highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11,alimiter=limit=0.95"
_DEEPFILTER_SOURCE_CACHE: dict[str, tuple[Any, Any, str]] = {}
_DEEPFILTER_SOURCE_CACHE_LOCK = threading.Lock()
_DEEPFILTER_SOURCE_INFERENCE_LOCK = threading.Lock()


def _normalize_model_dir(model_dir: Path) -> Path:
    if (model_dir / "config.ini").exists():
        return model_dir
    nested = [p for p in model_dir.rglob("config.ini") if p.is_file()]
    if nested:
        return nested[0].parent
    return model_dir


def enhance_demo_audio(case_id: str) -> dict[str, str]:
    original = generate_demo_audio(case_id, noisy=True)
    enhanced = generate_demo_audio(case_id, noisy=False)
    return {
        "original_audio_url": audio_url(original),
        "enhanced_audio_url": audio_url(enhanced),
        "method": "Demo cached enhancement",
    }


def _resolve_deepfilternet_source_dir() -> Path:
    source_dir = Path(os.getenv("DEEPFILTERNET_SOURCE_DIR", str(DEFAULT_SOURCE_DIR))).resolve()
    if not (source_dir / "df" / "enhance.py").exists():
        if importlib.util.find_spec("df.enhance") is not None:
            return Path()
        raise RuntimeError(
            "DeepFilterNet source code not found. Set DEEPFILTERNET_SOURCE_DIR to the official "
            "DeepFilterNet/DeepFilterNet directory that contains df/enhance.py, or install the "
            "official package with: python -m pip install deepfilternet."
        )
    return source_dir


def _resolve_deepfilternet_model_dir() -> Path | None:
    configured = os.getenv("DEEPFILTERNET_MODEL_DIR")
    if configured:
        model_dir = Path(configured).resolve()
        if not model_dir.exists():
            raise RuntimeError(f"DeepFilterNet model directory not found: {model_dir}")
        return _normalize_model_dir(model_dir)

    if DEFAULT_MODEL_DIR.exists():
        return _normalize_model_dir(DEFAULT_MODEL_DIR)

    models_dir = PROJECT_VENDOR_DIR / "models"
    if models_dir.exists():
        candidates = [p for p in models_dir.iterdir() if p.is_dir() and p.name.lower().startswith("deepfilternet")]
        if candidates:
            return _normalize_model_dir(sorted(candidates, key=lambda p: p.name, reverse=True)[0])

    return None


def denoise_audio_with_source(path: Path) -> tuple[Path, str]:
    """Denoise uploaded audio through the official DeepFilterNet source API."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = _resolve_deepfilternet_source_dir()
    model_dir = _resolve_deepfilternet_model_dir()

    if source_dir and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    try:
        enhance_module = importlib.import_module("df.enhance")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "DeepFilterNet source dependencies are not available. Install the official project "
            "requirements, or install DeepFilterNet in editable mode from its source tree."
        ) from exc

    init_df = enhance_module.init_df
    load_audio = enhance_module.load_audio
    enhance = enhance_module.enhance
    save_audio = enhance_module.save_audio

    out_dir = UPLOAD_DIR / f"deepfilternet_source_{uuid.uuid4().hex[:10]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_DIR / f"{path.stem}_deepfilternet_source.wav"
    model_base_dir = str(model_dir) if model_dir else None

    try:
        model, df_state, model_name = _get_deepfilternet_source_model(init_df, model_base_dir, model_dir)
        audio, _ = load_audio(str(path), df_state.sr(), "cpu")
        with _DEEPFILTER_SOURCE_INFERENCE_LOCK:
            with torch.no_grad():
                enhanced = enhance(model, df_state, audio)
        save_audio(
            str(path),
            enhanced.to("cpu"),
            sr=df_state.sr(),
            output_dir=str(out_dir),
            suffix="deepfilternet_source",
            log=False,
        )
    except Exception as exc:
        raise RuntimeError(f"DeepFilterNet source inference failed: {exc}") from exc

    candidates = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        shutil.copyfile(candidates[0], output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("DeepFilterNet source inference finished without producing an output WAV file")

    return output_path, f"DeepFilterNet source inference ({model_name})"


def _get_deepfilternet_source_model(init_df: Any, model_base_dir: str | None, model_dir: Path | None) -> tuple[Any, Any, str]:
    cache_key = model_base_dir or "__default__"
    with _DEEPFILTER_SOURCE_CACHE_LOCK:
        cached = _DEEPFILTER_SOURCE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        init_result = init_df(model_base_dir=model_base_dir, log_file=None)
        model_name = model_dir.name if model_dir else "default pretrained model"
        cached = (init_result[0], init_result[1], model_name)
        _DEEPFILTER_SOURCE_CACHE[cache_key] = cached
        return cached


def denoise_audio_with_cli(path: Path) -> tuple[Path, str]:
    """Denoise uploaded audio through the official DeepFilterNet CLI."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:10]

    deepfilter_cmd = shutil.which("deepFilter") or shutil.which("deep-filter")
    if not deepfilter_cmd:
        raise RuntimeError("DeepFilterNet CLI not found. Install it with: python -m pip install deepfilternet")

    out_dir = UPLOAD_DIR / f"deepfilter_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    command_variants = [
        [deepfilter_cmd, str(path), "-o", str(out_dir)],
        [deepfilter_cmd, str(path), "--output-dir", str(out_dir)],
    ]
    errors: list[str] = []
    for cmd in command_variants:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            candidates = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                final_path = UPLOAD_DIR / f"{path.stem}_deepfilter.wav"
                shutil.copyfile(candidates[0], final_path)
                return final_path, "DeepFilterNet denoise"
            errors.append(f"{cmd[0]} finished without producing a WAV file")
        except (subprocess.CalledProcessError, OSError) as exc:
            errors.append(str(exc))

    raise RuntimeError(f"DeepFilterNet failed to enhance audio: {'; '.join(errors)}")


def denoise_audio(path: Path) -> tuple[Path, str]:
    """Denoise uploaded audio with DeepFilterNet.

    Backends:
    - cli: official deepFilter/deep-filter command. This is the default for classroom demos.
    - source: official source code + pretrained model directory.
    """
    backend = _get_deepfilternet_backend()
    if backend == "source":
        return denoise_audio_with_source(path)
    if backend == "cli":
        return denoise_audio_with_cli(path)
    raise RuntimeError("Unsupported DEEPFILTERNET_BACKEND. Use 'source' or 'cli'.")


def denoise_audio_in_chunks(path: Path, duration: float) -> tuple[Path, str]:
    """Denoise long audio chunk-by-chunk to keep memory bounded."""
    chunk_seconds = _get_enhancement_chunk_seconds()
    chunk_count = int((duration + chunk_seconds - 0.001) // chunk_seconds)
    max_chunks = _get_enhancement_max_chunks()
    if chunk_count > max_chunks:
        raise RuntimeError(f"Audio requires {chunk_count} enhancement chunks, over limit {max_chunks}")

    work_dir = UPLOAD_DIR / f"enhancement_chunks_{uuid.uuid4().hex[:10]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = _split_audio_to_chunks(path, work_dir, chunk_seconds, duration)
    enhanced_chunks: list[Path] = []
    try:
        enhanced_chunks.extend(_denoise_chunk_paths(chunk_paths))

        output_path = UPLOAD_DIR / f"{path.stem}_deepfilter_chunked.wav"
        _concat_audio_chunks(enhanced_chunks, output_path)
        worker_note = _enhancement_parallel_label(len(enhanced_chunks))
        return output_path, f"DeepFilterNet chunked denoise ({len(enhanced_chunks)} chunks x {chunk_seconds:g}s, {worker_note})"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        for enhanced_path in enhanced_chunks:
            if enhanced_path.parent == UPLOAD_DIR and enhanced_path.exists():
                enhanced_path.unlink(missing_ok=True)


def _denoise_chunk_paths(chunk_paths: list[Path]) -> list[Path]:
    backend = _get_deepfilternet_backend()
    workers = _get_enhancement_workers()
    if backend != "cli" or workers <= 1 or len(chunk_paths) <= 1:
        return [_denoise_chunk_path(chunk_path) for chunk_path in chunk_paths]

    indexed_results: list[tuple[int, Path]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk_paths))) as executor:
            futures = {
                executor.submit(_denoise_chunk_path, chunk_path): index
                for index, chunk_path in enumerate(chunk_paths)
            }
            for future in as_completed(futures):
                indexed_results.append((futures[future], future.result()))
    except Exception:
        for _, enhanced_path in indexed_results:
            if enhanced_path.parent == UPLOAD_DIR and enhanced_path.exists():
                enhanced_path.unlink(missing_ok=True)
        raise
    return [path for _, path in sorted(indexed_results, key=lambda item: item[0])]


def _denoise_chunk_path(chunk_path: Path) -> Path:
    enhanced_path, _ = denoise_audio(chunk_path)
    return enhanced_path


def postprocess_enhanced_audio(path: Path) -> tuple[Path, str]:
    """Normalize enhanced audio loudness for listening and ASR, falling back to the model output."""
    output_path = UPLOAD_DIR / f"{path.stem}_loudness.wav"
    if apply_audio_filter(path, output_path, POST_ENHANCEMENT_LOUDNESS_FILTER):
        return output_path, "ok"
    output_path.unlink(missing_ok=True)
    return path, "fallback"


def enhance_uploaded_audio(path: Path) -> dict:
    duration = get_audio_duration_seconds(path)
    if duration is not None and duration > _get_enhancement_max_seconds():
        denoised_path, denoise_method = denoise_audio_in_chunks(path, duration)
    else:
        denoised_path, denoise_method = denoise_audio(path)
    enhanced_path, loudness_status = postprocess_enhanced_audio(denoised_path)
    method = denoise_method
    if loudness_status == "ok":
        method = f"{denoise_method} + loudness normalization"
    return {
        "original_audio_url": audio_url(path),
        "enhanced_audio_url": audio_url(enhanced_path),
        "method": method,
        "metrics": {
            **_loudness_metrics(loudness_status),
            **_enhancement_runtime_metrics(duration),
        },
    }


def _loudness_metrics(status: str) -> dict[str, str]:
    return {
        "响度预处理": "highpass + loudnorm(-20 LUFS) + limiter",
        "增强后响度处理": "loudnorm(-18 LUFS) + limiter",
        "响度处理状态": status,
    }


def _enhancement_runtime_metrics(duration: float | None) -> dict[str, str]:
    return {
        "增强分块并行": _enhancement_parallel_label_for_duration(duration),
        "DeepFilterNet模型缓存": "source-cache" if _get_deepfilternet_backend() == "source" else "cli-process",
    }


def should_skip_enhancement(path: Path) -> bool:
    """Backward-compatible guard: long audio is no longer skipped, it is chunked."""
    return False


def should_chunk_enhancement(path: Path) -> bool:
    duration = get_audio_duration_seconds(path)
    return duration is not None and duration > _get_enhancement_max_seconds()


def _split_audio_to_chunks(path: Path, work_dir: Path, chunk_seconds: float, duration: float) -> list[Path]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for chunked DeepFilterNet enhancement")

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
            "48000",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to create enhancement chunk {index}")
        chunk_paths.append(chunk_path)
        start += chunk_seconds
        index += 1
    return chunk_paths


def _concat_audio_chunks(chunk_paths: list[Path], output_path: Path) -> None:
    if not chunk_paths:
        raise RuntimeError("No enhanced chunks to concatenate")

    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to concatenate enhanced chunks")

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
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    finally:
        list_path.unlink(missing_ok=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Chunked DeepFilterNet concatenation produced an empty output")


def _get_enhancement_max_seconds() -> float:
    raw = os.getenv("ENHANCEMENT_MAX_SECONDS", str(DEFAULT_ENHANCEMENT_MAX_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_ENHANCEMENT_MAX_SECONDS
    return max(1.0, value)


def _get_enhancement_chunk_seconds() -> float:
    raw = os.getenv("ENHANCEMENT_CHUNK_SECONDS", str(DEFAULT_ENHANCEMENT_CHUNK_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_ENHANCEMENT_CHUNK_SECONDS
    return max(5.0, value)


def _get_enhancement_max_chunks() -> int:
    raw = os.getenv("ENHANCEMENT_MAX_CHUNKS", str(DEFAULT_ENHANCEMENT_MAX_CHUNKS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ENHANCEMENT_MAX_CHUNKS
    return max(1, value)


def _get_deepfilternet_backend() -> str:
    return os.getenv("DEEPFILTERNET_BACKEND", "cli").strip().lower() or "cli"


def _get_enhancement_workers() -> int:
    raw = os.getenv("ENHANCEMENT_WORKERS", str(DEFAULT_ENHANCEMENT_WORKERS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ENHANCEMENT_WORKERS
    return max(1, value)


def _enhancement_parallel_label(chunk_count: int) -> str:
    if _get_deepfilternet_backend() == "cli" and chunk_count > 1 and _get_enhancement_workers() > 1:
        return f"{_get_enhancement_workers()} workers"
    return "sequential"


def _enhancement_parallel_label_for_duration(duration: float | None) -> str:
    if duration is None or duration <= _get_enhancement_max_seconds():
        return "sequential"
    chunk_seconds = _get_enhancement_chunk_seconds()
    chunk_count = int((duration + chunk_seconds - 0.001) // chunk_seconds)
    return _enhancement_parallel_label(chunk_count)
