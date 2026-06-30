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
from .audio_quality_service import (
    analyze_audio_quality,
    apply_audibility_pregain,
    quality_metric_strings,
    score_enhancement_candidate,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_VENDOR_DIR = BACKEND_DIR / "vendor" / "DeepFilterNet"
DEFAULT_SOURCE_DIR = PROJECT_VENDOR_DIR / "DeepFilterNet"
DEFAULT_MODEL_DIR = PROJECT_VENDOR_DIR / "models" / "DeepFilterNet3"
DEFAULT_ENHANCEMENT_MAX_SECONDS = 300.0
DEFAULT_ENHANCEMENT_CHUNK_SECONDS = 60.0
DEFAULT_ENHANCEMENT_MAX_CHUNKS = 120
DEFAULT_ENHANCEMENT_WORKERS = 2
DEFAULT_ENHANCEMENT_SKIP_SECONDS = 0.0
POST_ENHANCEMENT_LOUDNESS_FILTER = "highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11,alimiter=limit=0.95"
_DEEPFILTER_SOURCE_CACHE: dict[str, tuple[Any, Any, str]] = {}
_DEEPFILTER_SOURCE_CACHE_LOCK = threading.Lock()
_DEEPFILTER_SOURCE_INFERENCE_LOCK = threading.Lock()
_CLEARVOICE_ENHANCER_CACHE: dict[tuple[str, str], Any] = {}
_CLEARVOICE_ENHANCER_LOCK = threading.Lock()


def _prepare_clearvoice_runtime() -> None:
    runtime_dir = BACKEND_DIR / ".runtime" / "numba_cache"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(runtime_dir))


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

    _patch_torchaudio_legacy_io()

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


def _patch_torchaudio_legacy_io() -> None:
    """Route DeepFilterNet's torchaudio IO calls through soundfile for WAV compatibility."""
    try:
        torchaudio = importlib.import_module("torchaudio")
        soundfile = importlib.import_module("soundfile")
        torch = importlib.import_module("torch")
        from types import SimpleNamespace

        def info(path: str, **_kwargs):
            metadata = soundfile.info(path)
            return SimpleNamespace(
                sample_rate=metadata.samplerate,
                num_frames=metadata.frames,
                num_channels=metadata.channels,
                bits_per_sample=0,
                encoding=str(metadata.subtype or ""),
            )

        def load(path: str, frame_offset: int = 0, num_frames: int = -1, channels_first: bool = True, **_kwargs):
            stop = None if num_frames is None or num_frames < 0 else frame_offset + num_frames
            data, sample_rate = soundfile.read(
                path,
                start=max(0, frame_offset),
                stop=stop,
                always_2d=True,
                dtype="float32",
            )
            tensor = torch.from_numpy(data.T.copy() if channels_first else data.copy())
            return tensor, sample_rate

        def save(path: str, tensor, sample_rate: int, **_kwargs):
            audio = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
            if getattr(audio, "ndim", 0) == 2 and audio.shape[0] <= audio.shape[1]:
                audio = audio.T
            soundfile.write(path, audio, sample_rate)

        torchaudio.info = info
        torchaudio.load = load
        torchaudio.save = save
    except Exception:
        return


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
    skip_reason = _enhancement_skip_reason(duration)
    if skip_reason:
        return {
            "original_audio_url": audio_url(path),
            "enhanced_audio_url": audio_url(path),
            "method": f"Enhancement skipped ({skip_reason})",
            "metrics": {
                **_loudness_metrics("skipped"),
                **_enhancement_runtime_metrics(duration),
            },
        }

    original_quality = analyze_audio_quality(path)
    candidate_input, pregain_status, pregain_metrics = apply_audibility_pregain(path)
    candidate_duration = get_audio_duration_seconds(candidate_input) or duration

    if not _quality_router_enabled():
        denoised_path, denoise_method = _run_deepfilternet_candidate(candidate_input, candidate_duration)
        enhanced_path, loudness_status = postprocess_enhanced_audio(denoised_path)
        method = denoise_method
        if loudness_status == "ok":
            method = f"{denoise_method} + loudness normalization"
        enhanced_quality = analyze_audio_quality(enhanced_path)
        return {
            "original_audio_url": audio_url(path),
            "enhanced_audio_url": audio_url(enhanced_path),
            "method": method,
            "metrics": {
                **pregain_metrics,
                **quality_metric_strings("enhancement_selected", enhanced_quality),
                **_loudness_metrics(loudness_status),
                **_enhancement_runtime_metrics(duration),
                "quality_router_status": "disabled",
                "quality_pregain_status": pregain_status,
            },
        }

    selected = _select_enhancement_candidate(candidate_input, candidate_duration, original_quality)
    enhanced_path = selected["path"]
    method = selected["method"]
    return {
        "original_audio_url": audio_url(path),
        "enhanced_audio_url": audio_url(enhanced_path),
        "method": method,
        "metrics": {
            **pregain_metrics,
            **selected["metrics"],
            **_loudness_metrics(selected["loudness_status"]),
            **_enhancement_runtime_metrics(duration),
            "quality_router_status": "enabled",
            "quality_pregain_status": pregain_status,
        },
    }


def _run_deepfilternet_candidate(path: Path, duration: float | None) -> tuple[Path, str]:
    if duration is not None and duration > _get_enhancement_max_seconds():
        return denoise_audio_in_chunks(path, duration)
    return denoise_audio(path)


def _select_enhancement_candidate(path: Path, duration: float | None, original_quality) -> dict:
    attempts = []
    for candidate in _get_enhancement_candidates():
        try:
            denoised_path, denoise_method = _run_enhancement_candidate(candidate, path, duration)
            enhanced_path, loudness_status = postprocess_enhanced_audio(denoised_path)
            quality = analyze_audio_quality(enhanced_path)
            score = score_enhancement_candidate(original_quality, quality)
            attempts.append(
                {
                    "candidate": candidate,
                    "path": enhanced_path,
                    "method": f"{denoise_method} + loudness normalization + quality score {score:.1f}",
                    "score": score,
                    "quality": quality,
                    "loudness_status": loudness_status,
                    "status": "ok",
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "candidate": candidate,
                    "score": -1.0,
                    "status": f"skipped: {_short_error(exc)}",
                }
            )

    valid = [item for item in attempts if item.get("status") == "ok"]
    if not valid:
        fallback_quality = analyze_audio_quality(path)
        return {
            "path": path,
            "method": "Audibility pregain fallback (no enhancement candidate succeeded)",
            "loudness_status": "fallback",
            "metrics": {
                **quality_metric_strings("enhancement_selected", fallback_quality),
                **_candidate_metrics(attempts, "pregain-fallback"),
                "quality_router_selected_score": f"{score_enhancement_candidate(original_quality, fallback_quality):.1f}",
            },
        }

    selected = max(valid, key=lambda item: item["score"])
    return {
        "path": selected["path"],
        "method": selected["method"],
        "loudness_status": selected["loudness_status"],
        "metrics": {
            **quality_metric_strings("enhancement_selected", selected["quality"]),
            **_candidate_metrics(attempts, selected["candidate"]),
            "quality_router_selected_score": f"{selected['score']:.1f}",
        },
    }


def _run_enhancement_candidate(candidate: str, path: Path, duration: float | None) -> tuple[Path, str]:
    if candidate == "deepfilternet":
        return _run_deepfilternet_candidate(path, duration)
    if candidate == "clearvoice":
        return _run_clearvoice_enhancement_candidate(path)
    raise RuntimeError(f"Unsupported enhancement candidate: {candidate}")


def _run_clearvoice_enhancement_candidate(path: Path) -> tuple[Path, str]:
    model_name = os.getenv("CLEARVOICE_ENHANCE_MODEL", "MossFormer2_SE_48K").strip() or "MossFormer2_SE_48K"
    output_dir = UPLOAD_DIR / f"clearvoice_enhance_{uuid.uuid4().hex[:10]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        enhancer = _get_clearvoice_enhancer(model_name)
        enhancer(str(path), online_write=True, output_path=str(output_dir))
    except Exception as exc:
        raise RuntimeError(f"ClearVoice enhancement failed: {_short_error(exc)}") from exc

    candidates = sorted(output_dir.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"ClearVoice enhancement ({model_name}) did not produce a WAV file")

    output_path = UPLOAD_DIR / f"{path.stem}_{model_name.lower()}_clearvoice.wav"
    shutil.copyfile(candidates[0], output_path)
    shutil.rmtree(output_dir, ignore_errors=True)
    return output_path, f"ClearVoice {model_name} enhancement"


def _get_clearvoice_enhancer(model_name: str) -> Any:
    cuda_mode = os.getenv("CLEARVOICE_ENHANCE_USE_CUDA", os.getenv("CLEARVOICE_USE_CUDA", "auto")).strip().lower() or "auto"
    cache_key = (model_name, cuda_mode)
    with _CLEARVOICE_ENHANCER_LOCK:
        cached = _CLEARVOICE_ENHANCER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            _prepare_clearvoice_runtime()
            clearvoice_module = importlib.import_module("clearvoice")
            clearvoice_class = getattr(clearvoice_module, "ClearVoice")
            cwd = Path.cwd()
            with _clearvoice_cuda_policy(cuda_mode):
                try:
                    os.chdir(BACKEND_DIR)
                    model = clearvoice_class(task="speech_enhancement", model_names=[model_name])
                finally:
                    os.chdir(cwd)
        except Exception as exc:
            raise RuntimeError(
                "ClearVoice is not available. Install the official package in the backend Python "
                "environment with: python -m pip install clearvoice"
            ) from exc
        _CLEARVOICE_ENHANCER_CACHE[cache_key] = model
        return model


class _clearvoice_cuda_policy:
    def __init__(self, mode: str):
        self.mode = mode
        self._torch = None
        self._mps = None
        self._cuda_available = None
        self._mps_available = None

    def __enter__(self):
        if self.mode not in {"0", "false", "no", "cpu"}:
            return self
        self._torch = importlib.import_module("torch")
        self._mps = importlib.import_module("torch.backends.mps")
        self._cuda_available = self._torch.cuda.is_available
        self._mps_available = self._mps.is_available
        self._torch.cuda.is_available = lambda: False
        self._mps.is_available = lambda: False
        return self

    def __exit__(self, *_exc_info):
        if self._torch is not None and self._cuda_available is not None:
            self._torch.cuda.is_available = self._cuda_available
        if self._mps is not None and self._mps_available is not None:
            self._mps.is_available = self._mps_available
        return False


def _candidate_metrics(attempts: list[dict], selected: str) -> dict[str, str]:
    labels = []
    for item in attempts:
        score = item.get("score", -1.0)
        score_text = f"{score:.1f}" if score >= 0 else item.get("status", "skipped")
        labels.append(f"{item['candidate']}={score_text}")
    return {
        "quality_router_enhancement_candidates": "; ".join(labels),
        "quality_router_selected_enhancement": selected,
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
    duration = get_audio_duration_seconds(path)
    return bool(_enhancement_skip_reason(duration))


def _enhancement_skip_reason(duration: float | None) -> str:
    backend = _get_deepfilternet_backend()
    if backend in {"off", "disabled", "placeholder", "skip", "none"}:
        return f"DEEPFILTERNET_BACKEND={backend}"

    skip_seconds = _get_enhancement_skip_seconds()
    if skip_seconds > 0 and duration is not None and duration > skip_seconds:
        return f"duration>{skip_seconds:g}s"
    return ""


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


def _get_enhancement_candidates() -> list[str]:
    raw = os.getenv("ENHANCEMENT_CANDIDATES", "deepfilternet,clearvoice").strip()
    candidates = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return candidates or ["deepfilternet"]


def _quality_router_enabled() -> bool:
    return os.getenv("QUALITY_ROUTER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _get_enhancement_skip_seconds() -> float:
    raw = os.getenv("ENHANCEMENT_SKIP_SECONDS", str(DEFAULT_ENHANCEMENT_SKIP_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_ENHANCEMENT_SKIP_SECONDS
    return max(0.0, value)


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


def _short_error(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 100:
        message = f"{message[:97]}..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
