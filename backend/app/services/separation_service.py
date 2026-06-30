from __future__ import annotations

import importlib
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
import threading

from dotenv import load_dotenv

from .audio_service import UPLOAD_DIR, audio_url, ffmpeg_executable, get_audio_duration_seconds, resolve_static_url
from .audio_quality_service import analyze_audio_quality, score_audio_quality


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_SEPARATION_MODEL = "speechbrain/sepformer-libri2mix"
DEFAULT_SEPARATION_DEVICE = "auto"
DEFAULT_MAX_SECONDS = 60.0
DEFAULT_CHUNK_SECONDS = 60.0
DEFAULT_MAX_CHUNKS = 120
SPEECHBRAIN_SAVEDIR = BACKEND_DIR / "models" / "speechbrain" / "sepformer-libri2mix"
_SEPARATOR_CACHE: dict[tuple[str, str], Any] = {}
_CLEARVOICE_SEPARATOR_CACHE: dict[tuple[str, str], Any] = {}
_CLEARVOICE_SEPARATOR_LOCK = threading.Lock()


def _prepare_clearvoice_runtime() -> None:
    runtime_dir = BACKEND_DIR / ".runtime" / "numba_cache"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(runtime_dir))


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


def separate_with_quality_router(enhanced_audio_url: str, transcript: list[dict] | None = None) -> dict:
    if not _quality_router_enabled():
        diarized = build_speaker_tracks_from_transcript(enhanced_audio_url, transcript or [])
        if diarized.get("method") != "Placeholder fallback":
            return diarized
        return separate_uploaded_audio(enhanced_audio_url)

    expected_speakers = _get_expected_speaker_count(transcript or [])
    overlap_ratio = _transcript_overlap_ratio(transcript or [])
    attempts = []
    for candidate in _get_separation_candidates():
        try:
            result = _run_separation_candidate(candidate, enhanced_audio_url, transcript or [])
            score = _score_separation_result(
                result,
                expected_speakers=expected_speakers,
                overlap_ratio=overlap_ratio,
            )
            attempts.append({"candidate": candidate, "result": result, "score": score, "status": "ok"})
        except Exception as exc:
            attempts.append({"candidate": candidate, "score": -1.0, "status": f"skipped: {_short_error(exc)}"})

    valid = [item for item in attempts if item.get("status") == "ok"]
    if not valid:
        fallback = _placeholder_upload_result(enhanced_audio_url)
        return {
            **_with_fallback_status(fallback, "Quality router fallback: no separation candidate succeeded"),
            "metrics": _separation_candidate_metrics(attempts, "placeholder"),
        }

    _apply_candidate_diagnostic_rerank(valid, enhanced_audio_url)
    selected = max(valid, key=lambda item: item["score"])
    _refine_selected_speechbrain_candidate(selected, enhanced_audio_url)
    result = _annotate_separation_result(selected["result"], selected["candidate"], selected["score"])
    result["metrics"] = {
        **result.get("metrics", {}),
        **_separation_candidate_metrics(attempts, selected["candidate"]),
        "quality_router_selected_separation_score": f"{selected['score']:.1f}",
    }
    if expected_speakers:
        result["metrics"]["quality_router_expected_speakers"] = str(expected_speakers)
    if overlap_ratio > 0:
        result["metrics"]["quality_router_transcript_overlap_ratio"] = f"{overlap_ratio:.3f}"
    return result


def build_speaker_tracks_from_transcript(enhanced_audio_url: str, transcript: list[dict]) -> dict:
    """Create stable per-speaker listening tracks from diarized transcript intervals.

    This is intentionally a meeting-diarization track, not a claim of hard blind-source
    waveform separation. Non-target speaker regions are attenuated so the classroom demo
    can audibly inspect each speaker stream while keeping the full meeting timeline.
    """
    source_path = _resolve_static_url(enhanced_audio_url)
    fallback = _placeholder_upload_result(enhanced_audio_url)
    intervals = _speaker_intervals(transcript)
    if source_path is None or not source_path.exists():
        return _with_fallback_status(fallback, "Fallback enhanced mix: enhanced audio file not found")
    if not intervals:
        return _with_fallback_status(fallback, "Fallback enhanced mix: no speaker timestamps")

    try:
        tracks = _write_gated_speaker_tracks(source_path, intervals)
    except Exception as exc:
        return _with_fallback_status(fallback, f"Fallback enhanced mix: gated track failed: {_short_error(exc)}")

    if not tracks:
        return _with_fallback_status(fallback, "Fallback enhanced mix: no speaker tracks generated")
    return {
        "method": "FunASR speaker diarization gated tracks",
        "status": "ok-diarization-gated",
        "track_count": str(len(tracks)),
        "tracks": tracks,
    }


def _separate_audio(source_path: Path | None, source_url: str, output_stem: str, fallback: dict) -> dict:
    backend = os.getenv("SEPARATION_BACKEND", "placeholder").strip().lower() or "placeholder"
    if backend in {"placeholder", "demo", "off"}:
        return fallback

    if source_path is None or not source_path.exists():
        return _with_fallback_status(fallback, "Enhanced audio file not found")

    if backend == "mossformer2":
        try:
            return _run_clearvoice_mossformer2_separation(source_path, output_stem=f"{output_stem}_mossformer2")
        except Exception as exc:
            return _with_fallback_status(fallback, f"MossFormer2 failed: {_short_error(exc)}")
    if backend != "speechbrain":
        return _with_fallback_status(fallback, f"Unsupported backend: {backend}")

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


def _separate_with_speechbrain(
    source_path: Path,
    output_stem: str,
    max_seconds: float | None = None,
    *,
    model_name_override: str | None = None,
    refine_tracks: bool = True,
) -> dict:
    speechbrain = importlib.import_module("speechbrain.inference.separation")
    fetching = importlib.import_module("speechbrain.utils.fetching")
    torch = importlib.import_module("torch")
    torchaudio = importlib.import_module("torchaudio")
    _patch_torchaudio_soundfile_io(torchaudio, torch)

    separator_class = getattr(speechbrain, "SepformerSeparation")
    local_strategy = getattr(getattr(fetching, "LocalStrategy"), "COPY")
    model_name = (
        model_name_override
        or os.getenv("SEPARATION_MODEL", DEFAULT_SEPARATION_MODEL).strip()
        or DEFAULT_SEPARATION_MODEL
    )
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

    refinement_metrics = (
        _apply_speechbrain_track_refinement(source_path, tracks)
        if refine_tracks
        else {"stft_mask_refinement": "skipped_router_candidate"}
    )
    consistency_metrics = _apply_mixture_consistency_projection(source_path, tracks)
    return {
        "method": f"SpeechBrain SepFormer ({model_name}, {device})",
        "status": "ok",
        "track_count": str(len(tracks)),
        "tracks": tracks,
        "metrics": {**refinement_metrics, **consistency_metrics},
    }


def _separate_with_speechbrain_chunks(
    source_path: Path,
    output_stem: str,
    duration: float,
    *,
    model_name_override: str | None = None,
    refine_tracks: bool = True,
) -> dict:
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
                model_name_override=model_name_override,
                refine_tracks=refine_tracks,
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

        consistency_metrics = _apply_mixture_consistency_projection(source_path, tracks)
        return {
            "method": f"SpeechBrain SepFormer chunked ({len(chunk_paths)} chunks x {chunk_seconds:g}s)",
            "status": "ok-chunked",
            "track_count": str(len(tracks)),
            "tracks": tracks,
            "metrics": consistency_metrics,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        for track_path in intermediate_tracks:
            track_path.unlink(missing_ok=True)


def _run_separation_candidate(candidate: str, enhanced_audio_url: str, transcript: list[dict]) -> dict:
    source_path = _resolve_static_url(enhanced_audio_url)
    if candidate == "gated":
        result = build_speaker_tracks_from_transcript(enhanced_audio_url, transcript)
        if result.get("method") == "Placeholder fallback":
            raise RuntimeError(result.get("status", "gated track unavailable"))
        return result
    if source_path is None or not source_path.exists():
        raise RuntimeError("Enhanced audio file not found")
    if candidate == "speechbrain":
        duration = get_audio_duration_seconds(source_path)
        if duration is not None and duration > _get_max_seconds():
            return _separate_with_speechbrain_chunks(
                source_path,
                f"router_speechbrain_{uuid.uuid4().hex[:8]}",
                duration,
                refine_tracks=False,
            )
        return _separate_with_speechbrain(
            source_path,
            f"router_speechbrain_{uuid.uuid4().hex[:8]}",
            max_seconds=_get_max_seconds(),
            refine_tracks=False,
        )
    if candidate in {"libri2mix", "speechbrain-libri2mix", "speechbrain_libri2mix"}:
        duration = get_audio_duration_seconds(source_path)
        if duration is not None and duration > _get_max_seconds():
            return _separate_with_speechbrain_chunks(
                source_path,
                f"router_libri2mix_{uuid.uuid4().hex[:8]}",
                duration,
                model_name_override="speechbrain/sepformer-libri2mix",
                refine_tracks=False,
            )
        return _separate_with_speechbrain(
            source_path,
            f"router_libri2mix_{uuid.uuid4().hex[:8]}",
            max_seconds=_get_max_seconds(),
            model_name_override="speechbrain/sepformer-libri2mix",
            refine_tracks=False,
        )
    if candidate == "mossformer2":
        return _run_clearvoice_mossformer2_separation(source_path, output_stem=f"router_mossformer2_{uuid.uuid4().hex[:8]}")
    raise RuntimeError(f"Unsupported separation candidate: {candidate}")


def _run_clearvoice_mossformer2_separation(source_path: Path, output_stem: str) -> dict:
    model_name = os.getenv("MOSSFORMER2_SEPARATION_MODEL", "MossFormer2_SS_16K").strip() or "MossFormer2_SS_16K"
    out_dir = UPLOAD_DIR / f"{output_stem}_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        separator = _get_clearvoice_separator(model_name)
        separator(str(source_path), online_write=True, output_path=str(out_dir))
    except Exception as exc:
        raise RuntimeError(f"ClearVoice/MossFormer2 separation failed: {_short_error(exc)}") from exc

    candidates = sorted(out_dir.rglob("*.wav"), key=lambda p: p.name)
    if not candidates:
        raise RuntimeError(f"ClearVoice/MossFormer2 ({model_name}) did not produce separated WAV tracks")

    tracks = []
    for index, candidate_path in enumerate(candidates, start=1):
        final_path = UPLOAD_DIR / f"{output_stem}_speaker_{index}.wav"
        shutil.copyfile(candidate_path, final_path)
        tracks.append(
            {
                "track_id": f"{output_stem}_speaker_{index}",
                "label": f"MossFormer2 speaker {index}",
                "audio_url": audio_url(final_path),
                "description": f"ClearVoice {model_name} output track {index}.",
            }
        )
    shutil.rmtree(out_dir, ignore_errors=True)
    consistency_metrics = _apply_mixture_consistency_projection(source_path, tracks)
    return {
        "method": f"ClearVoice {model_name}",
        "status": "ok-mossformer2",
        "track_count": str(len(tracks)),
        "tracks": tracks,
        "metrics": consistency_metrics,
    }


def _get_clearvoice_separator(model_name: str) -> Any:
    cuda_mode = os.getenv("MOSSFORMER2_USE_CUDA", os.getenv("CLEARVOICE_USE_CUDA", "auto")).strip().lower() or "auto"
    cache_key = (model_name, cuda_mode)
    with _CLEARVOICE_SEPARATOR_LOCK:
        cached = _CLEARVOICE_SEPARATOR_CACHE.get(cache_key)
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
                    model = clearvoice_class(task="speech_separation", model_names=[model_name])
                finally:
                    os.chdir(cwd)
        except Exception as exc:
            raise RuntimeError(
                "ClearVoice is not available. Install the official package in the backend Python "
                "environment with: python -m pip install clearvoice"
            ) from exc
        _CLEARVOICE_SEPARATOR_CACHE[cache_key] = model
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


def _score_separation_result(
    result: dict,
    *,
    expected_speakers: int | None = None,
    overlap_ratio: float = 0.0,
) -> float:
    tracks = result.get("tracks", [])
    if not tracks:
        return 0.0
    score = 25.0 + min(30.0, len(tracks) * 10.0)
    track_scores = []
    for track in tracks:
        path = resolve_static_url(track.get("audio_url", ""))
        if path is not None:
            track_scores.append(score_audio_quality(analyze_audio_quality(path)))
    if track_scores:
        score += sum(track_scores) / len(track_scores) * 0.35
    status = str(result.get("status", "")).lower()
    method = str(result.get("method", "")).lower()
    if "placeholder" in status or "fallback" in status:
        score -= 30
    if "mossformer2" in method:
        score += 8
    if "speechbrain" in method:
        score += 4
    if "libri2mix" in method:
        score += _get_libri2mix_candidate_bonus()
    if "gated" in method:
        score += 2
    if expected_speakers and expected_speakers > 1 and len(tracks) < expected_speakers:
        missing_ratio = (expected_speakers - len(tracks)) / expected_speakers
        score -= 45.0 * missing_ratio
    if "mossformer2" in method and overlap_ratio >= _get_mossformer_overlap_threshold():
        score += _get_mossformer_overlap_boost() * min(1.0, overlap_ratio / 0.35)
    return max(0.0, min(100.0, score))


def _refine_selected_speechbrain_candidate(selected: dict, source_url: str) -> None:
    if not _is_speechbrain_candidate(str(selected.get("candidate", ""))):
        return
    result = selected.get("result", {})
    source_path = _resolve_static_url(source_url)
    if source_path is None or not source_path.exists():
        result["metrics"] = {
            **result.get("metrics", {}),
            "stft_mask_refinement": "skipped_missing_source",
        }
        return
    metrics = _apply_speechbrain_track_refinement(source_path, result.get("tracks", []) or [])
    result["metrics"] = {**result.get("metrics", {}), **metrics}


def _is_speechbrain_candidate(candidate: str) -> bool:
    return candidate in {"speechbrain", "libri2mix", "speechbrain-libri2mix", "speechbrain_libri2mix"}


def _annotate_separation_result(result: dict, candidate: str, score: float) -> dict:
    tracks = []
    for track in result.get("tracks", []):
        tracks.append(
            {
                **track,
                "description": (
                    f"{track.get('description', '')} Quality-aware separation candidate={candidate}, "
                    f"score={score:.1f}."
                ),
            }
        )
    return {
        **result,
        "tracks": tracks,
    }


def _separation_candidate_metrics(attempts: list[dict], selected: str) -> dict[str, str]:
    labels = []
    for item in attempts:
        score = item.get("score", -1.0)
        score_text = f"{score:.1f}" if score >= 0 else item.get("status", "skipped")
        labels.append(f"{item['candidate']}={score_text}")
    return {
        "quality_router_separation_candidates": "; ".join(labels),
        "quality_router_selected_separation": selected,
    }


def _apply_candidate_diagnostic_rerank(attempts: list[dict], source_url: str) -> None:
    if not _diagnostic_rerank_enabled():
        return
    source_path = resolve_static_url(source_url)
    if source_path is None or not source_path.exists():
        return

    diagnostics_by_candidate: dict[str, dict[str, float]] = {}
    for item in attempts:
        candidate = str(item.get("candidate", ""))
        result = item.get("result", {})
        diagnostics = _candidate_output_diagnostics(source_path, result)
        if not diagnostics:
            continue
        diagnostics_by_candidate[candidate] = diagnostics
        result["metrics"] = {
            **result.get("metrics", {}),
            **_format_candidate_diagnostics(candidate, diagnostics),
        }

    mossformer = diagnostics_by_candidate.get("mossformer2")
    if not mossformer:
        return
    speechbrain = diagnostics_by_candidate.get("speechbrain", {})
    if (
        mossformer.get("sum_mix_correlation", 0.0) >= _get_mossformer_diagnostic_mixcorr_threshold()
        and mossformer.get("track_overlap_ratio", 0.0) >= _get_mossformer_diagnostic_track_overlap_threshold()
        and speechbrain.get("inter_track_correlation", -1.0) <= _get_mossformer_diagnostic_speechbrain_corr_max()
    ):
        bonus = _get_mossformer_diagnostic_bonus()
        for item in attempts:
            if item.get("candidate") == "mossformer2":
                item["score"] = min(100.0, float(item.get("score", 0.0)) + bonus)
                result = item.get("result", {})
                result["metrics"] = {
                    **result.get("metrics", {}),
                    "quality_router_diagnostic_rerank": "mossformer2_overlap_reconstruction_boost",
                    "quality_router_mossformer2_diagnostic_bonus": f"{bonus:.1f}",
                }
                break


def _candidate_output_diagnostics(source_path: Path, result: dict) -> dict[str, float]:
    tracks = result.get("tracks", [])
    if len(tracks) < 2:
        return {}
    try:
        soundfile = importlib.import_module("soundfile")
        numpy = importlib.import_module("numpy")
        mixture, mixture_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
        mixture = mixture.mean(axis=1).astype("float32")
        track_arrays = []
        for track in tracks[:2]:
            track_path = resolve_static_url(track.get("audio_url", ""))
            if track_path is None or not track_path.exists():
                return {}
            data, track_rate = soundfile.read(str(track_path), always_2d=True, dtype="float32")
            mono = data.mean(axis=1, keepdims=True).astype("float32")
            if int(track_rate) != int(mixture_rate):
                mono = _resample_audio_array(mono, int(track_rate), int(mixture_rate), numpy)
            track_arrays.append(mono[:, 0].astype("float32"))
        mixture, first, second = _pad_mono_arrays([mixture, track_arrays[0], track_arrays[1]], numpy)
        summed = first + second
        mixture_energy = float(numpy.mean(mixture * mixture) + 1e-9)
        first_energy = float(numpy.mean(first * first) + 1e-9)
        second_energy = float(numpy.mean(second * second) + 1e-9)
        return {
            "sum_mix_correlation": _safe_correlation(summed, mixture, numpy),
            "sum_mix_residual_ratio": float(numpy.mean((summed - mixture) ** 2) / mixture_energy),
            "inter_track_correlation": _safe_correlation(first, second, numpy),
            "track_energy_balance": min(first_energy, second_energy) / max(first_energy, second_energy),
            "track_overlap_ratio": _track_activity_overlap_ratio(first, second, int(mixture_rate), numpy),
        }
    except Exception:
        return {}


def _pad_mono_arrays(arrays: list[Any], numpy: Any) -> list[Any]:
    target_len = max(len(array) for array in arrays)
    padded = []
    for array in arrays:
        if len(array) >= target_len:
            padded.append(array[:target_len].astype("float32"))
            continue
        output = numpy.zeros(target_len, dtype="float32")
        output[: len(array)] = array
        padded.append(output)
    return padded


def _safe_correlation(first: Any, second: Any, numpy: Any) -> float:
    if len(first) == 0 or len(second) == 0:
        return 0.0
    if float(numpy.std(first)) < 1e-9 or float(numpy.std(second)) < 1e-9:
        return 0.0
    return float(numpy.corrcoef(first, second)[0, 1])


def _track_activity_overlap_ratio(first: Any, second: Any, sample_rate: int, numpy: Any) -> float:
    first_rms = _frame_rms(first, sample_rate, numpy)
    second_rms = _frame_rms(second, sample_rate, numpy)
    frame_count = min(len(first_rms), len(second_rms))
    if frame_count == 0:
        return 0.0
    first_threshold = max(1e-4, float(numpy.percentile(first_rms, 95)) * 0.08)
    second_threshold = max(1e-4, float(numpy.percentile(second_rms, 95)) * 0.08)
    active = (first_rms[:frame_count] >= first_threshold) & (second_rms[:frame_count] >= second_threshold)
    return float(numpy.mean(active))


def _frame_rms(samples: Any, sample_rate: int, numpy: Any) -> Any:
    frame = max(1, int(sample_rate * 0.025))
    hop = max(1, int(sample_rate * 0.010))
    if len(samples) < frame:
        return numpy.asarray([float(numpy.sqrt(numpy.mean(samples * samples) + 1e-9))], dtype="float32")
    values = [
        float(numpy.sqrt(numpy.mean(samples[index : index + frame] ** 2) + 1e-9))
        for index in range(0, len(samples) - frame + 1, hop)
    ]
    return numpy.asarray(values, dtype="float32")


def _format_candidate_diagnostics(candidate: str, diagnostics: dict[str, float]) -> dict[str, str]:
    safe_candidate = candidate.replace("-", "_")
    return {
        f"quality_router_{safe_candidate}_sum_mix_correlation": f"{diagnostics['sum_mix_correlation']:.3f}",
        f"quality_router_{safe_candidate}_sum_mix_residual_ratio": f"{diagnostics['sum_mix_residual_ratio']:.3f}",
        f"quality_router_{safe_candidate}_inter_track_correlation": f"{diagnostics['inter_track_correlation']:.3f}",
        f"quality_router_{safe_candidate}_track_energy_balance": f"{diagnostics['track_energy_balance']:.3f}",
        f"quality_router_{safe_candidate}_track_overlap_ratio": f"{diagnostics['track_overlap_ratio']:.3f}",
    }


def _apply_mixture_consistency_projection(source_path: Path, tracks: list[dict]) -> dict[str, str]:
    if not _mixture_consistency_enabled() or len(tracks) < 2:
        return {"mixture_consistency_projection": "skipped"}
    try:
        soundfile = importlib.import_module("soundfile")
        numpy = importlib.import_module("numpy")
        track_paths = [resolve_static_url(track.get("audio_url", "")) for track in tracks]
        track_paths = [path for path in track_paths if path is not None and path.exists()]
        if len(track_paths) < 2:
            return {"mixture_consistency_projection": "skipped_missing_tracks"}

        track_arrays = []
        sample_rate = 0
        channels = 1
        for track_path in track_paths:
            data, track_rate = soundfile.read(str(track_path), always_2d=True, dtype="float32")
            if sample_rate == 0:
                sample_rate = int(track_rate)
                channels = int(data.shape[1])
            if int(track_rate) != sample_rate:
                return {"mixture_consistency_projection": "skipped_mismatched_track_rates"}
            if data.shape[1] != channels:
                data = data.mean(axis=1, keepdims=True)
                if channels > 1:
                    data = numpy.repeat(data, channels, axis=1)
            track_arrays.append(data.astype("float32"))

        target_len = max(len(data) for data in track_arrays)
        padded_tracks = [_pad_audio_array(data, target_len, channels, numpy) for data in track_arrays]
        mixture, mixture_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
        if mixture.shape[1] != channels:
            mixture = mixture.mean(axis=1, keepdims=True)
            if channels > 1:
                mixture = numpy.repeat(mixture, channels, axis=1)
        if int(mixture_rate) != sample_rate:
            mixture = _resample_audio_array(mixture, int(mixture_rate), sample_rate, numpy)
        mixture = _pad_audio_array(mixture, target_len, channels, numpy)

        residual = (mixture - numpy.sum(numpy.stack(padded_tracks, axis=0), axis=0)) / len(padded_tracks)
        limiter = _get_mixture_consistency_limiter()
        for track_path, original, padded in zip(track_paths, track_arrays, padded_tracks):
            projected = numpy.clip(padded + residual, -limiter, limiter)
            soundfile.write(str(track_path), projected[: len(original)], sample_rate)
        for track in tracks:
            track["description"] = f"{track.get('description', '')} Mixture consistency projection applied."
        return {"mixture_consistency_projection": "applied"}
    except Exception as exc:
        return {"mixture_consistency_projection": f"failed:{_short_error(exc)}"}


def _apply_speechbrain_track_refinement(source_path: Path, tracks: list[dict]) -> dict[str, str]:
    metrics = _apply_stft_mask_refinement(source_path, tracks)
    if metrics.get("stft_mask_refinement") == "applied":
        metrics.update(_apply_low_overlap_leakage_suppression(tracks))
        metrics.update(_apply_speechbrain_residual_projection(source_path, tracks))
    else:
        metrics["low_overlap_leakage_suppression"] = "skipped_without_stft_refinement"
        metrics["speechbrain_residual_projection"] = "skipped_without_stft_refinement"
    return metrics


def _apply_stft_mask_refinement(source_path: Path, tracks: list[dict]) -> dict[str, str]:
    if not _stft_mask_refinement_enabled() or len(tracks) < 2:
        return {"stft_mask_refinement": "skipped"}
    try:
        soundfile = importlib.import_module("soundfile")
        numpy = importlib.import_module("numpy")
        torch = importlib.import_module("torch")

        track_paths = [resolve_static_url(track.get("audio_url", "")) for track in tracks]
        track_paths = [path for path in track_paths if path is not None and path.exists()]
        if len(track_paths) < 2:
            return {"stft_mask_refinement": "skipped_missing_tracks"}

        mixture, mixture_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
        mixture_mono = mixture.mean(axis=1).astype("float32")
        if len(mixture_mono) == 0:
            return {"stft_mask_refinement": "skipped_empty_mixture"}

        target_len = len(mixture_mono)
        estimates = []
        for track_path in track_paths:
            data, track_rate = soundfile.read(str(track_path), always_2d=True, dtype="float32")
            mono = data.mean(axis=1, keepdims=True).astype("float32")
            if int(track_rate) != int(mixture_rate):
                mono = _resample_audio_array(mono, int(track_rate), int(mixture_rate), numpy)
            estimates.append(_pad_mono_audio(mono[:, 0], target_len, numpy))

        n_fft = min(_get_stft_mask_n_fft(), max(16, target_len))
        if n_fft % 2:
            n_fft -= 1
        n_fft = max(16, n_fft)
        hop = min(_get_stft_mask_hop(), max(1, n_fft // 2))
        power = _get_stft_mask_power()
        limiter = _get_stft_mask_limiter()
        floor = 1e-7

        window = torch.hann_window(n_fft, dtype=torch.float32)
        mixture_tensor = torch.as_tensor(mixture_mono, dtype=torch.float32)
        mixture_spec = torch.stft(
            mixture_tensor,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        estimate_specs = [
            torch.stft(
                torch.as_tensor(estimate, dtype=torch.float32),
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            for estimate in estimates
        ]
        magnitudes = torch.stack([(torch.abs(spec) + floor) ** power for spec in estimate_specs], dim=0)
        masks = magnitudes / torch.sum(magnitudes, dim=0, keepdim=True).clamp_min(floor)

        for track_path, mask in zip(track_paths, masks):
            refined = torch.istft(
                mask * mixture_spec,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                length=target_len,
            )
            refined_audio = refined.detach().cpu().numpy().astype("float32")
            peak = float(numpy.max(numpy.abs(refined_audio))) if len(refined_audio) else 0.0
            if peak > limiter:
                refined_audio = (refined_audio * (limiter / peak)).astype("float32")
            soundfile.write(str(track_path), refined_audio, int(mixture_rate))

        for track in tracks:
            track["description"] = f"{track.get('description', '')} STFT mask refinement applied."
        return {
            "stft_mask_refinement": "applied",
            "stft_mask_n_fft": str(n_fft),
            "stft_mask_hop": str(hop),
            "stft_mask_power": f"{power:g}",
        }
    except Exception as exc:
        return {"stft_mask_refinement": f"failed:{_short_error(exc)}"}


def _apply_low_overlap_leakage_suppression(tracks: list[dict]) -> dict[str, str]:
    if not _low_overlap_leakage_suppression_enabled() or len(tracks) < 2:
        return {"low_overlap_leakage_suppression": "skipped"}
    try:
        soundfile = importlib.import_module("soundfile")
        numpy = importlib.import_module("numpy")
        track_paths = [resolve_static_url(track.get("audio_url", "")) for track in tracks]
        track_paths = [path for path in track_paths if path is not None and path.exists()]
        if len(track_paths) < 2:
            return {"low_overlap_leakage_suppression": "skipped_missing_tracks"}

        track_arrays = []
        sample_rate = 0
        for track_path in track_paths:
            data, track_rate = soundfile.read(str(track_path), always_2d=True, dtype="float32")
            if sample_rate == 0:
                sample_rate = int(track_rate)
            if int(track_rate) != sample_rate:
                return {"low_overlap_leakage_suppression": "skipped_mismatched_track_rates"}
            track_arrays.append(data.mean(axis=1).astype("float32"))
        if len(track_arrays) < 2:
            return {"low_overlap_leakage_suppression": "skipped_missing_tracks"}

        target_len = max(len(data) for data in track_arrays)
        padded_tracks = [_pad_mono_audio(data, target_len, numpy) for data in track_arrays]
        overlap_ratio = _track_activity_overlap_ratio(padded_tracks[0], padded_tracks[1], sample_rate, numpy)
        if overlap_ratio > _get_low_overlap_suppression_overlap_threshold():
            return {
                "low_overlap_leakage_suppression": "skipped_high_overlap",
                "low_overlap_leakage_overlap_ratio": f"{overlap_ratio:.3f}",
            }

        frame = max(1, int(sample_rate * 0.032))
        hop = max(1, int(sample_rate * 0.008))
        frame_rms = numpy.stack([_frame_rms_custom(track, frame, hop, numpy) for track in padded_tracks], axis=0)
        frame_count = int(min(row.shape[0] for row in frame_rms))
        frame_rms = frame_rms[:, :frame_count]
        if frame_count <= 0:
            return {"low_overlap_leakage_suppression": "skipped_empty_frames"}

        db = 20.0 * numpy.log10(numpy.maximum(frame_rms, 1e-8))
        max_db = numpy.max(db, axis=0)
        winner = numpy.argmax(db, axis=0)
        sorted_db = numpy.sort(db, axis=0)
        margin = sorted_db[-1] - sorted_db[-2]
        dominance_db = _get_low_overlap_suppression_dominance_db()
        loser_gain = _get_low_overlap_suppression_loser_gain()
        active_floor_db = _get_low_overlap_suppression_active_floor_db()
        centers = numpy.asarray([index * hop + frame // 2 for index in range(frame_count)], dtype="float32")
        timeline = numpy.arange(target_len, dtype="float32")

        for track_index, (track_path, original, padded) in enumerate(zip(track_paths, track_arrays, padded_tracks)):
            frame_gain = numpy.ones(frame_count, dtype="float32")
            suppress = (winner != track_index) & (margin >= dominance_db) & (max_db >= active_floor_db)
            frame_gain[suppress] = loser_gain
            if len(frame_gain) > 1:
                gain = numpy.interp(timeline, centers, frame_gain, left=frame_gain[0], right=frame_gain[-1]).astype("float32")
            else:
                gain = numpy.full(target_len, float(frame_gain[0]), dtype="float32")
            output = (padded * gain).astype("float32")
            soundfile.write(str(track_path), output[: len(original)], sample_rate)

        for track in tracks:
            track["description"] = f"{track.get('description', '')} Low-overlap leakage suppression applied."
        return {
            "low_overlap_leakage_suppression": "applied",
            "low_overlap_leakage_overlap_ratio": f"{overlap_ratio:.3f}",
            "low_overlap_leakage_dominance_db": f"{dominance_db:g}",
            "low_overlap_leakage_loser_gain": f"{loser_gain:g}",
        }
    except Exception as exc:
        return {"low_overlap_leakage_suppression": f"failed:{_short_error(exc)}"}


def _apply_speechbrain_residual_projection(source_path: Path, tracks: list[dict]) -> dict[str, str]:
    if not _speechbrain_residual_projection_enabled() or len(tracks) < 2:
        return {"speechbrain_residual_projection": "skipped"}
    try:
        soundfile = importlib.import_module("soundfile")
        numpy = importlib.import_module("numpy")
        track_paths = [resolve_static_url(track.get("audio_url", "")) for track in tracks]
        track_paths = [path for path in track_paths if path is not None and path.exists()]
        if len(track_paths) < 2:
            return {"speechbrain_residual_projection": "skipped_missing_tracks"}

        track_arrays = []
        sample_rate = 0
        channels = 1
        for track_path in track_paths:
            data, track_rate = soundfile.read(str(track_path), always_2d=True, dtype="float32")
            if sample_rate == 0:
                sample_rate = int(track_rate)
                channels = int(data.shape[1])
            if int(track_rate) != sample_rate:
                return {"speechbrain_residual_projection": "skipped_mismatched_track_rates"}
            if data.shape[1] != channels:
                data = data.mean(axis=1, keepdims=True)
                if channels > 1:
                    data = numpy.repeat(data, channels, axis=1)
            track_arrays.append(data.astype("float32"))

        target_len = max(len(data) for data in track_arrays)
        padded_tracks = [_pad_audio_array(data, target_len, channels, numpy) for data in track_arrays]
        mixture, mixture_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
        if mixture.shape[1] != channels:
            mixture = mixture.mean(axis=1, keepdims=True)
            if channels > 1:
                mixture = numpy.repeat(mixture, channels, axis=1)
        if int(mixture_rate) != sample_rate:
            mixture = _resample_audio_array(mixture, int(mixture_rate), sample_rate, numpy)
        mixture = _pad_audio_array(mixture, target_len, channels, numpy)

        amount = _get_speechbrain_residual_projection_amount()
        residual = (mixture - numpy.sum(numpy.stack(padded_tracks, axis=0), axis=0)) / len(padded_tracks)
        limiter = _get_speechbrain_residual_projection_limiter()
        for track_path, original, padded in zip(track_paths, track_arrays, padded_tracks):
            projected = numpy.clip(padded + amount * residual, -limiter, limiter)
            soundfile.write(str(track_path), projected[: len(original)], sample_rate)

        for track in tracks:
            track["description"] = f"{track.get('description', '')} SpeechBrain residual projection applied."
        return {
            "speechbrain_residual_projection": "applied",
            "speechbrain_residual_projection_amount": f"{amount:g}",
        }
    except Exception as exc:
        return {"speechbrain_residual_projection": f"failed:{_short_error(exc)}"}


def _frame_rms_custom(samples: Any, frame: int, hop: int, numpy: Any) -> Any:
    if len(samples) < frame:
        return numpy.asarray([float(numpy.sqrt(numpy.mean(samples * samples) + 1e-9))], dtype="float32")
    values = [
        float(numpy.sqrt(numpy.mean(samples[index : index + frame] ** 2) + 1e-9))
        for index in range(0, len(samples) - frame + 1, hop)
    ]
    return numpy.asarray(values, dtype="float32")


def _pad_mono_audio(data: Any, target_len: int, numpy: Any) -> Any:
    if len(data) >= target_len:
        return data[:target_len].astype("float32")
    output = numpy.zeros(target_len, dtype="float32")
    output[: len(data)] = data
    return output


def _pad_audio_array(data: Any, target_len: int, channels: int, numpy: Any) -> Any:
    if len(data) >= target_len:
        return data[:target_len].astype("float32")
    output = numpy.zeros((target_len, channels), dtype="float32")
    output[: len(data), : data.shape[1]] = data
    return output


def _resample_audio_array(data: Any, source_rate: int, target_rate: int, numpy: Any) -> Any:
    if source_rate == target_rate or len(data) == 0:
        return data.astype("float32")
    target_len = max(1, int(round(len(data) * target_rate / source_rate)))
    source_x = numpy.linspace(0.0, 1.0, num=len(data), endpoint=False)
    target_x = numpy.linspace(0.0, 1.0, num=target_len, endpoint=False)
    channels = []
    for channel in range(data.shape[1]):
        channels.append(numpy.interp(target_x, source_x, data[:, channel]))
    return numpy.stack(channels, axis=1).astype("float32")


def _mixture_consistency_enabled() -> bool:
    return os.getenv("SEPARATION_MIXTURE_CONSISTENCY", "false").strip().lower() not in {"0", "false", "no", "off"}


def _stft_mask_refinement_enabled() -> bool:
    return os.getenv("SEPARATION_STFT_MASK_REFINEMENT", "true").strip().lower() not in {"0", "false", "no", "off"}


def _low_overlap_leakage_suppression_enabled() -> bool:
    return os.getenv("SEPARATION_LOW_OVERLAP_LEAKAGE_SUPPRESSION", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _speechbrain_residual_projection_enabled() -> bool:
    return os.getenv("SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _get_mixture_consistency_limiter() -> float:
    raw = os.getenv("SEPARATION_MIXTURE_CONSISTENCY_LIMIT", "0.98").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.98
    return min(1.0, max(0.1, value))


def _get_stft_mask_n_fft() -> int:
    return _get_int_env("SEPARATION_STFT_MASK_N_FFT", 1024, minimum=16, maximum=8192)


def _get_stft_mask_hop() -> int:
    return _get_int_env("SEPARATION_STFT_MASK_HOP", 128, minimum=1, maximum=4096)


def _get_stft_mask_power() -> float:
    return _get_float_env("SEPARATION_STFT_MASK_POWER", 1.5, minimum=0.1, maximum=8.0)


def _get_stft_mask_limiter() -> float:
    return _get_float_env("SEPARATION_STFT_MASK_LIMIT", 0.99, minimum=0.1, maximum=1.0)


def _get_low_overlap_suppression_overlap_threshold() -> float:
    return _get_float_env("SEPARATION_LOW_OVERLAP_THRESHOLD", 0.2, minimum=0.0, maximum=1.0)


def _get_low_overlap_suppression_dominance_db() -> float:
    return _get_float_env("SEPARATION_LOW_OVERLAP_DOMINANCE_DB", 3.0, minimum=0.0, maximum=60.0)


def _get_low_overlap_suppression_loser_gain() -> float:
    return _get_float_env("SEPARATION_LOW_OVERLAP_LOSER_GAIN", 0.1, minimum=0.0, maximum=1.0)


def _get_low_overlap_suppression_active_floor_db() -> float:
    return _get_float_env("SEPARATION_LOW_OVERLAP_ACTIVE_FLOOR_DB", -45.0, minimum=-120.0, maximum=0.0)


def _get_speechbrain_residual_projection_amount() -> float:
    return _get_float_env("SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION_AMOUNT", 1.0, minimum=0.0, maximum=2.0)


def _get_speechbrain_residual_projection_limiter() -> float:
    return _get_float_env("SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION_LIMIT", 0.99, minimum=0.1, maximum=1.0)


def _get_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _get_float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


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
    if len(message) > 300:
        message = f"{message[:297]}..."
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
            savedir=str(_speechbrain_savedir(model_name)),
            run_opts={"device": device},
            local_strategy=local_strategy,
        )
        _SEPARATOR_CACHE[cache_key] = separator
    return separator


def _speechbrain_savedir(model_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip()).strip("._")
    return BACKEND_DIR / "models" / "speechbrain" / (safe_name or "separator")


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


def _get_separation_candidates() -> list[str]:
    raw = os.getenv("SEPARATION_CANDIDATES", "libri2mix,mossformer2,gated").strip()
    candidates = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return candidates or ["gated"]


def _get_expected_speaker_count(transcript: list[dict]) -> int | None:
    for name in ("SEPARATION_EXPECTED_SPEAKERS", "SEPARATION_MIN_TRACKS"):
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 1:
            return value

    speaker_count = len(_speaker_intervals(transcript))
    return speaker_count if speaker_count > 1 else None


def _transcript_overlap_ratio(transcript: list[dict]) -> float:
    intervals = [interval for values in _speaker_intervals(transcript).values() for interval in values]
    if len(intervals) < 2:
        return 0.0
    events: list[tuple[float, int]] = []
    total_start = math.inf
    total_end = 0.0
    for start, end in intervals:
        if end <= start:
            continue
        total_start = min(total_start, start)
        total_end = max(total_end, end)
        events.append((start, 1))
        events.append((end, -1))
    if not events or total_end <= total_start:
        return 0.0
    active = 0
    previous = total_start
    overlap_seconds = 0.0
    for timestamp, delta in sorted(events):
        if timestamp > previous and active >= 2:
            overlap_seconds += timestamp - previous
        active += delta
        previous = timestamp
    return max(0.0, min(1.0, overlap_seconds / (total_end - total_start)))


def _get_mossformer_overlap_threshold() -> float:
    raw = os.getenv("MOSSFORMER2_OVERLAP_THRESHOLD", "0.08").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.08
    return min(1.0, max(0.0, value))


def _get_mossformer_overlap_boost() -> float:
    raw = os.getenv("MOSSFORMER2_OVERLAP_BOOST", "8.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 8.0
    return min(20.0, max(0.0, value))


def _diagnostic_rerank_enabled() -> bool:
    return os.getenv("SEPARATION_DIAGNOSTIC_RERANK", "true").strip().lower() not in {"0", "false", "no", "off"}


def _get_mossformer_diagnostic_mixcorr_threshold() -> float:
    raw = os.getenv("MOSSFORMER2_DIAGNOSTIC_MIXCORR_THRESHOLD", "0.986").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.986
    return min(1.0, max(-1.0, value))


def _get_mossformer_diagnostic_track_overlap_threshold() -> float:
    raw = os.getenv("MOSSFORMER2_DIAGNOSTIC_TRACK_OVERLAP_THRESHOLD", "0.34").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.34
    return min(1.0, max(0.0, value))


def _get_mossformer_diagnostic_speechbrain_corr_max() -> float:
    raw = os.getenv("MOSSFORMER2_DIAGNOSTIC_SPEECHBRAIN_CORR_MAX", "-0.0015").strip()
    try:
        value = float(raw)
    except ValueError:
        return -0.0015
    return min(1.0, max(-1.0, value))


def _get_mossformer_diagnostic_bonus() -> float:
    raw = os.getenv("MOSSFORMER2_DIAGNOSTIC_BONUS", "8.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 8.0
    return min(20.0, max(0.0, value))


def _get_libri2mix_candidate_bonus() -> float:
    return _get_float_env("SEPARATION_LIBRI2MIX_BONUS", 1.5, minimum=-20.0, maximum=20.0)


def _quality_router_enabled() -> bool:
    return os.getenv("QUALITY_ROUTER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


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


def _speaker_intervals(transcript: list[dict]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    speaker_aliases: dict[str, str] = {}
    for item in transcript or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _parse_timestamp(item.get("start"))
        end = _parse_timestamp(item.get("end"))
        if math.isnan(start) or math.isnan(end) or end <= start:
            continue
        raw_speaker = str(item.get("speaker") or "").strip() or "说话人 A"
        speaker = speaker_aliases.setdefault(raw_speaker, _stable_speaker_label(raw_speaker, len(speaker_aliases)))
        grouped.setdefault(speaker, []).append((start, end))
    return {speaker: _merge_intervals(intervals) for speaker, intervals in grouped.items()}


def _write_gated_speaker_tracks(source_path: Path, intervals: dict[str, list[tuple[float, float]]]) -> list[dict]:
    soundfile = importlib.import_module("soundfile")
    numpy = importlib.import_module("numpy")
    data, sample_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
    if len(data) == 0 or sample_rate <= 0:
        raise RuntimeError("Source audio is empty")

    tracks = []
    background_gain = _get_gated_background_gain()
    target_gain = _get_gated_target_gain()
    limiter = _get_gated_limiter()
    fade_samples = int(sample_rate * _get_gated_fade_ms() / 1000.0)
    for index, (speaker, speaker_intervals) in enumerate(sorted(intervals.items()), start=1):
        gains = numpy.full((len(data), 1), background_gain, dtype="float32")
        for start, end in speaker_intervals:
            start_sample = max(0, min(len(data), int(start * sample_rate)))
            end_sample = max(start_sample, min(len(data), int(end * sample_rate)))
            if end_sample > start_sample:
                _apply_speaker_gain_window(gains, start_sample, end_sample, background_gain, target_gain, fade_samples)
        output = numpy.clip(data * gains, -limiter, limiter)

        safe_label = _safe_filename(speaker)
        output_path = UPLOAD_DIR / f"{source_path.stem}_{safe_label}_diarized.wav"
        soundfile.write(str(output_path), output, sample_rate)
        speech_seconds = sum(max(0.0, end - start) for start, end in speaker_intervals)
        background_db = _gain_to_db(background_gain)
        target_db = _gain_to_db(target_gain)
        tracks.append(
            {
                "track_id": f"{source_path.stem}_{safe_label}",
                "label": speaker,
                "audio_url": audio_url(output_path),
                "description": (
                    "FunASR speaker diarization gated track: "
                    f"突出{speaker}时间段({target_db:+.1f} dB)，"
                    f"其他说话人区域强衰减({background_db:.1f} dB)，讲话约{speech_seconds:.0f}s。"
                ),
            }
        )
    return tracks


def _apply_speaker_gain_window(gains, start: int, end: int, background_gain: float, target_gain: float, fade_samples: int) -> None:
    gains[start:end, :] = target_gain
    if fade_samples <= 0:
        return
    numpy = importlib.import_module("numpy")
    segment_len = max(0, end - start)
    fade_len = min(fade_samples, max(1, segment_len // 2))
    if fade_len <= 1:
        return
    fade_in = numpy.linspace(background_gain, target_gain, fade_len, dtype="float32").reshape(-1, 1)
    fade_out = numpy.linspace(target_gain, background_gain, fade_len, dtype="float32").reshape(-1, 1)
    gains[start : start + fade_len, :] = fade_in
    gains[end - fade_len : end, :] = fade_out


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


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _stable_speaker_label(raw_speaker: str, index: int) -> str:
    if raw_speaker.startswith("说话人 "):
        return raw_speaker
    if raw_speaker.startswith("璇磋瘽浜"):
        suffix = chr(ord("A") + index) if index < 26 else str(index + 1)
        return f"说话人 {suffix}"
    return raw_speaker


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return cleaned or "speaker"


def _get_gated_background_gain() -> float:
    raw = os.getenv("SPEAKER_TRACK_BACKGROUND_GAIN", "0.015").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.015
    return min(0.5, max(0.0, value))


def _get_gated_target_gain() -> float:
    raw = os.getenv("SPEAKER_TRACK_TARGET_GAIN", "1.25").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.25
    return min(2.0, max(0.1, value))


def _get_gated_fade_ms() -> float:
    raw = os.getenv("SPEAKER_TRACK_FADE_MS", "25").strip()
    try:
        value = float(raw)
    except ValueError:
        return 25.0
    return min(250.0, max(0.0, value))


def _get_gated_limiter() -> float:
    raw = os.getenv("SPEAKER_TRACK_LIMIT", "0.98").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.98
    return min(1.0, max(0.1, value))


def _gain_to_db(gain: float) -> float:
    if gain <= 0:
        return -120.0
    return 20.0 * math.log10(gain)


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


def _patch_torchaudio_soundfile_io(torchaudio: Any, torch: Any) -> None:
    try:
        if not hasattr(torchaudio, "load"):
            return
        soundfile = importlib.import_module("soundfile")
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


def _time_dimension(shape: tuple[int, ...]) -> int:
    if len(shape) >= 3:
        return 1
    if len(shape) == 2 and shape[0] > shape[1] and shape[1] <= 8:
        return 0
    if len(shape) == 2:
        return 1
    return 0
