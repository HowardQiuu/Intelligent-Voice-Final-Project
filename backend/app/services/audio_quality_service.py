from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .audio_service import UPLOAD_DIR, apply_audio_filter


EPSILON = 1e-12
DEFAULT_TARGET_LUFS = -18.0
DEFAULT_MAX_GAIN_DB = 24.0
DEFAULT_LOW_RMS_DBFS = -34.0
DEFAULT_LOW_PEAK_DBFS = -24.0


@dataclass(frozen=True)
class AudioQuality:
    rms_dbfs: float
    peak_dbfs: float
    silent_ratio: float
    clipping_ratio: float
    spectral_centroid_hz: float
    duration_seconds: float
    sample_rate: int


def analyze_audio_quality(path: Path | None) -> AudioQuality:
    if path is None or not path.exists():
        return _empty_quality()

    try:
        soundfile = __import__("soundfile")
        total_samples = 0
        sum_squares = 0.0
        peak = 0.0
        silent_blocks = 0
        block_count = 0
        clipping_samples = 0
        centroid_weighted = 0.0
        centroid_blocks = 0
        sample_rate = int(soundfile.info(str(path)).samplerate or 0)

        for block in soundfile.blocks(str(path), blocksize=16000, always_2d=True, dtype="float32"):
            if len(block) == 0:
                continue
            mono = block.mean(axis=1)
            abs_block = abs(mono)
            block_peak = float(abs_block.max(initial=0.0))
            block_rms = float(math.sqrt(float((mono * mono).mean()) + EPSILON))
            total_samples += len(mono)
            sum_squares += float((mono * mono).sum())
            peak = max(peak, block_peak)
            clipping_samples += int((abs_block >= 0.98).sum())
            block_count += 1
            if block_rms <= 0.006:
                silent_blocks += 1
            centroid = _spectral_centroid_hz(mono, sample_rate)
            if centroid > 0:
                centroid_weighted += centroid
                centroid_blocks += 1

        if total_samples <= 0 or sample_rate <= 0:
            return _empty_quality()
        rms = math.sqrt(sum_squares / total_samples)
        return AudioQuality(
            rms_dbfs=_dbfs(rms),
            peak_dbfs=_dbfs(peak),
            silent_ratio=silent_blocks / block_count if block_count else 0.0,
            clipping_ratio=clipping_samples / total_samples,
            spectral_centroid_hz=centroid_weighted / centroid_blocks if centroid_blocks else 0.0,
            duration_seconds=total_samples / sample_rate,
            sample_rate=sample_rate,
        )
    except Exception:
        return _empty_quality()


def apply_audibility_pregain(path: Path) -> tuple[Path, str, dict[str, str]]:
    if not _env_bool("ENHANCEMENT_PREGAIN_ENABLED", True):
        metrics = analyze_audio_quality(path)
        return path, "disabled", quality_metric_strings("pregain_input", metrics)

    before = analyze_audio_quality(path)
    if not _needs_pregain(before):
        return path, "skipped", quality_metric_strings("pregain_input", before)

    gain_db = _recommended_gain_db(before)
    output_path = UPLOAD_DIR / f"{path.stem}_pregain.wav"
    filter_spec = _pregain_filter(gain_db)
    if not apply_audio_filter(path, output_path, filter_spec):
        return path, "fallback", {
            **quality_metric_strings("pregain_input", before),
            "quality_pregain_status": "fallback",
            "quality_pregain_gain_db": f"{gain_db:.1f}",
        }

    after = analyze_audio_quality(output_path)
    return output_path, "ok", {
        **quality_metric_strings("pregain_input", before),
        **quality_metric_strings("pregain_output", after),
        "quality_pregain_status": "ok",
        "quality_pregain_gain_db": f"{gain_db:.1f}",
    }


def quality_metric_strings(prefix: str, quality: AudioQuality) -> dict[str, str]:
    return {
        f"{prefix}_rms_dbfs": _format_db(quality.rms_dbfs),
        f"{prefix}_peak_dbfs": _format_db(quality.peak_dbfs),
        f"{prefix}_silent_ratio": _format_percent(quality.silent_ratio),
        f"{prefix}_clipping_ratio": _format_percent(quality.clipping_ratio),
        f"{prefix}_spectral_centroid_hz": f"{quality.spectral_centroid_hz:.0f}",
    }


def score_audio_quality(quality: AudioQuality) -> float:
    if quality.sample_rate <= 0:
        return 0.0
    score = 55.0
    if quality.rms_dbfs < -36:
        score -= 25
    elif quality.rms_dbfs < -30:
        score -= 12
    elif quality.rms_dbfs > -10:
        score -= 10
    else:
        score += 12
    if quality.peak_dbfs < -24:
        score -= 12
    if quality.silent_ratio > 0.65:
        score -= 18
    elif quality.silent_ratio < 0.45:
        score += 8
    if quality.clipping_ratio > 0.01:
        score -= 25
    elif quality.clipping_ratio > 0.002:
        score -= 10
    if 300 <= quality.spectral_centroid_hz <= 4200:
        score += 7
    return max(0.0, min(100.0, score))


def score_enhancement_candidate(original: AudioQuality, candidate: AudioQuality) -> float:
    score = score_audio_quality(candidate)
    score += max(-12.0, min(18.0, candidate.rms_dbfs - original.rms_dbfs))
    if candidate.clipping_ratio > original.clipping_ratio + 0.002:
        score -= 15
    if candidate.silent_ratio < original.silent_ratio:
        score += min(8.0, (original.silent_ratio - candidate.silent_ratio) * 20)
    return max(0.0, min(120.0, score))


def _needs_pregain(quality: AudioQuality) -> bool:
    return quality.rms_dbfs <= _env_float("ENHANCEMENT_PREGAIN_LOW_RMS_DBFS", DEFAULT_LOW_RMS_DBFS) or (
        quality.peak_dbfs <= _env_float("ENHANCEMENT_PREGAIN_LOW_PEAK_DBFS", DEFAULT_LOW_PEAK_DBFS)
        and quality.rms_dbfs <= -28
    )


def _recommended_gain_db(quality: AudioQuality) -> float:
    target = _env_float("ENHANCEMENT_TARGET_LUFS", DEFAULT_TARGET_LUFS)
    max_gain = _env_float("ENHANCEMENT_MAX_GAIN_DB", DEFAULT_MAX_GAIN_DB)
    if quality.rms_dbfs <= -90:
        return max_gain
    return max(0.0, min(max_gain, target - quality.rms_dbfs))


def _pregain_filter(gain_db: float) -> str:
    target = _env_float("ENHANCEMENT_TARGET_LUFS", DEFAULT_TARGET_LUFS)
    return (
        "highpass=f=80,"
        f"volume={gain_db:.2f}dB,"
        "acompressor=threshold=-28dB:ratio=3:attack=5:release=80,"
        f"loudnorm=I={target:g}:TP=-2:LRA=8,"
        "alimiter=limit=0.95"
    )


def _spectral_centroid_hz(samples, sample_rate: int) -> float:
    if sample_rate <= 0 or len(samples) < 64:
        return 0.0
    try:
        numpy = __import__("numpy")
        window = samples[: min(len(samples), 8192)]
        spectrum = numpy.abs(numpy.fft.rfft(window))
        total = float(spectrum.sum())
        if total <= EPSILON:
            return 0.0
        freqs = numpy.fft.rfftfreq(len(window), d=1.0 / sample_rate)
        return float((freqs * spectrum).sum() / total)
    except Exception:
        return 0.0


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(EPSILON, float(value)))


def _empty_quality() -> AudioQuality:
    return AudioQuality(
        rms_dbfs=-120.0,
        peak_dbfs=-120.0,
        silent_ratio=1.0,
        clipping_ratio=0.0,
        spectral_centroid_hz=0.0,
        duration_seconds=0.0,
        sample_rate=0,
    )


def _format_db(value: float) -> str:
    return f"{value:.1f}"


def _format_percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.1%}"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default
