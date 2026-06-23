from __future__ import annotations

import math
import os
from pathlib import Path

from .audio_service import get_audio_duration_seconds


DEFAULT_CHUNK_SECONDS = 60.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 5.0


def build_chunk_plan(audio_path: Path | None) -> dict:
    duration = get_audio_duration_seconds(audio_path) if audio_path else None
    chunk_seconds = _get_float_env("CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)
    overlap_seconds = min(_get_float_env("CHUNK_OVERLAP_SECONDS", DEFAULT_CHUNK_OVERLAP_SECONDS), chunk_seconds / 2)

    if duration is None or duration <= 0:
        return {
            "summary": "Unable to read audio duration; using single-pass fallback.",
            "duration_seconds": "",
            "chunk_seconds": f"{chunk_seconds:.0f}",
            "overlap_seconds": f"{overlap_seconds:.0f}",
            "chunk_count": "1",
            "chunks": [
                {
                    "chunk_id": "chunk_001",
                    "start": "00:00",
                    "end": "unknown",
                    "duration_seconds": 0.0,
                    "status": "fallback",
                    "description": "Duration metadata unavailable; keep one logical chunk.",
                }
            ],
        }

    step_seconds = max(1.0, chunk_seconds - overlap_seconds)
    chunk_count = max(1, math.ceil(max(duration - overlap_seconds, 0) / step_seconds))
    chunks = []
    for index in range(chunk_count):
        start = index * step_seconds
        end = min(start + chunk_seconds, duration)
        chunks.append(
            {
                "chunk_id": f"chunk_{index + 1:03d}",
                "start": _format_seconds(start),
                "end": _format_seconds(end),
                "duration_seconds": round(max(0.0, end - start), 2),
                "status": "planned",
                "description": "Reserved chunk for enhancement/separation/ASR processing.",
            }
        )
        if end >= duration:
            break

    mode = "chunked" if len(chunks) > 1 else "single-pass"
    return {
        "summary": (
            f"{mode}: {len(chunks)} chunk(s), "
            f"{chunk_seconds:.0f}s window, {overlap_seconds:.0f}s overlap"
        ),
        "duration_seconds": f"{duration:.2f}",
        "chunk_seconds": f"{chunk_seconds:.0f}",
        "overlap_seconds": f"{overlap_seconds:.0f}",
        "chunk_count": str(len(chunks)),
        "chunks": chunks,
    }


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, value)


def _format_seconds(value: float) -> str:
    total = int(round(value))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
