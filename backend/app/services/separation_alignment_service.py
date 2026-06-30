from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from .audio_service import get_audio_duration_seconds, resolve_static_url


TranscriptFn = Callable[[Path, str], dict]
REPO_ROOT = Path(__file__).resolve().parents[3]


def align_transcript_to_separation_tracks(transcript: list[dict], separated_tracks: list[dict]) -> tuple[list[dict], dict]:
    """Align ASR segments to already-produced separation tracks by segment energy.

    This step uses only the project pipeline outputs: ASR timestamps and separated
    track audio. It does not read TextGrid references, so it can run in normal
    inference without leaking evaluation labels into the pipeline.
    """

    track_audio = _load_track_audio(separated_tracks)
    if not transcript or not track_audio:
        return transcript, {
            "status": "skipped",
            "reason": "missing_transcript_or_tracks",
            "aligned_segments": 0,
            "multi_track_segments": 0,
        }

    aligned: list[dict] = []
    aligned_count = 0
    multi_count = 0
    for segment in transcript:
        start = _time_to_seconds(segment.get("start"))
        end = _time_to_seconds(segment.get("end"))
        scored = _score_segment_tracks(start, end, track_audio)
        active = _active_tracks(scored)
        primary = active[0] if active else (scored[0] if scored else None)
        next_segment = dict(segment)
        if primary:
            aligned_count += 1
            next_segment["primary_track_id"] = primary["track_id"]
            next_segment["primary_track_label"] = primary["label"]
            next_segment["separation_tracks"] = [item["track_id"] for item in active] or [primary["track_id"]]
            if len(next_segment["separation_tracks"]) > 1:
                multi_count += 1
        else:
            next_segment["primary_track_id"] = None
            next_segment["primary_track_label"] = None
            next_segment["separation_tracks"] = []
        aligned.append(next_segment)

    return aligned, {
        "status": "ok",
        "aligned_segments": aligned_count,
        "total_segments": len(transcript),
        "multi_track_segments": multi_count,
        "track_count": len(track_audio),
    }


def build_textgrid_separation_evaluation(
    *,
    separated_tracks: list[dict],
    display_name: str = "",
    reference_audio_path: Path | None = None,
    transcribe_track: TranscriptFn | None = None,
) -> dict:
    """Evaluate separation output against TextGrid after the pipeline has run.

    TextGrid is treated strictly as ground-truth evidence. It is not used to
    decide the separation backend, relabel separation tracks, or align ASR
    segments during inference.
    """

    textgrid_path = find_textgrid_for_audio(reference_audio_path, display_name)
    if textgrid_path is None:
        return {"status": "no_textgrid", "source": "none", "track_matches": [], "overlap_segments": []}

    reference_segments = parse_textgrid(textgrid_path)
    speaker_texts = _speaker_texts(reference_segments)
    evaluation = {
        "status": "reference_loaded",
        "source": "textgrid",
        "textgrid_path": str(textgrid_path),
        "reference_speaker_count": len(speaker_texts),
        "reference_segment_count": len(reference_segments),
        "reference_overlap_ratio": round(_overlap_ratio(reference_segments), 4),
        "track_count": len(separated_tracks),
        "track_matches": [],
        "overlap_segments": _overlap_segments(reference_segments),
    }
    if not separated_tracks:
        return {**evaluation, "status": "no_tracks"}

    used_speakers: set[str] = set()
    track_matches = []
    for index, track in enumerate(separated_tracks, start=1):
        track_path = resolve_static_url(str(track.get("audio_url", "")))
        track_text = ""
        transcription_status = "skipped"
        if transcribe_track is not None and track_path is not None and track_path.exists():
            try:
                result = transcribe_track(track_path, f"{display_name or 'track'}::{track.get('label', index)}")
                track_text = str(result.get("enhanced_asr_text") or "").strip()
                transcription_status = str(result.get("signal_metrics", {}).get("ASR 状态", "ok"))
            except Exception as exc:
                transcription_status = f"failed:{exc.__class__.__name__}"

        candidates = _score_track_text_against_speakers(
            track_text=track_text,
            speaker_texts=speaker_texts,
            used_speakers=used_speakers,
        )
        best = candidates[0] if candidates else {"speaker": "", "score": 0.0}
        if best["speaker"] and best["score"] > 0:
            used_speakers.add(best["speaker"])
        track_matches.append(
            {
                "track_id": str(track.get("track_id", f"track_{index}")),
                "track_label": str(track.get("label", f"track {index}")),
                "matched_reference_speaker": best["speaker"],
                "text_similarity": round(float(best["score"]), 4),
                "match_method": "track_asr_vs_textgrid" if track_text else "not_transcribed",
                "track_text": track_text,
                "transcription_status": transcription_status,
                "candidate_scores": [
                    {"speaker": item["speaker"], "score": round(float(item["score"]), 4)}
                    for item in candidates[: min(5, len(candidates))]
                ],
            }
        )

    matched = sum(1 for item in track_matches if item.get("matched_reference_speaker") and item.get("text_similarity", 0) > 0)
    status = "evaluated" if transcribe_track is not None else "reference_loaded_track_asr_skipped"
    return {**evaluation, "status": status, "matched_track_count": matched, "track_matches": track_matches}


def alignment_metrics(alignment: dict) -> dict[str, str]:
    if not alignment:
        return {"asr_separation_alignment_status": "missing"}
    return {
        "asr_separation_alignment_status": str(alignment.get("status", "")),
        "asr_separation_alignment_segments": str(alignment.get("aligned_segments", 0)),
        "asr_separation_alignment_multi_track_segments": str(alignment.get("multi_track_segments", 0)),
    }


def evaluation_metrics(evaluation: dict) -> dict[str, str]:
    if not evaluation:
        return {"textgrid_separation_eval_status": "missing"}
    metrics = {
        "textgrid_separation_eval_status": str(evaluation.get("status", "")),
        "textgrid_reference_speakers": str(evaluation.get("reference_speaker_count", 0)),
        "textgrid_reference_overlap_ratio": f"{float(evaluation.get('reference_overlap_ratio', 0.0)):.3f}",
        "textgrid_eval_matched_tracks": str(evaluation.get("matched_track_count", 0)),
    }
    for index, item in enumerate(evaluation.get("track_matches", []), start=1):
        speaker = item.get("matched_reference_speaker") or "unmatched"
        score = float(item.get("text_similarity", 0.0))
        metrics[f"textgrid_eval_track_{index}"] = f"{item.get('track_label', item.get('track_id'))}->{speaker} ({score:.2f})"
    return metrics


def should_transcribe_evaluation_tracks(reference_audio_path: Path | None = None, display_name: str = "") -> bool:
    mode = os.getenv("SEPARATION_EVAL_TRANSCRIBE_TRACKS", "auto").strip().lower() or "auto"
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off", "disabled"}:
        return False
    if find_textgrid_for_audio(reference_audio_path, display_name) is None:
        return False
    if reference_audio_path is None or not reference_audio_path.exists():
        return False
    duration = get_audio_duration_seconds(reference_audio_path)
    if duration is None:
        return False
    return duration <= _evaluation_max_transcribe_seconds()


def parse_textgrid(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    segments: list[dict] = []
    current_speaker = ""
    current_start: float | None = None
    current_end: float | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("name = "):
            current_speaker = _unquote(line.split("=", 1)[1].strip())
            continue
        if line.startswith("xmin = "):
            current_start = _parse_float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("xmax = "):
            current_end = _parse_float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("text = "):
            content = _clean_text(_unquote(line.split("=", 1)[1].strip()))
            if current_speaker and current_start is not None and current_end is not None and content:
                segments.append(
                    {
                        "speaker": current_speaker,
                        "start_seconds": current_start,
                        "end_seconds": current_end,
                        "start": _format_time(current_start),
                        "end": _format_time(current_end),
                        "text": content,
                    }
                )
            current_start = None
            current_end = None
    return sorted(segments, key=lambda item: (item["start_seconds"], item["end_seconds"], item["speaker"]))


def find_textgrid_for_audio(reference_audio_path: Path | None = None, display_name: str = "") -> Path | None:
    candidates = []
    if reference_audio_path is not None:
        candidates.extend([reference_audio_path.name, reference_audio_path.stem])
    if display_name:
        candidates.extend([Path(display_name).name, Path(display_name).stem])
    roots = [
        REPO_ROOT / "data" / "Eval_Ali_far" / "textgrid_dir",
        REPO_ROOT / "data" / "Eval_Ali_near" / "textgrid_dir",
    ]
    for value in candidates:
        meeting_id = _meeting_id_from_name(value)
        if not meeting_id:
            continue
        for root in roots:
            path = root / f"{meeting_id}.TextGrid"
            if path.exists():
                return path
    return None


def _load_track_audio(separated_tracks: list[dict]) -> list[dict]:
    loaded = []
    for track in separated_tracks:
        path = resolve_static_url(str(track.get("audio_url", "")))
        if path is None or not path.exists():
            continue
        try:
            import soundfile

            samples, sample_rate = soundfile.read(str(path), always_2d=False)
            loaded.append(
                {
                    "track_id": str(track.get("track_id", path.stem)),
                    "label": str(track.get("label", track.get("track_id", path.stem))),
                    "samples": samples,
                    "sample_rate": int(sample_rate),
                }
            )
        except Exception:
            continue
    return loaded


def _score_segment_tracks(start: float, end: float, track_audio: list[dict]) -> list[dict]:
    if end <= start:
        return []
    scored = []
    for track in track_audio:
        sample_rate = track["sample_rate"]
        start_sample = max(0, int(start * sample_rate))
        end_sample = max(start_sample + 1, int(end * sample_rate))
        samples = track["samples"]
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        window = samples[start_sample:min(end_sample, len(samples))]
        if len(window) == 0:
            rms = 0.0
        else:
            import numpy

            rms = float(numpy.sqrt(numpy.mean(numpy.square(window.astype("float64"))) + 1e-12))
        scored.append({"track_id": track["track_id"], "label": track["label"], "rms": rms})
    return sorted(scored, key=lambda item: item["rms"], reverse=True)


def _active_tracks(scored: list[dict]) -> list[dict]:
    if not scored:
        return []
    max_rms = max(float(scored[0]["rms"]), 1e-12)
    ratio = _alignment_active_ratio()
    floor = _alignment_active_floor()
    return [item for item in scored if float(item["rms"]) >= floor and float(item["rms"]) >= max_rms * ratio]


def _score_track_text_against_speakers(
    *,
    track_text: str,
    speaker_texts: dict[str, str],
    used_speakers: set[str],
) -> list[dict]:
    normalized_track_text = _normalize_text(track_text)
    candidates = []
    for speaker, reference_text in speaker_texts.items():
        score = _char_f1(normalized_track_text, _normalize_text(reference_text)) if normalized_track_text else 0.0
        if speaker in used_speakers:
            score -= 0.05
        candidates.append({"speaker": speaker, "score": max(0.0, score)})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _speaker_texts(segments: list[dict]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for item in segments:
        grouped.setdefault(str(item["speaker"]), []).append(str(item.get("text", "")))
    return {speaker: "".join(parts) for speaker, parts in grouped.items()}


def _overlap_ratio(segments: list[dict]) -> float:
    intervals = [(float(item["start_seconds"]), float(item["end_seconds"])) for item in segments if item["end_seconds"] > item["start_seconds"]]
    if not intervals:
        return 0.0
    events: list[tuple[float, int]] = []
    first = min(start for start, _ in intervals)
    last = max(end for _, end in intervals)
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    active = 0
    previous = first
    overlap = 0.0
    for timestamp, delta in sorted(events):
        if timestamp > previous and active >= 2:
            overlap += timestamp - previous
        active += delta
        previous = timestamp
    return max(0.0, min(1.0, overlap / (last - first))) if last > first else 0.0


def _overlap_segments(segments: list[dict], limit: int = 20) -> list[dict]:
    overlaps = []
    sorted_segments = sorted(segments, key=lambda item: (item["start_seconds"], item["end_seconds"]))
    for index, first in enumerate(sorted_segments):
        active = [first]
        for second in sorted_segments[index + 1 :]:
            if second["start_seconds"] >= first["end_seconds"]:
                break
            if second["end_seconds"] > first["start_seconds"] and second["speaker"] != first["speaker"]:
                active.append(second)
        if len(active) < 2:
            continue
        start = max(item["start_seconds"] for item in active)
        end = min(item["end_seconds"] for item in active)
        if end <= start:
            continue
        speakers = sorted({str(item["speaker"]) for item in active})
        overlaps.append({"start": _format_time(start), "end": _format_time(end), "speakers": speakers})
        if len(overlaps) >= limit:
            break
    return overlaps


def _char_f1(text: str, reference: str) -> float:
    if not text or not reference:
        return 0.0
    text_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for char in text:
        text_counts[char] = text_counts.get(char, 0) + 1
    for char in reference:
        ref_counts[char] = ref_counts.get(char, 0) + 1
    overlap = sum(min(count, ref_counts.get(char, 0)) for char, count in text_counts.items())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(text)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _alignment_active_ratio() -> float:
    return _float_env("ASR_SEPARATION_ALIGNMENT_ACTIVE_RATIO", 0.35, minimum=0.0, maximum=1.0)


def _alignment_active_floor() -> float:
    return _float_env("ASR_SEPARATION_ALIGNMENT_ACTIVE_FLOOR", 1e-5, minimum=0.0, maximum=1.0)


def _evaluation_max_transcribe_seconds() -> float:
    return _float_env("SEPARATION_EVAL_MAX_TRANSCRIBE_SECONDS", 120.0, minimum=1.0, maximum=3600.0)


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _meeting_id_from_name(name: str) -> str:
    stem = Path(name).stem
    match = re.match(r"^(R\d+_M\d+)", stem)
    return match.group(1) if match else stem


def _normalize_text(value: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》|]+", "", value).lower()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _time_to_seconds(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(text)
    except ValueError:
        return 0.0


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
