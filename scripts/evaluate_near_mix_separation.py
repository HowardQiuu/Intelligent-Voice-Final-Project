from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy
import soundfile


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_service import UPLOAD_DIR, audio_url, resolve_static_url  # noqa: E402
from app.services.separation_service import separate_with_quality_router  # noqa: E402
from app.services.speaker_count_estimation_service import estimate_speaker_count_from_tracks  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    rows = load_manifest(Path(args.manifest))
    selected = [row for row in rows if not args.meeting or row["meeting"] == args.meeting]
    selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No matching near-mix rows found.")

    results = []
    for row in selected:
        results.append(evaluate_row(row, args))

    passed = all(
        item["track_count_ok"]
        and item["estimated_count_ok"]
        and item["min_matched_similarity"] >= args.threshold
        and item["min_estimated_track_quality"] >= args.threshold
        for item in results
    )
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": args.threshold,
        "seconds": args.seconds,
        "separation_uses_reference_sources": False,
        "results": results,
        "passed": passed,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_summary_report(report) if args.summary_only else report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


def _summary_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": report.get("created_at"),
        "threshold": report.get("threshold"),
        "seconds": report.get("seconds"),
        "passed": report.get("passed"),
        "results": [
            {
                "meeting": item.get("meeting"),
                "expected_speakers": item.get("expected_speakers"),
                "track_count_ok": item.get("track_count_ok"),
                "estimated_count_ok": item.get("estimated_count_ok"),
                "global_estimated_speaker_count": item.get("global_estimated_speaker_count"),
                "speaker_count_embedding_backend": item.get("speaker_count_embedding_backend"),
                "speaker_count_embedding_backend_status": item.get("speaker_count_embedding_backend_status"),
                "speaker_count_cluster_stability": item.get("speaker_count_cluster_stability"),
                "estimated_counts": item.get("estimated_counts"),
                "min_estimated_track_quality": item.get("min_estimated_track_quality"),
                "min_matched_similarity": item.get("min_matched_similarity"),
            }
            for item in report.get("results", [])
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run blind separation on near-mix audio, then evaluate separated tracks "
            "against clean source tracks after inference."
        )
    )
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "near_mix_dataset_v1" / "manifest.jsonl"))
    parser.add_argument("--meeting", default="R8001_M8004")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=20.0, help="Evaluate an initial clip to keep model smoke tests practical.")
    parser.add_argument("--offset", type=float, default=-1.0, help="Clip start seconds; negative means choose a high-energy window from the mixture only.")
    parser.add_argument("--windows", type=int, default=10, help="Number of non-overlapping mix-selected windows to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output", default=str(REPO_ROOT / ".runtime" / "near_mix_separation_eval.json"))
    parser.add_argument("--candidate", default="", help="Optional SEPARATION_CANDIDATES override, for example libri2mix.")
    parser.add_argument("--summary-only", action="store_true", help="Print a compact summary while still writing full JSON to --output.")
    parser.add_argument(
        "--expected-speakers",
        type=int,
        default=0,
        help="Override expected track count for evaluation only; not passed into separation.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mix_path = REPO_ROOT / row["mix_path"]
    expected_speakers = int(args.expected_speakers or row["speaker_count"])
    reference_paths = [REPO_ROOT / path for path in row["source_paths"]]
    offsets = [args.offset] if args.offset >= 0 else _select_high_energy_offsets(mix_path, args.seconds, args.windows)
    window_results = []
    for window_index, offset in enumerate(offsets, start=1):
        window_results.append(evaluate_window(row, args, expected_speakers, reference_paths, window_index, offset))

    per_reference = aggregate_reference_scores(window_results, reference_paths)
    min_similarity = min((item["best_similarity"] for item in per_reference), default=0.0)
    track_count_ok = all(item["track_count_ok"] for item in window_results)
    estimated_counts = [
        int(window.get("speaker_count_estimation", {}).get("estimated_speaker_count", 0))
        for window in window_results
    ]
    global_tracks = [
        {
            "track_id": f"w{window['window_index']}_{track_index}_{Path(str(track.get('audio_url', ''))).name}",
            "label": f"window {window['window_index']} track {track_index}",
            "audio_url": track.get("audio_url", ""),
        }
        for window in window_results
        for track_index, track in enumerate(window.get("tracks", []) or [], start=1)
    ]
    global_estimation = estimate_speaker_count_from_tracks(global_tracks, max_tracks=max(1, len(global_tracks)))
    global_estimated_speaker_count = int(global_estimation.get("global_estimated_speaker_count", 0))
    estimated_count_ok = bool(global_tracks) and global_estimated_speaker_count == expected_speakers
    min_estimated_quality = min(
        (
            float(track.get("quality_score", 0.0))
            for track in global_estimation.get("tracks", []) or []
            if track.get("accepted")
        ),
        default=0.0,
    )
    return {
        "meeting": row["meeting"],
        "expected_speakers": expected_speakers,
        "window_count": len(window_results),
        "track_count_ok": track_count_ok,
        "estimated_count_ok": estimated_count_ok,
        "global_estimated_speaker_count": global_estimated_speaker_count,
        "speaker_count_embedding_backend": global_estimation.get("embedding_backend", ""),
        "speaker_count_embedding_backend_status": global_estimation.get("embedding_backend_status", ""),
        "speaker_count_cluster_stability": global_estimation.get("cluster_stability", 0.0),
        "speaker_count_global_cluster_summary": global_estimation.get("clusters", []),
        "estimated_counts": estimated_counts,
        "min_estimated_track_quality": round(min_estimated_quality, 4),
        "min_matched_similarity": round(min_similarity, 4),
        "all_tracks_over_threshold": bool(per_reference) and min_similarity >= args.threshold,
        "per_reference": per_reference,
        "windows": window_results,
    }


def evaluate_window(
    row: dict[str, Any],
    args: argparse.Namespace,
    expected_speakers: int,
    reference_paths: list[Path],
    window_index: int,
    offset: float,
) -> dict[str, Any]:
    clip_path, clip_offset = _write_clip(
        REPO_ROOT / row["mix_path"],
        f"{row['meeting']}_eval_mix_{uuid.uuid4().hex[:8]}.wav",
        args.seconds,
        offset,
    )

    previous_env = {
        name: os.environ.get(name)
        for name in [
            "QUALITY_ROUTER_ENABLED",
            "SEPARATION_EXPECTED_SPEAKERS",
            "SEPARATION_CANDIDATES",
            "SEPARATION_RECURSIVE_EXPANSION",
        ]
    }
    try:
        os.environ["QUALITY_ROUTER_ENABLED"] = "true"
        os.environ.pop("SEPARATION_EXPECTED_SPEAKERS", None)
        os.environ["SEPARATION_RECURSIVE_EXPANSION"] = "true"
        if args.candidate:
            os.environ["SEPARATION_CANDIDATES"] = args.candidate
        separation = separate_with_quality_router(audio_url(clip_path), transcript=[])
    finally:
        _restore_env(previous_env)
        clip_path.unlink(missing_ok=True)

    # Reference sources are intentionally loaded only after separation has completed.
    track_paths = [
        path
        for path in (resolve_static_url(str(track.get("audio_url", ""))) for track in separation.get("tracks", []))
        if path is not None and path.exists()
    ]
    matches = match_tracks_to_references(track_paths, reference_paths, args.seconds, clip_offset)
    return {
        "window_index": window_index,
        "clip_offset_seconds": round(clip_offset, 3),
        "separation_method": separation.get("method", ""),
        "separation_status": separation.get("status", ""),
        "track_count": len(track_paths),
        "track_count_ok": len(track_paths) == expected_speakers,
        "min_window_similarity": round(min((item["similarity"] for item in matches), default=0.0), 4),
        "matches": matches,
        "tracks": [
            {"track_id": track.get("track_id", f"track_{index}"), "audio_url": track.get("audio_url", "")}
            for index, track in enumerate(separation.get("tracks", []) or [], start=1)
        ],
        "speaker_count_estimation": separation.get("speaker_count_estimation", {}),
        "metrics": separation.get("metrics", {}),
    }


def aggregate_reference_scores(window_results: list[dict[str, Any]], reference_paths: list[Path]) -> list[dict[str, Any]]:
    output = []
    for reference in reference_paths:
        candidates = [
            {**match, "window_index": window["window_index"], "clip_offset_seconds": window["clip_offset_seconds"]}
            for window in window_results
            for match in window.get("matches", [])
            if match.get("reference") == reference.name
        ]
        best = max(candidates, key=lambda item: item.get("similarity", 0.0), default=None)
        output.append(
            {
                "reference": reference.name,
                "best_similarity": round(float(best.get("similarity", 0.0)), 4) if best else 0.0,
                "best_track": best.get("track", "") if best else "",
                "best_window_index": best.get("window_index", 0) if best else 0,
                "best_clip_offset_seconds": best.get("clip_offset_seconds", 0.0) if best else 0.0,
                "best_reference_active_ratio": best.get("reference_active_ratio", 0.0) if best else 0.0,
            }
        )
    return output


def match_tracks_to_references(
    track_paths: list[Path],
    reference_paths: list[Path],
    seconds: float,
    offset: float,
) -> list[dict[str, Any]]:
    if not track_paths or not reference_paths:
        return []
    track_audio = [_read_mono(path, seconds) for path in track_paths]
    reference_audio = [_read_mono(path, seconds, offset=offset) for path in reference_paths]
    reference_activity = [reference_activity_stats(reference) for reference in reference_audio]
    pair_scores = [
        [track_reference_similarity(track, reference) for reference in reference_audio]
        for track in track_audio
    ]
    track_count = len(track_paths)
    reference_count = len(reference_paths)
    best_order: tuple[int, ...] = tuple(range(min(track_count, reference_count)))
    best_score = -1.0
    for order in itertools.permutations(range(reference_count), min(track_count, reference_count)):
        score = sum(pair_scores[track_index][ref_index] for track_index, ref_index in enumerate(order))
        if score > best_score:
            best_score = score
            best_order = order
    matches = []
    for track_index, ref_index in enumerate(best_order):
        matches.append(
            {
                "track": track_paths[track_index].name,
                "reference": reference_paths[ref_index].name,
                "similarity": round(pair_scores[track_index][ref_index], 4),
                "reference_active_ratio": round(reference_activity[ref_index]["active_ratio"], 4),
                "reference_rms": round(reference_activity[ref_index]["rms"], 6),
            }
        )
    return matches


def track_reference_similarity(track: numpy.ndarray, reference: numpy.ndarray) -> float:
    track, reference = _pad_pair(track, reference)
    if len(track) == 0:
        return 0.0
    active = numpy.abs(reference) >= max(1e-5, float(numpy.percentile(numpy.abs(reference), 75)) * 0.25)
    if int(numpy.sum(active)) >= max(128, len(reference) // 100):
        track_score = track[active]
        reference_score = reference[active]
    else:
        track_score = track
        reference_score = reference
    corr = abs(_safe_corr(track_score, reference_score))
    activity = _activity_f1(track, reference)
    sisdr = _si_sdr_similarity(track, reference)
    corr_activity = 0.7 * corr + 0.3 * activity
    return float(max(0.0, min(1.0, max(corr_activity, sisdr))))


def reference_activity_stats(reference: numpy.ndarray) -> dict[str, float]:
    if len(reference) == 0:
        return {"active_ratio": 0.0, "rms": 0.0}
    rms = float(numpy.sqrt(numpy.mean(reference * reference) + 1e-12))
    frame_rms = _frame_rms(reference)
    threshold = max(1e-5, float(numpy.percentile(frame_rms, 75)) * 0.25)
    return {
        "active_ratio": float(numpy.mean(frame_rms >= threshold)) if len(frame_rms) else 0.0,
        "rms": rms,
    }


def _si_sdr_similarity(track: numpy.ndarray, reference: numpy.ndarray) -> float:
    reference_energy = float(numpy.dot(reference, reference))
    if reference_energy <= 1e-9:
        return 0.0
    scale = float(numpy.dot(track, reference) / reference_energy)
    target = scale * reference
    noise = track - target
    target_energy = float(numpy.dot(target, target))
    noise_energy = float(numpy.dot(noise, noise))
    if target_energy <= 1e-12:
        return 0.0
    si_sdr = 10.0 * numpy.log10((target_energy + 1e-12) / (noise_energy + 1e-12))
    return float(max(0.0, min(1.0, (si_sdr + 10.0) / 30.0)))


def _write_clip(source_path: Path, name: str, seconds: float, offset: float) -> tuple[Path, float]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    data, sample_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
    if offset < 0:
        offset = _select_high_energy_offset(data.mean(axis=1).astype("float32"), int(sample_rate), seconds)
    start = max(0, min(len(data), int(sample_rate * offset)))
    frame_count = min(len(data) - start, int(sample_rate * seconds))
    output_path = UPLOAD_DIR / name
    soundfile.write(str(output_path), data[start : start + frame_count], sample_rate)
    return output_path, float(start / sample_rate)


def _read_mono(path: Path, seconds: float, *, offset: float = 0.0) -> numpy.ndarray:
    data, sample_rate = soundfile.read(str(path), always_2d=True, dtype="float32")
    start = max(0, min(len(data), int(sample_rate * offset)))
    frame_count = min(len(data) - start, int(sample_rate * seconds))
    return data[start : start + frame_count].mean(axis=1).astype("float32")


def _select_high_energy_offset(samples: numpy.ndarray, sample_rate: int, seconds: float) -> float:
    offsets = _select_high_energy_offsets_from_samples(samples, sample_rate, seconds, 1)
    return offsets[0] if offsets else 0.0


def _select_high_energy_offsets(source_path: Path, seconds: float, count: int) -> list[float]:
    data, sample_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
    return _select_high_energy_offsets_from_samples(
        data.mean(axis=1).astype("float32"),
        int(sample_rate),
        seconds,
        max(1, count),
    )


def _select_high_energy_offsets_from_samples(
    samples: numpy.ndarray,
    sample_rate: int,
    seconds: float,
    count: int,
) -> list[float]:
    window = max(1, int(sample_rate * seconds))
    if len(samples) <= window:
        return [0.0]
    hop = max(1, int(sample_rate * min(1.0, max(0.25, seconds / 4.0))))
    candidates = []
    for start in range(0, len(samples) - window + 1, hop):
        segment = samples[start : start + window]
        energy = float(numpy.mean(segment * segment))
        candidates.append((start, energy))
    selected: list[int] = []
    min_distance = max(1, window)
    for start, _energy in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(start - previous) >= min_distance for previous in selected):
            selected.append(start)
            if len(selected) >= count:
                break
    if not selected and candidates:
        selected.append(max(candidates, key=lambda item: item[1])[0])
    return [float(start / sample_rate) for start in sorted(selected)]


def _pad_pair(first: numpy.ndarray, second: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    target = max(len(first), len(second))
    if target == 0:
        return first, second
    return _pad(first, target), _pad(second, target)


def _pad(value: numpy.ndarray, target: int) -> numpy.ndarray:
    if len(value) >= target:
        return value[:target].astype("float32")
    output = numpy.zeros(target, dtype="float32")
    output[: len(value)] = value
    return output


def _safe_corr(first: numpy.ndarray, second: numpy.ndarray) -> float:
    first = first - float(numpy.mean(first))
    second = second - float(numpy.mean(second))
    denom = float(numpy.linalg.norm(first) * numpy.linalg.norm(second))
    if denom <= 1e-9:
        return 0.0
    return float(numpy.dot(first, second) / denom)


def _activity_f1(track: numpy.ndarray, reference: numpy.ndarray) -> float:
    track_rms = _frame_rms(track)
    reference_rms = _frame_rms(reference)
    frame_count = min(len(track_rms), len(reference_rms))
    if frame_count == 0:
        return 0.0
    track_active = track_rms[:frame_count] >= max(1e-5, float(numpy.percentile(track_rms, 75)) * 0.25)
    reference_active = reference_rms[:frame_count] >= max(1e-5, float(numpy.percentile(reference_rms, 75)) * 0.25)
    true_positive = float(numpy.sum(track_active & reference_active))
    precision = true_positive / max(1.0, float(numpy.sum(track_active)))
    recall = true_positive / max(1.0, float(numpy.sum(reference_active)))
    if precision + recall <= 1e-9:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _frame_rms(samples: numpy.ndarray) -> numpy.ndarray:
    frame = 1024
    hop = 512
    if len(samples) < frame:
        return numpy.asarray([float(numpy.sqrt(numpy.mean(samples * samples) + 1e-12))], dtype="float32")
    return numpy.asarray(
        [
            float(numpy.sqrt(numpy.mean(samples[index : index + frame] ** 2) + 1e-12))
            for index in range(0, len(samples) - frame + 1, hop)
        ],
        dtype="float32",
    )


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
