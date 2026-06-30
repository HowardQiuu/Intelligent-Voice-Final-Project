from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy
import soundfile


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.separation_alignment_service import parse_textgrid  # noqa: E402


def main() -> int:
    args = parse_args()
    near_dir = Path(args.near_data_dir)
    output_dir = Path(args.output_dir)
    audio_out = output_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    groups = group_near_audio(near_dir / "audio_dir")
    if not groups:
        raise SystemExit(f"No near wav files found under {near_dir / 'audio_dir'}")

    manifest_rows = []
    for meeting, paths in sorted(groups.items()):
        if args.meeting and meeting != args.meeting:
            continue
        if len(paths) < args.min_speakers:
            continue
        row = build_meeting_mix(
            meeting=meeting,
            paths=paths,
            near_dir=near_dir,
            audio_out=audio_out,
            args=args,
        )
        manifest_rows.append(row)
        print(
            f"{meeting}: speakers={row['speaker_count']} seconds={row['duration_seconds']:.2f} "
            f"mix={row['mix_path']}"
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": "AliMeeting near/headset all-speaker close-talk mixtures",
        "warning": "Mixtures are synthesized from near/headset tracks. They are clearer than far-field recordings and should be reported as close-talk validation mixtures.",
        "near_data_dir": str(near_dir),
        "output_dir": str(output_dir),
        "meeting_count": len(manifest_rows),
        "total_duration_seconds": round(sum(float(row["duration_seconds"]) for row in manifest_rows), 3),
        "mask_inactive": args.mask_inactive,
        "source_rms": args.source_rms,
        "peak": args.peak,
        "chunk_seconds": args.chunk_seconds,
        "manifest_path": str(manifest_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an all-speaker close-talk mixture dataset from AliMeeting near/headset tracks.")
    parser.add_argument("--near-data-dir", default=str(REPO_ROOT / "data" / "source" / "Eval_Ali_near"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "near_mix_dataset_v1"))
    parser.add_argument("--meeting", default="", help="Optional single meeting id, for example R8009_M8019.")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--source-rms", type=float, default=0.06)
    parser.add_argument("--peak", type=float, default=0.95)
    parser.add_argument("--mask-inactive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def group_near_audio(audio_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(audio_dir.glob("*.wav")):
        if "_N_SPK" not in path.name:
            continue
        meeting = path.name.split("_N_SPK", 1)[0]
        groups[meeting].append(path)
    return groups


def build_meeting_mix(
    *,
    meeting: str,
    paths: list[Path],
    near_dir: Path,
    audio_out: Path,
    args: argparse.Namespace,
) -> dict:
    infos = [soundfile.info(str(path)) for path in paths]
    sample_rates = {info.samplerate for info in infos}
    channels = {info.channels for info in infos}
    if len(sample_rates) != 1:
        raise RuntimeError(f"{meeting} has mixed sample rates: {sample_rates}")
    if channels != {1}:
        raise RuntimeError(f"{meeting} expects mono near tracks, got channels={channels}")

    sample_rate = int(infos[0].samplerate)
    min_frames = min(int(info.frames) for info in infos)
    max_frames = max(int(info.frames) for info in infos)
    speakers = [speaker_from_path(path) for path in paths]
    source_paths = [audio_out / f"{meeting}_source_{index:02d}_{speaker}.wav" for index, speaker in enumerate(speakers, start=1)]
    mix_path = audio_out / f"{meeting}_near_all_speakers_mix.wav"

    intervals_by_speaker = {}
    if args.mask_inactive:
        for speaker in speakers:
            textgrid_path = near_dir / "textgrid_dir" / f"{meeting}_{speaker}.TextGrid"
            intervals_by_speaker[speaker] = load_intervals(textgrid_path) if textgrid_path.exists() else []

    chunk_frames = max(1, int(round(args.chunk_seconds * sample_rate)))
    peak = 0.0
    with open_sources(paths) as readers:
        with open_outputs([mix_path, *source_paths], sample_rate) as writers:
            frame_offset = 0
            while frame_offset < min_frames:
                frames = min(chunk_frames, min_frames - frame_offset)
                source_chunks = []
                for path, speaker, reader in zip(paths, speakers, readers):
                    chunk = reader.read(frames, dtype="float32", always_2d=False)
                    chunk = numpy.asarray(chunk, dtype="float32")
                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1).astype("float32")
                    if len(chunk) < frames:
                        chunk = numpy.pad(chunk, (0, frames - len(chunk))).astype("float32")
                    chunk = normalize_rms(chunk, args.source_rms)
                    if args.mask_inactive:
                        mask = activity_mask(intervals_by_speaker.get(speaker, []), frame_offset, frames, sample_rate)
                        chunk = (chunk * mask).astype("float32")
                    source_chunks.append(chunk)

                mixture = numpy.sum(source_chunks, axis=0).astype("float32")
                chunk_peak = float(max([numpy.max(numpy.abs(mixture)) if mixture.size else 0.0, *[numpy.max(numpy.abs(item)) if item.size else 0.0 for item in source_chunks]]))
                peak = max(peak, chunk_peak)
                writers[0].write(mixture)
                for writer, chunk in zip(writers[1:], source_chunks):
                    writer.write(chunk)
                frame_offset += frames

    scale = 1.0
    if peak > args.peak:
        scale = args.peak / peak
        scale_wav_in_place(mix_path, scale, sample_rate, chunk_frames)
        for source_path in source_paths:
            scale_wav_in_place(source_path, scale, sample_rate, chunk_frames)

    return {
        "meeting": meeting,
        "sample_rate": sample_rate,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "duration_seconds": round(min_frames / sample_rate, 3),
        "min_frames": min_frames,
        "max_frames": max_frames,
        "delta_frames": max_frames - min_frames,
        "mask_inactive": bool(args.mask_inactive),
        "mix_path": str(mix_path),
        "source_paths": [str(path) for path in source_paths],
        "near_source_paths": [str(path) for path in paths],
        "peak_before_scale": round(float(peak), 6),
        "peak_scale": round(float(scale), 6),
    }


def speaker_from_path(path: Path) -> str:
    return "N_SPK" + path.stem.split("_N_SPK", 1)[1]


def load_intervals(path: Path) -> list[tuple[float, float]]:
    return [(float(item["start_seconds"]), float(item["end_seconds"])) for item in parse_textgrid(path)]


def activity_mask(intervals: list[tuple[float, float]], frame_offset: int, frames: int, sample_rate: int) -> numpy.ndarray:
    mask = numpy.zeros(frames, dtype="float32")
    chunk_start = frame_offset / sample_rate
    chunk_end = (frame_offset + frames) / sample_rate
    for start, end in intervals:
        left = max(chunk_start, start)
        right = min(chunk_end, end)
        if right <= left:
            continue
        left_index = max(0, min(frames, int(round(left * sample_rate)) - frame_offset))
        right_index = max(left_index, min(frames, int(round(right * sample_rate)) - frame_offset))
        mask[left_index:right_index] = 1.0
    return mask


def normalize_rms(samples: numpy.ndarray, target_rms: float) -> numpy.ndarray:
    rms = float(numpy.sqrt(numpy.mean(numpy.square(samples)) + 1e-12)) if samples.size else 0.0
    if rms <= 1e-8:
        return samples.astype("float32")
    return (samples * (target_rms / rms)).astype("float32")


def scale_wav_in_place(path: Path, scale: float, sample_rate: int, chunk_frames: int) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with soundfile.SoundFile(str(path), "r") as reader:
        with soundfile.SoundFile(str(temp_path), "w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as writer:
            while True:
                chunk = reader.read(chunk_frames, dtype="float32", always_2d=False)
                if len(chunk) == 0:
                    break
                writer.write((numpy.asarray(chunk, dtype="float32") * scale).astype("float32"))
    temp_path.replace(path)


class open_sources:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.handles: list[soundfile.SoundFile] = []

    def __enter__(self) -> list[soundfile.SoundFile]:
        self.handles = [soundfile.SoundFile(str(path), "r") for path in self.paths]
        return self.handles

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.close()


class open_outputs:
    def __init__(self, paths: list[Path], sample_rate: int):
        self.paths = paths
        self.sample_rate = sample_rate
        self.handles: list[soundfile.SoundFile] = []

    def __enter__(self) -> list[soundfile.SoundFile]:
        self.handles = [
            soundfile.SoundFile(str(path), "w", samplerate=self.sample_rate, channels=1, subtype="PCM_16")
            for path in self.paths
        ]
        return self.handles

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
