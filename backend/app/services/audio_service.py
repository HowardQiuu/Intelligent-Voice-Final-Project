from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_AUDIO_DIR = BASE_DIR / "static" / "audio"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"


def ensure_audio_dirs() -> None:
    STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def has_ffmpeg() -> bool:
    return _ffmpeg_executable() is not None


def ffmpeg_executable() -> str | None:
    return _ffmpeg_executable()


def normalize_upload(input_path: Path, output_name: str) -> Path:
    ensure_audio_dirs()
    output_path = UPLOAD_DIR / f"{output_name}.wav"
    if output_path.resolve() == input_path.resolve():
        output_path = UPLOAD_DIR / f"{output_name}_normalized.wav"
    ffmpeg = _ffmpeg_executable()
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-filter:a",
            "loudnorm",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, OSError):
            shutil.copyfile(input_path, output_path)
    else:
        shutil.copyfile(input_path, output_path)
    return output_path


def _ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        imageio_ffmpeg = __import__("imageio_ffmpeg")
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def generate_demo_audio(case_id: str, noisy: bool = False) -> Path:
    ensure_audio_dirs()
    suffix = "original" if noisy else "enhanced"
    path = STATIC_AUDIO_DIR / f"{case_id}_{suffix}.wav"
    if path.exists():
        return path

    sample_rate = 16000
    duration_seconds = 6
    total = sample_rate * duration_seconds
    freqs = {
        "clear_meeting": (220, 330),
        "noisy_meeting": (180, 280),
        "overlap_meeting": (240, 360),
    }.get(case_id, (220, 330))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(total):
            t = i / sample_rate
            envelope = 0.45 + 0.35 * math.sin(2 * math.pi * 1.7 * t) ** 2
            speech = (
                0.42 * math.sin(2 * math.pi * freqs[0] * t)
                + 0.24 * math.sin(2 * math.pi * freqs[1] * t)
                + 0.08 * math.sin(2 * math.pi * 720 * t)
            ) * envelope
            noise = 0.0
            if noisy:
                noise += 0.18 * math.sin(2 * math.pi * 70 * t)
                noise += 0.08 * math.sin(2 * math.pi * 1200 * t)
                if int(t * 5) % 9 == 0:
                    noise += 0.20 * math.sin(2 * math.pi * 2100 * t)
            value = max(-1.0, min(1.0, speech + noise))
            frames.extend(int(value * 28000).to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(frames))
    return path


def ensure_demo_audios(case_ids: list[str]) -> None:
    for case_id in case_ids:
        generate_demo_audio(case_id, noisy=True)
        generate_demo_audio(case_id, noisy=False)


def audio_url(path: Path) -> str:
    if "uploads" in path.parts:
        return f"/static/uploads/{path.name}"
    return f"/static/audio/{path.name}"


def resolve_static_url(url: str) -> Path | None:
    if url.startswith("/static/audio/"):
        return STATIC_AUDIO_DIR / Path(url).name
    if url.startswith("/static/uploads/"):
        return UPLOAD_DIR / Path(url).name
    return None


def get_audio_duration_seconds(path: Path) -> float | None:
    """Read duration from metadata without loading the full waveform."""
    try:
        with wave.open(str(path), "rb") as wav:
            frame_rate = wav.getframerate()
            if frame_rate <= 0:
                return None
            return wav.getnframes() / frame_rate
    except (wave.Error, OSError, EOFError):
        pass

    try:
        soundfile = __import__("soundfile")
        info = soundfile.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / info.samplerate
    except Exception:
        pass

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
