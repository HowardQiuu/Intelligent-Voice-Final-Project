from __future__ import annotations

import html
import math
import uuid
import wave
from pathlib import Path

from .audio_service import UPLOAD_DIR, audio_url


def generate_enhancement_visual(original_path: Path | None, enhanced_path: Path | None, label: str) -> tuple[str | None, dict[str, str]]:
    if original_path is None or enhanced_path is None or not original_path.exists() or not enhanced_path.exists():
        return None, {"增强可视化": "音频文件不可读"}

    original = _read_envelope(original_path)
    enhanced = _read_envelope(enhanced_path)
    if original is None or enhanced is None:
        return None, {"增强可视化": "暂仅支持 WAV 可视化"}

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_DIR / f"enhancement_visual_{label}_{uuid.uuid4().hex[:8]}.svg"
    output_path.write_text(_render_svg(original, enhanced), encoding="utf-8")

    return audio_url(output_path), {
        "增强可视化": "已生成波形/能量对比图",
        "原始平均能量": f"{original['avg_rms']:.3f}",
        "增强后平均能量": f"{enhanced['avg_rms']:.3f}",
        "平均能量变化": f"{_ratio_change(original['avg_rms'], enhanced['avg_rms'])}",
        "原始峰值": f"{original['peak']:.3f}",
        "增强后峰值": f"{enhanced['peak']:.3f}",
    }


def _read_envelope(path: Path, bins: int = 180, max_samples_per_bin: int = 4000) -> dict | None:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            total_frames = wav.getnframes()
            if sample_width not in {1, 2, 4} or channels <= 0 or frame_rate <= 0 or total_frames <= 0:
                return None

            frames_per_bin = max(1, math.ceil(total_frames / bins))
            envelope = []
            peak = 0.0
            weighted_rms = 0.0
            frames_seen = 0
            max_abs = float(2 ** (sample_width * 8 - 1))

            while wav.tell() < total_frames and len(envelope) < bins:
                remaining = total_frames - wav.tell()
                current_frames = min(frames_per_bin, remaining)
                raw = wav.readframes(current_frames)
                if not raw:
                    break
                rms, local_peak = _sample_rms(raw, channels, sample_width, max_abs, max_samples_per_bin)
                envelope.append(rms)
                peak = max(peak, local_peak)
                weighted_rms += rms * current_frames
                frames_seen += current_frames

            avg_rms = weighted_rms / frames_seen if frames_seen else 0.0
            duration = total_frames / frame_rate
            return {"envelope": envelope, "avg_rms": avg_rms, "peak": peak, "duration": duration}
    except (wave.Error, OSError, EOFError):
        return _read_soundfile_envelope(path, bins=bins)


def _read_soundfile_envelope(path: Path, bins: int = 180) -> dict | None:
    try:
        soundfile = __import__("soundfile")
        info = soundfile.info(str(path))
        if info.samplerate <= 0 or info.frames <= 0:
            return None
        frames_per_bin = max(1, math.ceil(info.frames / bins))
        envelope = []
        peak = 0.0
        weighted_rms = 0.0
        frames_seen = 0
        for block in soundfile.blocks(str(path), blocksize=frames_per_bin, always_2d=True, dtype="float32"):
            if len(envelope) >= bins:
                break
            channel = block[:, 0]
            if channel.size == 0:
                continue
            rms = float(math.sqrt(float((channel * channel).mean())))
            local_peak = float(abs(channel).max())
            envelope.append(rms)
            peak = max(peak, local_peak)
            weighted_rms += rms * len(channel)
            frames_seen += len(channel)
        avg_rms = weighted_rms / frames_seen if frames_seen else 0.0
        return {"envelope": envelope, "avg_rms": avg_rms, "peak": peak, "duration": info.frames / info.samplerate}
    except Exception:
        return None


def _sample_rms(raw: bytes, channels: int, sample_width: int, max_abs: float, max_samples: int) -> tuple[float, float]:
    frame_width = channels * sample_width
    frame_count = len(raw) // frame_width
    if frame_count <= 0:
        return 0.0, 0.0

    stride = max(1, frame_count // max_samples)
    total = 0.0
    peak = 0.0
    count = 0
    for frame_index in range(0, frame_count, stride):
        offset = frame_index * frame_width
        value = int.from_bytes(raw[offset : offset + sample_width], "little", signed=sample_width != 1)
        if sample_width == 1:
            value -= 128
        normalized = min(1.0, abs(value) / max_abs)
        total += normalized * normalized
        peak = max(peak, normalized)
        count += 1

    rms = math.sqrt(total / count) if count else 0.0
    return rms, peak


def _render_svg(original: dict, enhanced: dict) -> str:
    width = 960
    height = 420
    top_a = 118
    top_b = 274
    chart_w = 820
    left = 92
    data_a = _polyline(original["envelope"], left, top_a, chart_w, 52)
    data_b = _polyline(enhanced["envelope"], left, top_b, chart_w, 52)
    change = _ratio_change(original["avg_rms"], enhanced["avg_rms"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="960" height="420" rx="8" fill="#f7fbff"/>
  <text x="32" y="42" font-family="Microsoft YaHei, Arial" font-size="24" font-weight="700" fill="#172033">语音增强可视化对比</text>
  <text x="32" y="72" font-family="Microsoft YaHei, Arial" font-size="14" fill="#60758c">上方为原始音频，下方为增强后音频；曲线越高表示局部能量越强。</text>
  <text x="32" y="122" font-family="Microsoft YaHei, Arial" font-size="16" font-weight="700" fill="#1559c7">原始</text>
  <line x1="{left}" y1="{top_a}" x2="{left + chart_w}" y2="{top_a}" stroke="#d5e4f4"/>
  <polyline points="{html.escape(data_a)}" fill="none" stroke="#d94f45" stroke-width="2.5"/>
  <text x="32" y="278" font-family="Microsoft YaHei, Arial" font-size="16" font-weight="700" fill="#1e9a63">增强后</text>
  <line x1="{left}" y1="{top_b}" x2="{left + chart_w}" y2="{top_b}" stroke="#d5e4f4"/>
  <polyline points="{html.escape(data_b)}" fill="none" stroke="#1e9a63" stroke-width="2.5"/>
  <rect x="32" y="338" width="896" height="54" rx="8" fill="#ffffff" stroke="#dfeaf7"/>
  <text x="56" y="370" font-family="Microsoft YaHei, Arial" font-size="14" fill="#172033">原始平均能量: {original['avg_rms']:.3f}</text>
  <text x="252" y="370" font-family="Microsoft YaHei, Arial" font-size="14" fill="#172033">增强后平均能量: {enhanced['avg_rms']:.3f}</text>
  <text x="480" y="370" font-family="Microsoft YaHei, Arial" font-size="14" fill="#172033">平均能量变化: {change}</text>
  <text x="690" y="370" font-family="Microsoft YaHei, Arial" font-size="14" fill="#172033">峰值: {original['peak']:.3f} -> {enhanced['peak']:.3f}</text>
</svg>"""


def _polyline(values: list[float], left: int, baseline: int, width: int, scale: int) -> str:
    if not values:
        return ""
    count = max(1, len(values) - 1)
    points = []
    for index, value in enumerate(values):
        x = left + (width * index / count)
        y = baseline - min(1.0, value * 3.0) * scale
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _ratio_change(before: float, after: float) -> str:
    if before <= 1e-9:
        return "0.0%"
    return f"{((after - before) / before) * 100:+.1f}%"
