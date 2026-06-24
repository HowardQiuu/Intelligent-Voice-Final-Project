from __future__ import annotations

import html
import math
import uuid
import wave
from pathlib import Path

from .audio_service import UPLOAD_DIR, audio_url


def generate_enhancement_visual(
    original_path: Path | None,
    enhanced_path: Path | None,
    label: str,
) -> tuple[str | None, dict[str, str]]:
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
        "增强可视化": "已生成波形/噪声底/清晰度对比图",
        "原始噪声底估计": f"{original['noise_floor']:.3f}",
        "增强后噪声底估计": f"{enhanced['noise_floor']:.3f}",
        "噪声底变化": _ratio_change(original["noise_floor"], enhanced["noise_floor"]),
        "清晰度代理变化": f"{original['clarity_db']:.1f} dB -> {enhanced['clarity_db']:.1f} dB",
        "平均能量变化(辅助)": _ratio_change(original["avg_rms"], enhanced["avg_rms"]),
        "峰值变化": f"{original['peak']:.3f} -> {enhanced['peak']:.3f}",
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
            return _add_enhancement_metrics(
                {"envelope": envelope, "avg_rms": avg_rms, "peak": peak, "duration": duration}
            )
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
        return _add_enhancement_metrics(
            {"envelope": envelope, "avg_rms": avg_rms, "peak": peak, "duration": info.frames / info.samplerate}
        )
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
    width = 1200
    height = 620
    top_a = 286
    top_b = 466
    chart_w = 1020
    left = 112
    ref_peak = max(original["peak"], enhanced["peak"], 1e-6)
    data_a = _polyline(original["envelope"], left, top_a, chart_w, 72, ref_peak)
    data_b = _polyline(enhanced["envelope"], left, top_b, chart_w, 72, ref_peak)
    noise_a_y = _metric_y(original["noise_floor"], top_a, 72, ref_peak)
    noise_b_y = _metric_y(enhanced["noise_floor"], top_b, 72, ref_peak)
    noise_change = _ratio_change(original["noise_floor"], enhanced["noise_floor"])
    energy_change = _ratio_change(original["avg_rms"], enhanced["avg_rms"])
    clarity_delta = enhanced["clarity_db"] - original["clarity_db"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" rx="8" fill="#f7fbff"/>
  <text x="36" y="48" font-family="Microsoft YaHei, Arial" font-size="26" font-weight="700" fill="#172033">语音增强可视化对比</text>
  <text x="36" y="82" font-family="Microsoft YaHei, Arial" font-size="15" fill="#60758c">增强不是简单放大音量；降噪成功时整体能量可能下降，重点看低能量噪声底是否降低，以及语音/噪声差距是否拉开。</text>

  <rect x="36" y="108" width="1128" height="98" rx="8" fill="#ffffff" stroke="#dfeaf7"/>
  <text x="60" y="140" font-family="Microsoft YaHei, Arial" font-size="15" font-weight="700" fill="#172033">噪声底估计</text>
  <text x="60" y="172" font-family="Microsoft YaHei, Arial" font-size="24" font-weight="700" fill="#1e9a63">{noise_change}</text>
  <text x="212" y="172" font-family="Microsoft YaHei, Arial" font-size="14" fill="#60758c">原始 {original['noise_floor']:.3f} -> 增强后 {enhanced['noise_floor']:.3f}</text>
  <text x="540" y="140" font-family="Microsoft YaHei, Arial" font-size="15" font-weight="700" fill="#172033">清晰度代理</text>
  <text x="540" y="172" font-family="Microsoft YaHei, Arial" font-size="24" font-weight="700" fill="#1559c7">{clarity_delta:+.1f} dB</text>
  <text x="700" y="172" font-family="Microsoft YaHei, Arial" font-size="14" fill="#60758c">语音强段 / 噪声底差距：{original['clarity_db']:.1f} dB -> {enhanced['clarity_db']:.1f} dB</text>

  <text x="36" y="242" font-family="Microsoft YaHei, Arial" font-size="17" font-weight="700" fill="#c7443e">原始音频包络</text>
  <line x1="{left}" y1="{top_a}" x2="{left + chart_w}" y2="{top_a}" stroke="#d5e4f4"/>
  <line x1="{left}" y1="{noise_a_y:.1f}" x2="{left + chart_w}" y2="{noise_a_y:.1f}" stroke="#f0a49d" stroke-width="2" stroke-dasharray="7 7"/>
  <polyline points="{html.escape(data_a)}" fill="none" stroke="#d94f45" stroke-width="2.5"/>

  <text x="36" y="422" font-family="Microsoft YaHei, Arial" font-size="17" font-weight="700" fill="#1e9a63">增强后音频包络</text>
  <line x1="{left}" y1="{top_b}" x2="{left + chart_w}" y2="{top_b}" stroke="#d5e4f4"/>
  <line x1="{left}" y1="{noise_b_y:.1f}" x2="{left + chart_w}" y2="{noise_b_y:.1f}" stroke="#78c8a7" stroke-width="2" stroke-dasharray="7 7"/>
  <polyline points="{html.escape(data_b)}" fill="none" stroke="#1e9a63" stroke-width="2.5"/>

  <text x="{left}" y="548" font-family="Microsoft YaHei, Arial" font-size="14" fill="#60758c">虚线表示低能量段噪声底估计；平均能量仅作为辅助，下降通常说明噪声被压低。</text>
  <rect x="36" y="568" width="1128" height="34" rx="8" fill="#ffffff" stroke="#dfeaf7"/>
  <text x="60" y="590" font-family="Microsoft YaHei, Arial" font-size="13" fill="#172033">平均能量: {original['avg_rms']:.3f} -> {enhanced['avg_rms']:.3f} ({energy_change})</text>
  <text x="390" y="590" font-family="Microsoft YaHei, Arial" font-size="13" fill="#172033">峰值: {original['peak']:.3f} -> {enhanced['peak']:.3f}</text>
  <text x="650" y="590" font-family="Microsoft YaHei, Arial" font-size="13" fill="#172033">低能量段比例: {original['quiet_ratio']:.0%} -> {enhanced['quiet_ratio']:.0%}</text>
</svg>"""


def _polyline(values: list[float], left: int, baseline: int, width: int, scale: int, ref_peak: float) -> str:
    if not values:
        return ""
    count = max(1, len(values) - 1)
    points = []
    for index, value in enumerate(values):
        x = left + (width * index / count)
        y = _metric_y(value, baseline, scale, ref_peak)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _metric_y(value: float, baseline: int, scale: int, ref_peak: float) -> float:
    return baseline - min(1.0, value / max(ref_peak, 1e-6)) * scale


def _ratio_change(before: float, after: float) -> str:
    if before <= 1e-9:
        return "0.0%"
    return f"{((after - before) / before) * 100:+.1f}%"


def _add_enhancement_metrics(data: dict) -> dict:
    envelope = data["envelope"]
    if not envelope:
        data.update({"noise_floor": 0.0, "speech_rms": 0.0, "clarity_db": 0.0, "quiet_ratio": 0.0})
        return data

    sorted_values = sorted(envelope)
    low_count = max(1, int(len(sorted_values) * 0.2))
    high_count = max(1, int(len(sorted_values) * 0.3))
    low_values = sorted_values[:low_count]
    high_values = sorted_values[-high_count:]
    noise_floor = sum(low_values) / len(low_values)
    speech_rms = sum(high_values) / len(high_values)
    clarity_db = 20.0 * math.log10((speech_rms + 1e-6) / (noise_floor + 1e-6))
    quiet_threshold = noise_floor * 1.2 + 1e-6
    quiet_ratio = sum(1 for value in envelope if value <= quiet_threshold) / len(envelope)
    data.update(
        {
            "noise_floor": noise_floor,
            "speech_rms": speech_rms,
            "clarity_db": clarity_db,
            "quiet_ratio": quiet_ratio,
        }
    )
    return data
