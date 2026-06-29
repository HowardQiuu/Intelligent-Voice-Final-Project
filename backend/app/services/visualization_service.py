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
    matched = _level_matched_comparison(original, enhanced)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_DIR / f"enhancement_visual_{label}_{uuid.uuid4().hex[:8]}.svg"
    output_path.write_text(_render_svg(original, enhanced, matched), encoding="utf-8")

    return audio_url(output_path), {
        "增强可视化": "已生成波形/噪声底/清晰度对比图",
        "增强诊断结论": matched["verdict"],
        "原始噪声底估计": f"{original['noise_floor']:.3f}",
        "增强后噪声底估计": f"{enhanced['noise_floor']:.3f}",
        "噪声底变化": _ratio_change(original["noise_floor"], enhanced["noise_floor"]),
        "响度归一噪声底变化": matched["matched_noise_change"],
        "清晰度代理变化": f"{original['clarity_db']:.1f} dB -> {enhanced['clarity_db']:.1f} dB",
        "清晰度代理差值": f"{matched['clarity_delta_db']:+.1f} dB",
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


def _render_svg(original: dict, enhanced: dict, matched: dict) -> str:
    width = 1200
    height = 690
    top_a = 352
    top_b = 532
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
    verdict_color = "#1e9a63" if matched["verdict_level"] == "good" else "#c78018" if matched["verdict_level"] == "mixed" else "#c7443e"
    audibility_color = "#1e9a63" if matched["energy_change_percent"] >= 20 else "#60758c"
    denoise_color = "#1e9a63" if _ratio_change_value(original["noise_floor"], matched["matched_enhanced_noise_floor"]) <= -5 else "#c7443e"
    clarity_color = "#1e9a63" if clarity_delta >= 0.5 else "#c7443e" if clarity_delta < -0.5 else "#60758c"
    verdict_lines = _svg_text_lines(str(matched["verdict"]), max_chars=34, max_lines=2)
    evidence_lines = _svg_text_lines(
        f"原始噪声底 {original['noise_floor']:.3f} -> 增强后 {enhanced['noise_floor']:.3f} ({noise_change})，这是响度抬升后的原始观测值，不单独代表降噪成功。",
        max_chars=58,
        max_lines=2,
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" rx="8" fill="#f7fbff"/>
  <text x="36" y="48" font-family="Microsoft YaHei, Arial" font-size="26" font-weight="700" fill="#172033">语音增强可视化对比</text>
  <text x="36" y="82" font-family="Microsoft YaHei, Arial" font-size="15" fill="#60758c">先看综合诊断；噪声底和清晰度是降噪证据，单纯放大音量会让原始噪声底数值同步变大。</text>

  <rect x="36" y="108" width="1128" height="168" rx="8" fill="#ffffff" stroke="#dfeaf7"/>
  <text x="60" y="138" font-family="Microsoft YaHei, Arial" font-size="15" font-weight="700" fill="#172033">综合诊断</text>
  {_svg_tspan_block(verdict_lines, x=60, y=170, line_height=24, font_size=20, fill=verdict_color, weight=700)}
  {_svg_tspan_block(evidence_lines, x=60, y=222, line_height=18, font_size=13, fill="#60758c")}

  <rect x="60" y="238" width="316" height="22" rx="4" fill="#fff7e8" stroke="#f0dfbf"/>
  <text x="76" y="254" font-family="Microsoft YaHei, Arial" font-size="12" fill="#9a6400">主结论优先；下方三项仅作为证据拆解</text>

  <rect x="420" y="144" width="210" height="88" rx="6" fill="#f9fcff" stroke="#e1ecf8"/>
  <text x="440" y="170" font-family="Microsoft YaHei, Arial" font-size="13" font-weight="700" fill="#172033">可听度 / 响度</text>
  <text x="440" y="200" font-family="Microsoft YaHei, Arial" font-size="22" font-weight="700" fill="{audibility_color}">{energy_change}</text>
  <text x="440" y="220" font-family="Microsoft YaHei, Arial" font-size="12" fill="#60758c">平均能量变化</text>

  <rect x="660" y="144" width="210" height="88" rx="6" fill="#f9fcff" stroke="#e1ecf8"/>
  <text x="680" y="170" font-family="Microsoft YaHei, Arial" font-size="13" font-weight="700" fill="#172033">降噪质量</text>
  <text x="680" y="200" font-family="Microsoft YaHei, Arial" font-size="22" font-weight="700" fill="{denoise_color}">{matched['matched_noise_change']}</text>
  <text x="680" y="220" font-family="Microsoft YaHei, Arial" font-size="12" fill="#60758c">响度归一噪声底</text>

  <rect x="900" y="144" width="210" height="88" rx="6" fill="#f9fcff" stroke="#e1ecf8"/>
  <text x="920" y="170" font-family="Microsoft YaHei, Arial" font-size="13" font-weight="700" fill="#172033">清晰度代理</text>
  <text x="920" y="200" font-family="Microsoft YaHei, Arial" font-size="22" font-weight="700" fill="{clarity_color}">{clarity_delta:+.1f} dB</text>
  <text x="920" y="220" font-family="Microsoft YaHei, Arial" font-size="12" fill="#60758c">{original['clarity_db']:.1f} dB -> {enhanced['clarity_db']:.1f} dB</text>

  <text x="36" y="308" font-family="Microsoft YaHei, Arial" font-size="17" font-weight="700" fill="#c7443e">原始音频包络</text>
  <line x1="{left}" y1="{top_a}" x2="{left + chart_w}" y2="{top_a}" stroke="#d5e4f4"/>
  <line x1="{left}" y1="{noise_a_y:.1f}" x2="{left + chart_w}" y2="{noise_a_y:.1f}" stroke="#f0a49d" stroke-width="2" stroke-dasharray="7 7"/>
  <polyline points="{html.escape(data_a)}" fill="none" stroke="#d94f45" stroke-width="2.5"/>

  <text x="36" y="488" font-family="Microsoft YaHei, Arial" font-size="17" font-weight="700" fill="#1e9a63">增强后音频包络</text>
  <line x1="{left}" y1="{top_b}" x2="{left + chart_w}" y2="{top_b}" stroke="#d5e4f4"/>
  <line x1="{left}" y1="{noise_b_y:.1f}" x2="{left + chart_w}" y2="{noise_b_y:.1f}" stroke="#78c8a7" stroke-width="2" stroke-dasharray="7 7"/>
  <polyline points="{html.escape(data_b)}" fill="none" stroke="#1e9a63" stroke-width="2.5"/>

  <text x="{left}" y="614" font-family="Microsoft YaHei, Arial" font-size="14" fill="#60758c">虚线表示低能量段噪声底估计；判断降噪时优先看响度归一噪声底和清晰度差值。</text>
  <rect x="36" y="634" width="1128" height="34" rx="8" fill="#ffffff" stroke="#dfeaf7"/>
  <text x="60" y="656" font-family="Microsoft YaHei, Arial" font-size="13" fill="#172033">平均能量: {original['avg_rms']:.3f} -> {enhanced['avg_rms']:.3f} ({energy_change})</text>
  <text x="390" y="656" font-family="Microsoft YaHei, Arial" font-size="13" fill="#172033">峰值: {original['peak']:.3f} -> {enhanced['peak']:.3f}</text>
  <text x="650" y="656" font-family="Microsoft YaHei, Arial" font-size="13" fill="#172033">低能量段比例: {original['quiet_ratio']:.0%} -> {enhanced['quiet_ratio']:.0%}</text>
</svg>"""


def _svg_text_lines(text: str, *, max_chars: int, max_lines: int) -> list[str]:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return [""]
    lines: list[str] = []
    remaining = cleaned
    while remaining and len(lines) < max_lines:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            break
        split_at = max(
            remaining.rfind("，", 0, max_chars + 1),
            remaining.rfind("；", 0, max_chars + 1),
            remaining.rfind("、", 0, max_chars + 1),
            remaining.rfind(" ", 0, max_chars + 1),
        )
        if split_at <= 0:
            split_at = max_chars
        lines.append(remaining[:split_at].rstrip("，；、 "))
        remaining = remaining[split_at:].lstrip("，；、 ")
    if remaining and lines:
        lines[-1] = f"{lines[-1].rstrip('。')}..."
    return lines


def _svg_tspan_block(
    lines: list[str],
    *,
    x: int,
    y: int,
    line_height: int,
    font_size: int,
    fill: str,
    weight: int | None = None,
) -> str:
    weight_attr = f' font-weight="{weight}"' if weight is not None else ""
    tspans = []
    for index, line in enumerate(lines):
        tspans.append(f'<tspan x="{x}" dy="{0 if index == 0 else line_height}">{html.escape(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Arial" font-size="{font_size}"'
        f'{weight_attr} fill="{fill}">{"".join(tspans)}</text>'
    )


def _level_matched_comparison(original: dict, enhanced: dict) -> dict:
    scale = original["avg_rms"] / max(enhanced["avg_rms"], 1e-9)
    matched_noise = enhanced["noise_floor"] * scale
    matched_speech = enhanced["speech_rms"] * scale
    matched_noise_change_value = _ratio_change_value(original["noise_floor"], matched_noise)
    clarity_delta = enhanced["clarity_db"] - original["clarity_db"]
    energy_change = _ratio_change_value(original["avg_rms"], enhanced["avg_rms"])

    if clarity_delta >= 0.5 and matched_noise_change_value <= 10:
        verdict = "增强有效：响度提升后，清晰度/噪声底仍保持改善"
        verdict_level = "good"
    elif energy_change > 20 and (clarity_delta < -0.5 or matched_noise_change_value > 15):
        verdict = "可听度提升，但降噪/清晰度未改善，主要收益来自音量抬升"
        verdict_level = "mixed"
    elif clarity_delta < -0.5 or matched_noise_change_value > 15:
        verdict = "增强质量变差：噪声底或清晰度代理恶化"
        verdict_level = "bad"
    else:
        verdict = "变化不明显：增强效果接近持平"
        verdict_level = "mixed"

    return {
        "matched_enhanced_noise_floor": matched_noise,
        "matched_enhanced_speech_rms": matched_speech,
        "matched_noise_change": _format_ratio_change(matched_noise_change_value),
        "clarity_delta_db": clarity_delta,
        "energy_change_percent": energy_change,
        "verdict": verdict,
        "verdict_level": verdict_level,
    }


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
    return _format_ratio_change(_ratio_change_value(before, after))


def _ratio_change_value(before: float, after: float) -> float:
    if before <= 1e-9:
        return 0.0
    return ((after - before) / before) * 100


def _format_ratio_change(value: float) -> str:
    return f"{value:+.1f}%"


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
