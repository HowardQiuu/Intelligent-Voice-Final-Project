import React from "react";
import { Activity, AudioLines, FileText, ListChecks, Settings2 } from "lucide-react";

const GROUPS = [
  {
    title: "自适应策略",
    icon: Settings2,
    match: (key) =>
      key.includes("主处理后端") ||
      key.includes("中文ASR模型") ||
      key.includes("自适应路径") ||
      key.includes("会议提取质量评分") ||
      key.includes("语音覆盖率") ||
      key.includes("疑似重叠比例") ||
      key.includes("静音比例") ||
      key.includes("quality_router"),
  },
  {
    title: "说话人画像",
    icon: AudioLines,
    match: (key) =>
      key.includes("检测说话人数") ||
      key.includes("说话人分段模型") ||
      key.includes("说话人会议画像") ||
      key.includes("按说话人摘要") ||
      key.includes("SenseVoice事件标签"),
  },
  {
    title: "ASR 转写",
    icon: FileText,
    match: (key) => key.startsWith("ASR") || key.includes("转写主题"),
  },
  {
    title: "语音增强",
    icon: AudioLines,
    match: (key) =>
      key.includes("增强") ||
      key.includes("噪声") ||
      key.includes("清晰度") ||
      key.includes("能量") ||
      key.includes("峰值") ||
      key.includes("pregain") ||
      key.includes("enhancement_selected"),
  },
  {
    title: "语音分离",
    icon: Settings2,
    match: (key) =>
      key.includes("分离") ||
      key.includes("separation_candidates") ||
      key.includes("selected_separation"),
  },
  {
    title: "摘要生成",
    icon: ListChecks,
    match: (key) => key.includes("摘要"),
  },
];

const METRIC_LABELS = {
  runtime_normalize_seconds: "后端音频归一化耗时",
  runtime_chunk_plan_seconds: "后端分块规划耗时",
  runtime_enhancement_seconds: "后端语音增强耗时",
  runtime_visual_seconds: "后端可视化生成耗时",
  runtime_separation_seconds: "后端语音分离耗时",
  runtime_asr_seconds: "后端 ASR 转写耗时",
  runtime_summary_seconds: "后端摘要生成耗时",
  runtime_topic_seconds: "后端主题分类耗时",
  runtime_total_seconds: "后端处理总耗时",
  quality_router_status: "Quality Router",
  quality_router_enhancement_candidates: "增强候选评分",
  quality_router_selected_enhancement: "选中增强候选",
  quality_router_selected_score: "选中增强评分",
  quality_router_separation_candidates: "分离候选评分",
  quality_router_selected_separation: "选中分离候选",
  quality_router_selected_separation_score: "选中分离评分",
  quality_pregain_status: "预增益状态",
  quality_pregain_gain_db: "预增益增益",
  pregain_input_rms_dbfs: "预增益前 RMS(dBFS)",
  pregain_output_rms_dbfs: "预增益后 RMS(dBFS)",
  pregain_input_peak_dbfs: "预增益前 Peak(dBFS)",
  pregain_output_peak_dbfs: "预增益后 Peak(dBFS)",
  pregain_input_silent_ratio: "预增益前静音比例",
  pregain_output_silent_ratio: "预增益后静音比例",
  pregain_input_clipping_ratio: "预增益前削波比例",
  pregain_output_clipping_ratio: "预增益后削波比例",
  enhancement_selected_rms_dbfs: "最终增强 RMS(dBFS)",
  enhancement_selected_peak_dbfs: "最终增强 Peak(dBFS)",
  enhancement_selected_silent_ratio: "最终增强静音比例",
  enhancement_selected_clipping_ratio: "最终增强削波比例",
  enhancement_selected_spectral_centroid_hz: "最终增强频谱质心",
};

export function ProcessingDiagnostics({ metrics = {} }) {
  const entries = Object.entries(metrics || {});
  if (entries.length === 0) return null;

  const used = new Set();
  const groups = GROUPS.map((group) => {
    const items = entries.filter(([key]) => group.match(key));
    items.forEach(([key]) => used.add(key));
    return { ...group, items };
  }).filter((group) => group.items.length > 0);

  const otherItems = entries.filter(([key]) => !used.has(key));
  if (otherItems.length > 0) {
    groups.push({
      title: "其他指标",
      icon: Activity,
      items: otherItems,
    });
  }

  return (
    <section className="panel diagnostics-panel">
      <div className="panel-title">
        <Activity size={20} />
        <h2>处理诊断</h2>
      </div>
      <div className="diagnostics-grid">
        {groups.map((group) => {
          const Icon = group.icon;
          return (
            <article className="diagnostic-card" key={group.title}>
              <h3>
                <Icon size={17} />
                {group.title}
              </h3>
              <div className="diagnostic-items">
                {group.items.map(([key, value]) => (
                  <MetricRow key={key} label={key} value={value} />
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function MetricRow({ label, value }) {
  const displayLabel = formatMetricLabel(label);
  const fullValue = formatValue(value, label);
  const compactValue = compactMetricValue(fullValue);
  const isLong = fullValue.length > 90;

  return (
    <p className={`metric-row${isLong ? " metric-row-long" : ""}`}>
      <span title={label}>{displayLabel}</span>
      <strong title={fullValue}>{compactValue}</strong>
    </p>
  );
}

function formatMetricLabel(label) {
  return METRIC_LABELS[label] || label;
}

function formatValue(value, label) {
  if (value === null || value === undefined) return "-";
  if (label?.startsWith("runtime_")) return `${value} 秒`;
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function compactMetricValue(value) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= 160) return normalized;
  return `${normalized.slice(0, 157)}...`;
}
