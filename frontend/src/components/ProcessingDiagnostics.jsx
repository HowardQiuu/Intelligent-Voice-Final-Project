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
      (key.includes("quality_router") && !key.includes("separation")) ||
      key === "processing_mode" ||
      key === "fast_path_mode" ||
      key === "separation_input_source",
  },
  {
    title: "分块与运行",
    icon: Activity,
    match: (key) =>
      key.startsWith("runtime_") ||
      key === "chunk_processing" ||
      key === "chunk_count" ||
      (key.startsWith("chunk_") && !key.startsWith("chunk_track_alignment")),
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
    title: "说话人估计",
    icon: AudioLines,
    match: (key) => key.includes("speaker_count") || key === "estimated_speaker_count",
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
      key.includes("separation") ||
      key.includes("recursive_blind") ||
      key.includes("stft_mask") ||
      key.includes("low_overlap") ||
      key.includes("speechbrain_residual") ||
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
  chunk_processing: "分块处理",
  chunk_count: "分块数量",
  separation_input_source: "分离输入源",
  fast_path_mode: "快速路径",
  processing_mode: "处理模式",
  mixture_consistency_projection: "混合一致性投影",
  recursive_blind_expansion: "递归盲扩展",
  recursive_blind_expansion_mode: "扩展模式",
  recursive_blind_expansion_target_tracks: "目标轨道",
  recursive_blind_expansion_auto_max_tracks: "最大轨道",
  recursive_blind_expansion_auto_max_depth: "最大深度",
  recursive_blind_expansion_steps: "扩展步数",
  recursive_blind_expansion_auto_decisions: "自动决策",
  stft_mask_refinement: "STFT 掩码细化",
  stft_mask_n_fft: "STFT FFT 点数",
  stft_mask_hop: "STFT hop",
  stft_mask_power: "STFT 幂次",
  low_overlap_leakage_suppression: "低重叠泄漏抑制",
  low_overlap_leakage_overlap_ratio: "重叠比例",
  low_overlap_leakage_dominance_db: "主导阈值",
  low_overlap_leakage_loser_gain: "弱轨增益",
  speechbrain_residual_projection: "残差投影",
  speechbrain_residual_projection_amount: "残差投影量",
  speaker_count_estimation_status: "估计状态",
  estimated_speaker_count: "估计说话人数",
  speaker_count_embedding_backend: "嵌入后端",
  speaker_count_embedding_backend_status: "嵌入状态",
  speaker_count_estimation_method: "估计方法",
  speaker_count_cluster_stability: "聚类稳定度",
  speaker_count_estimation_stability: "估计稳定度",
  speaker_count_min_track_quality: "最低轨道质量",
  speaker_count_estimation_min_track_quality: "估计最低质量",
  speaker_count_estimation_accepted_tracks: "采纳轨道数",
  speaker_count_estimation_candidate_tracks: "候选轨道数",
  speaker_count_global_cluster_summary: "全局聚类摘要",
  speaker_count_cluster_summary: "聚类摘要",
  speaker_count_global_counting_mode: "全局计数模式",
  speaker_count_raw_global_estimated_speaker_count: "原始全局人数",
  speaker_count_window_estimated_speaker_count: "窗口估计人数",
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
  const entries = Object.entries(metrics || {}).filter(([key]) => !shouldHideMetric(key));
  if (entries.length === 0) return null;

  const groups = buildDiagnosticGroups(entries);

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
            <article
              className={`diagnostic-card${group.wide ? " diagnostic-card-wide" : ""}`}
              key={group.title}
            >
              <h3>
                <span className="diagnostic-heading-text">
                  <Icon size={17} />
                  <span>{group.title}</span>
                </span>
                <span className="diagnostic-count">{group.items.length} 项</span>
              </h3>
              <div className={`diagnostic-items${group.wide ? " diagnostic-items-balanced" : ""}`}>
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

function shouldHideMetric(key) {
  return key.startsWith("chunk_track_alignment");
}

function buildDiagnosticGroups(entries) {
  const buckets = GROUPS.map((group) => ({ ...group, items: [] }));
  const otherItems = [];

  entries.forEach((entry) => {
    const [key] = entry;
    const group = buckets.find((bucket) => bucket.match(key));
    if (group) {
      group.items.push(entry);
    } else {
      otherItems.push(entry);
    }
  });

  const groups = buckets.filter((group) => group.items.length > 0);
  if (otherItems.length > 0) {
    groups.push({
      title: "其他指标",
      icon: Activity,
      items: otherItems,
    });
  }

  return groups.map((group) => ({
    ...group,
    wide: group.items.length > 10 || group.title === "其他指标",
  }));
}

function MetricRow({ label, value }) {
  const displayLabel = formatMetricLabel(label);
  const fullValue = formatValue(value, label);
  const compactValue = compactMetricValue(fullValue);
  const isLong = fullValue.length > 52 || displayLabel.length > 18 || label.length > 28;

  return (
    <p className={`metric-row${isLong ? " metric-row-long" : ""}`}>
      <span className="metric-label" title={label}>{displayLabel}</span>
      <strong className="metric-value" title={fullValue}>{compactValue}</strong>
    </p>
  );
}

function formatMetricLabel(label) {
  if (METRIC_LABELS[label]) return METRIC_LABELS[label];

  const speakerTrackMatch = label.match(/^speaker_count_track_(\d+)_(.+)$/);
  if (speakerTrackMatch) {
    const [, trackIndex, field] = speakerTrackMatch;
    const fieldLabel =
      {
        quality: "质量",
        global_speaker: "全局说话人",
        decision: "判定",
      }[field] || field.replace(/_/g, " ");
    return `轨道 ${Number(trackIndex) + 1} ${fieldLabel}`;
  }

  return label.replace(/_/g, " ");
}

function formatValue(value, label) {
  if (value === null || value === undefined) return "-";
  if (label?.startsWith("runtime_")) return `${value} 秒`;
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function compactMetricValue(value) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= 220) return normalized;
  return `${normalized.slice(0, 217)}...`;
}
