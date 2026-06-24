import React from "react";
import { Activity, AudioLines, FileText, ListChecks, Settings2 } from "lucide-react";

const GROUPS = [
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
      key.includes("峰值"),
  },
  {
    title: "语音分离",
    icon: Settings2,
    match: (key) => key.includes("分离"),
  },
  {
    title: "摘要生成",
    icon: ListChecks,
    match: (key) => key.includes("摘要"),
  },
];

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
  const fullValue = formatValue(value);
  const compactValue = compactMetricValue(fullValue);
  const isLong = fullValue.length > 90;

  return (
    <p className={`metric-row${isLong ? " metric-row-long" : ""}`}>
      <span>{label}</span>
      <strong title={fullValue}>{compactValue}</strong>
    </p>
  );
}

function formatValue(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function compactMetricValue(value) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= 160) return normalized;
  return `${normalized.slice(0, 157)}...`;
}
