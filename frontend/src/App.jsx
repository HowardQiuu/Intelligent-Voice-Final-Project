import React, { useEffect, useMemo, useRef, useState } from "react";
import { AudioWaveform, Loader2, Play, Upload } from "lucide-react";
import {
  fetchDemoCases,
  processDemo,
  uploadAudioFile,
} from "./api";
import { AudioCompare } from "./components/AudioCompare";
import { EmptyState } from "./components/EmptyState";
import { Pipeline } from "./components/Pipeline";
import { ProcessingDiagnostics } from "./components/ProcessingDiagnostics";
import { Summary } from "./components/Summary";
import { Transcript } from "./components/Transcript";

export function App() {
  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [processingMode, setProcessingMode] = useState("fast");
  const [uploadProgress, setUploadProgress] = useState(null);
  const uploadRef = useRef(null);

  useEffect(() => {
    fetchDemoCases()
      .then((data) => {
        setCases(data);
        setSelectedCase(data[1]?.id || data[0]?.id || "");
      })
      .catch(() => setError("后端暂不可用，请先启动 FastAPI 服务。"));
  }, []);

  const currentCase = useMemo(
    () => cases.find((item) => item.id === selectedCase),
    [cases, selectedCase],
  );
  const resultStats = useMemo(() => buildMeetingResultStats(result), [result]);

  async function runDemo(caseId = selectedCase) {
    if (!caseId) return;
    await runTask(() => processDemo(caseId, processingMode), "样例处理失败，请检查后端服务。");
  }

  async function uploadAudio(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await runTask(
      () =>
        uploadAudioFile(file, processingMode, (progress) => {
          setUploadProgress(progress);
        }),
      "上传处理失败，请检查音频格式或后端服务。",
    );
    setUploadProgress(null);
    event.target.value = "";
  }

  async function runTask(task, message) {
    setLoading(true);
    setError("");
    try {
      setResult(await task());
    } catch (err) {
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <CommandHeader
        currentCase={currentCase}
        loading={loading}
        processingMode={processingMode}
        resultStats={resultStats}
        selectedCase={selectedCase}
        onRunDemo={() => runDemo()}
        uploadRef={uploadRef}
      />
      <input ref={uploadRef} className="hidden-input" type="file" accept="audio/*" onChange={uploadAudio} />

      <section className="layout">
        <Sidebar
          cases={cases}
          currentCase={currentCase}
          loading={loading}
          processingMode={processingMode}
          selectedCase={selectedCase}
          onProcessingModeChange={setProcessingMode}
          onSelectCase={setSelectedCase}
        />

        <section className="workspace">
          {error && <div className="error">{error}</div>}
          {uploadProgress && <UploadProgress progress={uploadProgress} />}
          {!result ? (
            <EmptyState loading={loading} />
          ) : (
            <>
              <Pipeline steps={result.steps} />
              <ProcessingDiagnostics metrics={result.signal_metrics} />
              <AudioCompare result={result} />
              <Transcript result={result} />
              <Summary summary={result.summary} />
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function CommandHeader({ currentCase, loading, processingMode, resultStats, selectedCase, onRunDemo, uploadRef }) {
  return (
    <section className="command-header">
      <div className="brand-lockup">
        <div className="brand-mark">
          <AudioWaveform size={25} />
        </div>
        <div>
          <p className="eyebrow">VOICE PIPELINE CONSOLE</p>
          <h1>智能会议语音分离与转写系统</h1>
          <div className="case-meta">
            <span>{currentCase?.name || "等待后端样例"}</span>
            <span>{currentCase?.noise_level || "状态"} / {currentCase?.duration || "未运行"}</span>
            <span>{processingMode === "fast" ? "分离快评" : "完整流程"}</span>
          </div>
        </div>
      </div>
      <div className="result-strip" aria-label="当前处理结果概览">
        {resultStats.map((item) => (
          <span key={item.label}>
            <strong>{item.value}</strong>
            {item.label}
          </span>
        ))}
      </div>
      <div className="hero-actions" aria-label="主要操作">
        <button className="primary-btn" onClick={onRunDemo} disabled={loading || !selectedCase}>
          {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          运行样例
        </button>
        <button className="secondary-btn" onClick={() => uploadRef.current?.click()} disabled={loading}>
          <Upload size={18} />
          上传音频
        </button>
      </div>
    </section>
  );
}

function Sidebar({
  cases,
  currentCase,
  loading,
  processingMode,
  selectedCase,
  onProcessingModeChange,
  onSelectCase,
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-heading">
        <h2>会议样例</h2>
        <span>{cases.length} cases</span>
      </div>
      <div className="case-list">
        {cases.map((item) => (
          <button
            key={item.id}
            className={`case-card ${item.id === selectedCase ? "active" : ""}`}
            onClick={() => onSelectCase(item.id)}
          >
            <span>{item.name}</span>
            <small>{item.scene}</small>
            <strong>{item.noise_level} / {item.duration}</strong>
          </button>
        ))}
      </div>
      <div className="mode-box">
        <span className="mode-label">处理路径</span>
        <div className="mode-toggle" role="group" aria-label="处理路径">
          <button
            className={processingMode === "fast" ? "active" : ""}
            onClick={() => onProcessingModeChange("fast")}
            disabled={loading}
            type="button"
          >
            分离快评
          </button>
          <button
            className={processingMode === "full" ? "active" : ""}
            onClick={() => onProcessingModeChange("full")}
            disabled={loading}
            type="button"
          >
            完整流程
          </button>
        </div>
        <p>{processingMode === "fast" ? "只跑 quality router 模型分离快评，跳过 ASR 与摘要。" : "跑增强、ASR、分离、摘要等完整链路。"}</p>
      </div>
      {currentCase && <p className="case-note">{currentCase.description}</p>}
    </aside>
  );
}

function UploadProgress({ progress }) {
  const label = progress.phase === "processing" ? "上传完成，正在处理音频" : `正在分块上传 ${progress.percent}%`;
  return (
    <div className="upload-progress">
      <div className="upload-progress-header">
        <strong>{label}</strong>
        <span>{formatBytes(progress.uploadedBytes)} / {formatBytes(progress.totalBytes)}</span>
      </div>
      <div className="upload-progress-track">
        <span style={{ width: `${progress.percent}%` }} />
      </div>
    </div>
  );
}

function formatBytes(bytes = 0) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function buildMeetingResultStats(result) {
  if (!result) {
    return [
      { label: "链路状态", value: "待命" },
      { label: "说话人数", value: "0" },
      { label: "质量评分", value: "-" },
      { label: "时间戳", value: "0" },
    ];
  }
  const metrics = result.signal_metrics || {};
  return [
    { label: "链路状态", value: "完成" },
    { label: "说话人数", value: metricValue(metrics, "检测说话人数", String(result.separated_tracks?.length || 0)) },
    { label: "质量评分", value: metricValue(metrics, "会议提取质量评分", "-") },
    { label: "语音覆盖", value: metricValue(metrics, "语音覆盖率", "-") },
    { label: "时间戳", value: String(result.transcript?.length || 0) },
  ];
}

function metricValue(metrics, key, fallback) {
  return metrics?.[key] || fallback;
}

function buildResultStats(result) {
  if (!result) {
    return [
      { label: "链路状态", value: "待命" },
      { label: "分离轨道", value: "0" },
      { label: "时间戳", value: "0" },
    ];
  }
  return [
    { label: "链路状态", value: "完成" },
    { label: "分离轨道", value: String(result.separated_tracks?.length || 0) },
    { label: "时间戳", value: String(result.transcript?.length || 0) },
  ];
}
