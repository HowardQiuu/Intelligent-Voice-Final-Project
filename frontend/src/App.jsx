import React, { useEffect, useMemo, useRef, useState } from "react";
import { AudioWaveform, Loader2, Upload } from "lucide-react";
import {
  fetchDemoCases,
  processDemo,
  processLocalFile,
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
  const [localPath, setLocalPath] = useState("");
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

  async function runDemo(caseId = selectedCase) {
    if (!caseId) return;
    await runTask(() => processDemo(caseId), "样例处理失败，请检查后端服务。");
  }

  async function uploadAudio(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await runTask(
      () =>
        uploadAudioFile(file, (progress) => {
          setUploadProgress(progress);
        }),
      "上传处理失败，请检查音频格式或后端服务。",
    );
    setUploadProgress(null);
    event.target.value = "";
  }

  async function runLocalFile() {
    const path = localPath.trim();
    if (!path) {
      setError("请先粘贴本地音频文件路径。");
      return;
    }
    await runTask(() => processLocalFile(path), "本地文件处理失败，请检查路径是否存在，或后端服务是否已启动。");
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
      <Hero loading={loading} selectedCase={selectedCase} onRunDemo={() => runDemo()} uploadRef={uploadRef} />
      <input ref={uploadRef} className="hidden-input" type="file" accept="audio/*" onChange={uploadAudio} />

      <section className="layout">
        <Sidebar
          cases={cases}
          currentCase={currentCase}
          loading={loading}
          localPath={localPath}
          selectedCase={selectedCase}
          onLocalPathChange={setLocalPath}
          onProcessLocalFile={runLocalFile}
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

function Hero({ loading, selectedCase, onRunDemo, uploadRef }) {
  return (
    <section className="hero">
      <div>
        <p className="eyebrow">智能语音处理课程项目 Demo</p>
        <h1>复杂会议场景下的智能语音分离与转写系统</h1>
        <p className="hero-copy">
          用演示缓存与可替换模型接口跑通语音增强、说话人处理、自动转写和会议纪要生成链路。
        </p>
      </div>
      <div className="hero-actions">
        <button className="primary-btn" onClick={onRunDemo} disabled={loading || !selectedCase}>
          {loading ? <Loader2 className="spin" size={18} /> : <AudioWaveform size={18} />}
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
  localPath,
  selectedCase,
  onLocalPathChange,
  onProcessLocalFile,
  onSelectCase,
}) {
  return (
    <aside className="sidebar">
      <h2>会议样例</h2>
      <div className="case-list">
        {cases.map((item) => (
          <button
            key={item.id}
            className={`case-card ${item.id === selectedCase ? "active" : ""}`}
            onClick={() => onSelectCase(item.id)}
          >
            <span>{item.name}</span>
            <small>{item.scene}</small>
            <strong>{item.noise_level}噪声 / {item.duration}</strong>
          </button>
        ))}
      </div>
      <div className="local-file-box">
        <label>高级兜底：本地大文件路径</label>
        <div className="local-file-row">
          <input
            value={localPath}
            onChange={(event) => onLocalPathChange(event.target.value)}
            placeholder="C:\\workshop\\...\\meeting.flac"
            disabled={loading}
          />
          <button className="secondary-action" onClick={onProcessLocalFile} disabled={loading || !localPath.trim()}>
            处理本地文件
          </button>
        </div>
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
