import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AudioWaveform,
  CheckCircle2,
  ClipboardList,
  FileAudio,
  Loader2,
  Mic2,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function apiUrl(path) {
  if (path?.startsWith("/")) return `${API_BASE}${path}`;
  return path;
}

function App() {
  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const uploadRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/demo-cases`)
      .then((res) => res.json())
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
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/process-demo/${caseId}`, { method: "POST" });
      if (!res.ok) throw new Error("处理失败");
      setResult(await res.json());
    } catch (err) {
      setError("样例处理失败，请检查后端服务。");
    } finally {
      setLoading(false);
    }
  }

  async function uploadAudio(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error("上传失败");
      setResult(await res.json());
    } catch (err) {
      setError("上传处理失败，请检查音频格式或后端服务。");
    } finally {
      setLoading(false);
      event.target.value = "";
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">智能语音处理课程项目 Demo</p>
          <h1>复杂会议场景下的智能语音分离与转写系统</h1>
          <p className="hero-copy">
            用演示缓存与可替换模型接口跑通语音增强、说话人处理、自动转写和会议纪要生成链路。
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-btn" onClick={() => runDemo()} disabled={loading || !selectedCase}>
            {loading ? <Loader2 className="spin" size={18} /> : <AudioWaveform size={18} />}
            运行样例
          </button>
          <button className="secondary-btn" onClick={() => uploadRef.current?.click()} disabled={loading}>
            <Upload size={18} />
            上传音频
          </button>
          <input ref={uploadRef} className="hidden-input" type="file" accept="audio/*" onChange={uploadAudio} />
        </div>
      </section>

      <section className="layout">
        <aside className="sidebar">
          <h2>会议样例</h2>
          <div className="case-list">
            {cases.map((item) => (
              <button
                key={item.id}
                className={`case-card ${item.id === selectedCase ? "active" : ""}`}
                onClick={() => setSelectedCase(item.id)}
              >
                <span>{item.name}</span>
                <small>{item.scene}</small>
                <strong>{item.noise_level}噪声 · {item.duration}</strong>
              </button>
            ))}
          </div>
          {currentCase && <p className="case-note">{currentCase.description}</p>}
        </aside>

        <section className="workspace">
          {error && <div className="error">{error}</div>}
          {!result ? (
            <EmptyState loading={loading} />
          ) : (
            <>
              <Pipeline steps={result.steps} />
              <AudioCompare result={result} />
              <Transcript result={result} />
              <Summary summary={result.summary} metrics={result.signal_metrics} />
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function EmptyState({ loading }) {
  return (
    <div className="empty-state">
      {loading ? <Loader2 className="spin" size={42} /> : <Mic2 size={42} />}
      <h2>{loading ? "正在处理会议音频" : "选择样例或上传音频开始演示"}</h2>
      <p>系统会展示从原始会议音频到结构化会议纪要的完整处理过程。</p>
    </div>
  );
}

function Pipeline({ steps }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <ClipboardList size={20} />
        <h2>处理链路</h2>
      </div>
      <div className="pipeline">
        {steps.map((step) => (
          <div className="step" key={step.key}>
            <CheckCircle2 size={18} />
            <span>{step.name}</span>
            <small>{step.detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function AudioCompare({ result }) {
  return (
    <section className="panel two-col">
      <div>
        <div className="panel-title">
          <FileAudio size={20} />
          <h2>增强前后试听</h2>
        </div>
        <div className="audio-card">
          <span>原始会议音频</span>
          <audio controls src={apiUrl(result.original_audio_url)} />
        </div>
        <div className="audio-card enhanced">
          <span>增强后音频</span>
          <audio controls src={apiUrl(result.enhanced_audio_url)} />
        </div>
        {result.separated_tracks?.length > 0 && (
          <div className="separation-list">
            <h3>语音分离轨道</h3>
            {result.separated_tracks.map((track) => (
              <div className="audio-card separated" key={track.track_id}>
                <span>{track.label}</span>
                <small>{track.description}</small>
                <audio controls src={apiUrl(track.audio_url)} />
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="asr-compare">
        <h3>转写对比</h3>
        <label>直接转写</label>
        <p>{result.direct_asr_text}</p>
        <label>增强后转写</label>
        <p>{result.enhanced_asr_text}</p>
      </div>
    </section>
  );
}

function Transcript({ result }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <Mic2 size={20} />
        <h2>带时间戳转写</h2>
      </div>
      <div className="transcript">
        {result.transcript.map((seg, index) => (
          <article className="segment" key={`${seg.start}-${index}`}>
            <time>{seg.start} - {seg.end}</time>
            <strong>{seg.speaker}</strong>
            <p>{seg.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function Summary({ summary, metrics }) {
  return (
    <section className="panel two-col">
      <div>
        <div className="panel-title">
          <ClipboardList size={20} />
          <h2>会议纪要输出</h2>
        </div>
        <h3>{summary.title}</h3>
        <p className="abstract">{summary.abstract}</p>
        <div className="tags">
          {summary.keywords.map((keyword) => <span key={keyword}>{keyword}</span>)}
        </div>
      </div>
      <div className="summary-grid">
        <InfoList title="关键决策" items={summary.decisions} />
        <InfoList title="待办事项" items={summary.action_items} />
        <div className="metric-box">
          <h3>处理指标</h3>
          {Object.entries(metrics).map(([key, value]) => (
            <p key={key}><span>{key}</span><strong>{value}</strong></p>
          ))}
        </div>
      </div>
    </section>
  );
}

function InfoList({ title, items }) {
  return (
    <div className="info-list">
      <h3>{title}</h3>
      <ul>
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
