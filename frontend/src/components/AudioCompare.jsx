import React from "react";
import { FileAudio } from "lucide-react";
import { apiUrl } from "../api";

export function AudioCompare({ result }) {
  return (
    <section className="panel audio-panel">
      <div className="panel-title">
        <FileAudio size={20} />
        <h2>预处理音频与中文会议转写</h2>
      </div>

      <div className="audio-grid">
        <div className="audio-card">
          <span>原始会议音频</span>
          <audio controls src={apiUrl(result.original_audio_url)} />
        </div>
        <div className="audio-card enhanced">
          <span>预处理后混合音频</span>
          <audio controls src={apiUrl(result.enhanced_audio_url)} />
        </div>
      </div>

      <TranscriptPreview title="合并转写预览" text={result.enhanced_asr_text} limit={260} />

      {result.separated_tracks?.length > 0 && (
        <div className="separation-list">
          <h3>说话人轨道 / 分离轨道</h3>
          <SpeakerCountDiagnostics estimation={result.speaker_count_estimation} />
          <div className="track-grid">
            {result.separated_tracks.map((track) => (
              <div className="audio-card separated" key={track.track_id}>
                <span>{track.label}</span>
                <small>{track.description}</small>
                <audio controls src={apiUrl(track.audio_url)} />
                {track.track_enhancement_status && <small>增强: {track.track_enhancement_status}</small>}
                {track.asr_status && <small>ASR: {track.asr_status}</small>}
                <TranscriptPreview text={track.asr_text} limit={120} compact />
              </div>
            ))}
          </div>
          <TrackAlignmentOverview
            alignment={result.separation_alignment}
            transcript={result.transcript || []}
            tracks={result.separated_tracks || []}
          />
          <TextgridEvaluation evaluation={result.separation_evaluation} />
        </div>
      )}

      <ChunkPlan chunks={result.processing_chunks || []} />
    </section>
  );
}

function TranscriptPreview({ title, text, limit = 220, compact = false }) {
  const normalized = normalizeText(text);
  if (!normalized) return null;
  const clipped = normalized.length > limit;
  const preview = clipped ? `${normalized.slice(0, limit)}...` : normalized;
  return (
    <div className={compact ? "track-transcript-preview" : "enhanced-asr-card"}>
      {title && <h3>{title}</h3>}
      <p>{preview}</p>
      {clipped && (
        <details>
          <summary>展开全文</summary>
          <p>{normalized}</p>
        </details>
      )}
    </div>
  );
}

function SpeakerCountDiagnostics({ estimation }) {
  if (!estimation || !estimation.status) return null;
  const tracks = estimation.tracks || [];
  const clusters = estimation.clusters || [];
  return (
    <div className="speaker-count-panel">
      <div className="speaker-count-summary">
        <span>
          <strong>{estimation.global_estimated_speaker_count ?? estimation.estimated_speaker_count ?? 0}</strong>
          估计人数
        </span>
        <span>
          <strong>{estimation.embedding_backend || "-"}</strong>
          embedding
        </span>
        <span>
          <strong>{estimation.embedding_backend_status || estimation.status}</strong>
          后端状态
        </span>
        <span>
          <strong>{formatDecimal(estimation.cluster_stability ?? estimation.stability_score)}</strong>
          聚类稳定性
        </span>
      </div>
      {clusters.length > 0 && (
        <div className="speaker-cluster-grid">
          {clusters.map((cluster) => (
            <div className="speaker-cluster" key={cluster.global_speaker_id || cluster.cluster_id}>
              <strong>{speakerDisplayName(cluster.global_speaker_id || cluster.cluster_id)}</strong>
              <span>{(cluster.track_ids || []).length} tracks</span>
              <small>
                sim {formatDecimal(cluster.mean_similarity)} / stable {formatDecimal(cluster.stability_score)}
              </small>
            </div>
          ))}
        </div>
      )}
      {tracks.length > 0 && (
        <div className="speaker-track-tags">
          {tracks
            .filter((track) => track.accepted)
            .slice(0, 12)
            .map((track) => (
              <span key={track.track_id}>
                {track.label || speakerDisplayName(track.global_speaker_id)} / q {formatDecimal(track.quality_score)}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}

function TrackAlignmentOverview({ alignment, transcript, tracks }) {
  const alignmentInfo = alignment || {};
  const summaries = summarizeTrackTranscript(transcript, tracks);
  if (!summaries.length && alignmentInfo.status !== "ok") return null;
  return (
    <div className="alignment-panel">
      <h3>轨道转写概览</h3>
      <div className="alignment-summary">
        <span>已对齐分段：{alignmentInfo.aligned_segments || 0}</span>
        <span>已识别轨道：{summaries.length}</span>
        <span>分离轨道数：{alignmentInfo.track_count || tracks.length || 0}</span>
      </div>
      <div className="alignment-grid">
        {summaries.map((item) => (
          <div className="alignment-row" key={item.track_id}>
            <strong>{formatTrackTime(item.start, item.end)}</strong>
            <span>{item.label}</span>
            <small>{item.segmentCount} 个转写段</small>
            <p>{previewText(item.text, 90) || "暂无有效转写"}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function summarizeTrackTranscript(transcript, tracks) {
  const labels = new Map((tracks || []).map((track) => [track.track_id, track.label || track.track_id]));
  const grouped = new Map();
  for (const segment of transcript || []) {
    const trackId = segment.primary_track_id || (segment.separation_tracks || [])[0];
    if (!trackId) continue;
    const current = grouped.get(trackId) || {
      track_id: trackId,
      label: segment.primary_track_label || labels.get(trackId) || trackId,
      start: segment.start,
      end: segment.end,
      segmentCount: 0,
      texts: [],
    };
    current.segmentCount += 1;
    current.start = earliestTime(current.start, segment.start);
    current.end = latestTime(current.end, segment.end);
    if (segment.text) current.texts.push(segment.text);
    grouped.set(trackId, current);
  }
  return Array.from(grouped.values()).map((item) => ({
    ...item,
    text: normalizeText(item.texts.join(" ")),
  }));
}

function normalizeText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function previewText(value, limit) {
  const normalized = normalizeText(value);
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, limit)}...`;
}

function earliestTime(first, second) {
  return timeSeconds(second) < timeSeconds(first) ? second : first;
}

function latestTime(first, second) {
  return timeSeconds(second) > timeSeconds(first) ? second : first;
}

function timeSeconds(value) {
  const parts = String(value || "0").split(":").map(Number);
  if (parts.some(Number.isNaN)) return 0;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return Number(value || 0);
}

function formatTrackTime(start, end) {
  return `${start || "00:00"}-${end || "00:00"}`;
}

function TextgridEvaluation({ evaluation }) {
  const matches = evaluation?.track_matches || [];
  if (!evaluation || evaluation.source !== "textgrid") return null;
  return (
    <div className="alignment-panel">
      <h3>TextGrid 分离效果验证</h3>
      <div className="alignment-summary">
        <span>状态：{evaluation.status}</span>
        <span>参考说话人：{evaluation.reference_speaker_count || 0}</span>
        <span>参考重叠比例：{formatPercent(evaluation.reference_overlap_ratio)}</span>
      </div>
      {matches.length > 0 && (
        <div className="alignment-grid">
          {matches.map((item) => (
            <div className="alignment-row" key={item.track_id}>
              <strong>{item.track_label || item.track_id}</strong>
              <span>{item.matched_reference_speaker || "未匹配"}</span>
              <small>
                文本相似度 {formatPercent(item.text_similarity)} / {item.match_method}
              </small>
            </div>
          ))}
        </div>
      )}
      {evaluation.overlap_segments?.length > 0 && (
        <div className="overlap-list">
          {evaluation.overlap_segments.slice(0, 4).map((item, index) => (
            <span key={`${item.start}-${index}`}>
              {item.start}-{item.end}: {(item.speakers || []).join(" + ")}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function formatPercent(value) {
  const numeric = Number(value || 0);
  return `${Math.round(numeric * 100)}%`;
}

function formatDecimal(value) {
  const numeric = Number(value || 0);
  return numeric.toFixed(2);
}

function speakerDisplayName(value) {
  const text = String(value || "");
  const match = text.match(/^speaker_(\d+)$/);
  if (!match) return text || "未分配";
  const index = Number(match[1]);
  if (index >= 1 && index <= 26) return `说话人 ${String.fromCharCode(64 + index)}`;
  return `说话人 ${index}`;
}

function ChunkPlan({ chunks }) {
  if (!chunks.length) return null;
  return (
    <div className="chunk-list">
      <h3>分块处理计划</h3>
      <div className="chunk-grid">
        {chunks.slice(0, 8).map((chunk) => (
          <div className="chunk-item" key={chunk.chunk_id}>
            <strong>{chunk.chunk_id}</strong>
            <span>
              {chunk.start} - {chunk.end}
            </span>
            <small>{chunk.status}</small>
          </div>
        ))}
      </div>
      {chunks.length > 8 && <p className="chunk-more">还有 {chunks.length - 8} 个分块未展开显示</p>}
    </div>
  );
}
