import React from "react";
import { FileAudio } from "lucide-react";
import { apiUrl } from "../api";

export function AudioCompare({ result }) {
  return (
    <section className="panel audio-panel">
      <div className="panel-title">
        <FileAudio size={20} />
        <h2>增强音频与中文会议转写</h2>
      </div>

      <div className="audio-grid">
        <div className="audio-card">
          <span>原始会议音频</span>
          <audio controls src={apiUrl(result.original_audio_url)} />
        </div>
        <div className="audio-card enhanced">
          <span>增强后音频</span>
          <audio controls src={apiUrl(result.enhanced_audio_url)} />
        </div>
      </div>

      {result.enhanced_asr_text && (
        <div className="enhanced-asr-card">
          <h3>增强后转写</h3>
          <p>{result.enhanced_asr_text}</p>
        </div>
      )}

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
              </div>
            ))}
          </div>
          <SeparationAlignment alignment={result.separation_alignment} transcript={result.transcript || []} />
          <TextgridEvaluation evaluation={result.separation_evaluation} />
        </div>
      )}

      {result.enhancement_visual_url && (
        <div className="enhancement-visual">
          <h3>语音增强可视化</h3>
          <img
            src={apiUrl(result.enhancement_visual_url)}
            alt="语音增强前后波形、噪声底和清晰度对比图"
          />
        </div>
      )}

      <ChunkPlan chunks={result.processing_chunks || []} />
    </section>
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

function SeparationAlignment({ alignment, transcript }) {
  if (!alignment || alignment.status !== "ok") return null;
  const alignedSegments = (transcript || []).filter((segment) => segment.primary_track_id);
  return (
    <div className="alignment-panel">
      <h3>ASR 分段与分离轨道对齐</h3>
      <div className="alignment-summary">
        <span>已对齐分段：{alignment.aligned_segments || 0}</span>
        <span>多轨重叠分段：{alignment.multi_track_segments || 0}</span>
        <span>分离轨道数：{alignment.track_count || 0}</span>
      </div>
      <div className="alignment-grid">
        {alignedSegments.slice(0, 6).map((segment, index) => (
          <div className="alignment-row" key={`${segment.start}-${segment.end}-${index}`}>
            <strong>
              {segment.start}-{segment.end}
            </strong>
            <span>{segment.primary_track_label || segment.primary_track_id}</span>
            <small>{segment.text || "无文本"}</small>
          </div>
        ))}
      </div>
    </div>
  );
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
