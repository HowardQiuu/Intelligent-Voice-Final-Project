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
          <div className="track-grid">
            {result.separated_tracks.map((track) => (
              <div className="audio-card separated" key={track.track_id}>
                <span>{track.label}</span>
                <small>{track.description}</small>
                <audio controls src={apiUrl(track.audio_url)} />
              </div>
            ))}
          </div>
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
