import React, { useMemo, useState } from "react";
import { Mic2 } from "lucide-react";

const DEFAULT_VISIBLE_TOPICS = 4;

export function Transcript({ result }) {
  const [expanded, setExpanded] = useState(false);
  const speakerMap = useMemo(() => buildSpeakerMap(result), [result]);
  const topics = useMemo(() => normalizeTopics(result, speakerMap), [result, speakerMap]);
  const visibleTopics = expanded ? topics : topics.slice(0, DEFAULT_VISIBLE_TOPICS);
  const segmentCount = (result.transcript || []).length;
  const blockCount = topics.reduce((total, topic) => total + topic.blocks.length, 0);
  const hiddenCount = Math.max(topics.length - visibleTopics.length, 0);

  return (
    <section className="panel">
      <div className="panel-title transcript-title">
        <span>
          <Mic2 size={20} />
          <h2>主题化时间戳转写</h2>
        </span>
        <small>
          {topics.length} 个主题 / {blockCount} 个时间块 / {segmentCount} 条时间戳
        </small>
      </div>

      <div className="topic-transcript">
        {visibleTopics.map((topic) => (
          <article className="topic-card" key={topic.topic_id}>
            <div className="topic-card-header">
              <div>
                <h3>{topic.title}</h3>
                {topic.summary && <p>{topic.summary}</p>}
              </div>
              <span>{topic.blocks.length} 个时间块</span>
            </div>

            <div className="topic-block-strip">
              {topic.blocks.map((block) => (
                <section className="topic-time-block" key={block.block_id}>
                  <div className="topic-time-header">
                    <time>{block.start} - {block.end}</time>
                    <span>{block.segments.length} 条</span>
                  </div>
                  {block.summary && <p className="topic-block-summary">{block.summary}</p>}
                  <div className="timestamp-list">
                    {block.segments.map((seg, index) => (
                      <p key={`${seg.start}-${index}`}>
                        <time>{seg.start} - {seg.end}</time>
                        <strong>{seg.speaker || "说话人"}</strong>
                        <span>{seg.text}</span>
                      </p>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </article>
        ))}
      </div>

      {topics.length > DEFAULT_VISIBLE_TOPICS && (
        <button className="transcript-toggle" type="button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "收起主题转写" : `展开全部，剩余 ${hiddenCount} 个主题`}
        </button>
      )}
    </section>
  );
}

function normalizeTopics(result, speakerMap) {
  const topics = result.transcript_topics?.length > 0 ? result.transcript_topics : fallbackTopics(result.transcript || []);
  return topics.map((topic) => ({
    ...topic,
    blocks: (topic.blocks || []).map((block) => {
      const segments = (block.segments || []).map((segment) => ({
        ...segment,
        speaker: speakerMap.get(segmentKey(segment)) || speakerMap.get(normalizeSpeaker(segment.speaker)) || segment.speaker,
      }));
      return {
        ...block,
        segments,
        summary: buildBlockSummary(segments),
      };
    }),
  }));
}

function fallbackTopics(segments) {
  if (segments.length === 0) return [];
  return [
    {
      topic_id: "topic_fallback",
      title: "会议转写",
      summary: "后端未返回主题分类时的兜底转写分组。",
      blocks: [
        {
          block_id: "block_fallback",
          start: segments[0]?.start || "00:00",
          end: segments[segments.length - 1]?.end || "00:00",
          summary: "",
          segments,
        },
      ],
    },
  ];
}

function buildSpeakerMap(result) {
  const tracks = result.separated_tracks || [];
  const trackLabels =
    tracks.length > 0
      ? tracks.map((track, index) => track.label || `分离说话人 ${index + 1}`)
      : ["说话人 1", "说话人 2"];
  const map = new Map();
  const speakerOrder = [];

  (result.transcript || []).forEach((segment, index) => {
    const original = normalizeSpeaker(segment.speaker);
    if (original && !isGenericSpeaker(original) && !speakerOrder.includes(original)) {
      speakerOrder.push(original);
    }

    let label = original ? map.get(original) : "";
    if (!label) {
      if (original && !isGenericSpeaker(original)) {
        label = trackLabels[speakerOrder.indexOf(original) % trackLabels.length];
        map.set(original, label);
      } else {
        label = trackLabels[index % trackLabels.length];
      }
    }
    map.set(segmentKey(segment), label);
  });

  return map;
}

function buildBlockSummary(segments) {
  const text = segments
    .map((segment) => `${segment.speaker || "说话人"}: ${segment.text || ""}`)
    .join(" ");
  return text.length > 180 ? `${text.slice(0, 177)}...` : text;
}

function segmentKey(segment) {
  return `${segment.start || ""}|${segment.end || ""}|${segment.text || ""}`;
}

function normalizeSpeaker(value) {
  return String(value || "").trim();
}

function isGenericSpeaker(value) {
  return value === "说话人" || value === "未知说话人";
}
