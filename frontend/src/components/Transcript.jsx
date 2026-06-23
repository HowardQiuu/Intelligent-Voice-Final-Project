import React from "react";
import { Mic2 } from "lucide-react";

export function Transcript({ result }) {
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
