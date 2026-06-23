import React from "react";
import { ClipboardList } from "lucide-react";

export function Summary({ summary, metrics }) {
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
            <p key={key}>
              <span>{key}</span>
              <strong>{value}</strong>
            </p>
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
