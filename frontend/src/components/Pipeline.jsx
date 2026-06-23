import React from "react";
import { CheckCircle2, ClipboardList } from "lucide-react";

export function Pipeline({ steps }) {
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
