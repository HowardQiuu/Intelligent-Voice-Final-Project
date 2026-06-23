import React from "react";
import { Loader2, Mic2 } from "lucide-react";

export function EmptyState({ loading }) {
  return (
    <div className="empty-state">
      {loading ? <Loader2 className="spin" size={42} /> : <Mic2 size={42} />}
      <h2>{loading ? "正在处理会议音频" : "选择样例或上传音频开始演示"}</h2>
      <p>系统会展示从原始会议音频到结构化会议纪要的完整处理过程。</p>
    </div>
  );
}
