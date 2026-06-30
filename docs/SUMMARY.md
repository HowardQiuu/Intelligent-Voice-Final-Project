# 会议摘要模块

## 模块目标

摘要模块负责把 ASR 输出的会议文本整理成适合阅读的会议纪要，包括摘要、要点、决策和待办事项。

## 最终保留路径

当前摘要保留 LLM 主路径和本地兜底路径：

```text
ASR transcript
-> topic grouping
-> OpenAI-compatible LLM summary
-> content-aware local fallback summary
```

默认配置示例：

```text
LLM_ENABLED=true
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=20
LLM_TOPIC_WINDOW_SECONDS=120
LLM_TOPIC_MAX_BLOCKS=80
```

`LLM_API_KEY` 不应提交到仓库，只在本地 `backend/.env` 配置。

## 输出内容

摘要模块输出：

- 会议标题。
- 简短摘要。
- 关键讨论点。
- 会议决策。
- 待办事项。
- 主题分组后的转写块。
- 摘要生成状态和兜底原因。

## 兜底策略

以下情况会自动使用本地摘要兜底：

- 未配置 `LLM_API_KEY`。
- `LLM_ENABLED=false`。
- API 超时或返回格式异常。
- 转写文本过短或为空。

本地兜底会尽量从实际转写文本中抽取内容，而不是返回固定模板，保证演示时仍能看到与会议相关的摘要。

## 前端展示

前端 `Summary` 组件展示摘要、决策和待办；`Transcript` 组件展示分主题转写。若 LLM 不可用，页面会显示兜底摘要状态，便于汇报时解释系统鲁棒性。
