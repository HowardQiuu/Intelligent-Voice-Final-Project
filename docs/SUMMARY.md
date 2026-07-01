# 会议摘要模块

## 模块目标

摘要模块负责把 ASR 输出的会议文本整理成适合阅读的会议纪要，包括摘要、要点、决策和待办事项。

摘要只在 `processing_mode=full` 的完整流程中运行。`fast` 模式用于分离快评，会直接返回兜底摘要，不调用 LLM。

## 最终保留路径

当前摘要保留 LLM 主路径和本地兜底路径：

```text
aligned ASR transcript
-> OpenAI-compatible LLM summary
-> content-aware local fallback summary
```

这里的 `aligned ASR transcript` 指的是 ASR 片段已经经过分离轨道能量对齐，片段中可能带有 `primary_track_id`、`primary_track_label` 和 `separation_tracks`。

主题分组由 `transcript_topic_service` 独立生成 `transcript_topics`，用于前端按主题展示转写块；它不是摘要生成的前置步骤。

## 汇报讲解：摘要每一步具体做什么

摘要链路可以按下面顺序讲：

```text
ASR 完整文本 + 结构化 transcript
-> 构造会议纪要 prompt
-> 调用 OpenAI-compatible Chat Completions
-> 校验 JSON 字段
-> 低信息摘要检查
-> 本地抽取式兜底
```

### 1. 输入是什么

摘要模块接收两类输入：

```text
enhanced_asr_text: 完整转写文本
transcript: 带时间戳、说话人、文本的片段列表
```

prompt 会同时放入：

- 会议名称。
- 增强后 ASR 文本。
- 带时间戳和说话人标签的转写。

这样做的原因是：完整文本适合模型理解整体内容，带时间戳和说话人的 transcript 适合模型判断谁在什么时候说了什么。

### 2. LLM 主路径

摘要调用 OpenAI 兼容接口：

```text
POST {LLM_BASE_URL}/chat/completions
model = LLM_MODEL
temperature = 0.2
response_format = {"type": "json_object"}
```

system prompt 要求模型只输出 JSON，不输出 Markdown。JSON 必须包含：

```text
title
keywords
abstract
decisions
action_items
```

`temperature=0.2` 表示摘要生成偏稳定、少发散，适合会议纪要这种结构化任务。

### 3. JSON 校验

LLM 返回后，系统不会直接相信文本，而是：

```text
提取 JSON 对象
-> 校验字段类型
-> 清理字符串和列表
-> 缺失字段用安全默认值补齐
```

如果返回不是 JSON、字段缺失严重、HTTP 调用失败或超时，系统会进入本地兜底。

### 4. 低信息摘要检查

即使 LLM 成功返回 JSON，系统还会检查摘要是否“低信息”。如果转写本身有内容，但摘要里出现：

```text
未知会议
无法生成摘要
无决策
无待办
内容不完整
```

这类低价值标记，就认为 LLM 输出不可用，切换到本地抽取式兜底。这样可以避免页面上显示看似成功但没有信息量的纪要。

### 5. 本地抽取式兜底

本地兜底不是固定模板，而是从实际转写里抽内容：

```text
按中文标点切句
-> 前几句组成 abstract
-> 按关键词挑 decisions
-> 按关键词挑 action_items
-> 从文本中抽取 keywords
```

决策句关键词包括：

```text
确定、决定、确认、同意、采用、优先、可以、适合、需要
```

待办句关键词包括：

```text
后续、需要、准备、继续、确认、检查、优化、接入、补充、调整
```

关键词提取会优先匹配项目内置领域词，例如“语音增强”“语音转写”“说话人分段”等；不足时再按中文 2-6 字词频补齐。

### 6. 主题分组

主题分组不是摘要的前置步骤，而是另一个展示用下游。它会把 transcript 按时间窗口切块：

```text
LLM_TOPIC_WINDOW_SECONDS=120
LLM_TOPIC_MAX_BLOCKS=80
```

每个 block 包含开始时间、结束时间、片段列表和简短 summary。LLM 可用时，系统让模型把这些 block 归并成主题；LLM 不可用时，使用本地兜底主题。前端用 `transcript_topics` 展示“按主题组织的转写”。

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
