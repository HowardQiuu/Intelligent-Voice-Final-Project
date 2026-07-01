# 新音频完整数据链路

本文从“上传一条新的会议音频”出发，说明当前项目实际会怎样处理数据。系统可以分成四个核心处理段：语音增强、语音分离、ASR 转写、摘要提取。

## 入口

前端上传音频时走分片上传接口：

```text
POST /api/upload-session
POST /api/upload-session/{upload_id}/chunk
POST /api/upload-session/{upload_id}/complete
```

小文件也可以直接走：

```text
POST /api/upload
```

两个入口最终都会调用后端 `process_audio_path()`。前端有两个处理模式：

```text
fast:  分离快评，只跑原始音频上的分离质量路由，跳过增强、ASR、摘要
full:  完整流程，跑增强、分离、转写、主题整理和摘要
```

如果要展示“增强、分离、转写、摘要”四段完整链路，需要选择 `full` 模式或传入 `processing_mode=full`。

## Full 模式总流程

```text
上传原始音频
-> 保存到 backend/app/static/uploads
-> FFmpeg 标准化为单声道 WAV
-> 规划逻辑分块
-> 语音增强质量路由
-> 增强前后可视化
-> ASR 转写
-> 分离质量路由
-> ASR 片段与分离轨道能量对齐
-> TextGrid 事后验证（有参考标注时）
-> 会议质量诊断指标
-> 会议摘要
-> 主题分组
-> ProcessResult 返回前端
```

需要注意当前完整链路里 ASR 先于分离运行。这样分离模块可以拿到 transcript，用它估计重叠说话比例和辅助质量路由；分离完成后，再把 ASR 时间片段对齐到实际分离轨道。

## 1. 语音增强

输入是标准化后的 WAV。增强模块先做响度预增益，再根据质量路由尝试不同增强候选：

```text
normalized wav
-> audibility pregain
-> DeepFilterNet candidate
-> ClearVoice enhancement candidate
-> quality score
-> selected enhanced wav
-> post loudness normalization
```

当前默认候选：

```text
ENHANCEMENT_CANDIDATES=deepfilternet,clearvoice
DEEPFILTERNET_BACKEND=cli
CLEARVOICE_ENHANCE_MODEL=MossFormer2_SE_48K
```

若候选全部失败，系统不会中断，而是返回预增益后的音频作为兜底。增强结果主要供 ASR、摘要和前端试听使用。

## 2. 语音分离

分离模块默认更偏向使用原始混合音频，而不是增强音频：

```text
SEPARATION_INPUT_SOURCE=raw
```

原因是盲源分离模型通常更接近原始混合语音的训练分布。完整链路中仍然会保留增强音频给 ASR 使用，但分离质量路由会按配置选择 raw、normalized 或 enhanced。

默认分离候选：

```text
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
```

候选会分别输出多条说话人轨道，系统再根据轨道数量、能量、相关性、重叠诊断、模型偏置和 transcript 重叠比例打分。若所有真实候选失败，只返回 placeholder 轨道，不使用 ASR 说话人分段去伪造分离结果。

## 3. ASR 转写

ASR 使用增强后的音频作为输入：

```text
enhanced wav
-> FunASR / SenseVoice + VAD + CAM++ speaker analysis
-> faster-whisper fallback
-> placeholder fallback
```

默认配置文件中 `ASR_BACKEND=faster-whisper`，代码默认值为 `funasr`。实际运行以 `backend/.env` 为准。输出包含完整文本、按时间戳切分的 transcript、说话人标签和 ASR 运行指标。

## 4. 摘要提取

摘要模块使用对齐后的 transcript 和完整 ASR 文本：

```text
aligned transcript
-> OpenAI-compatible LLM summary
-> content-aware local fallback
```

LLM 可用时会生成会议标题、关键词、摘要、决策和待办。未配置 `LLM_API_KEY`、LLM 关闭或调用失败时，会使用本地抽取式兜底摘要，尽量从实际转写内容中提取信息。

主题分组是摘要之后的另一个下游步骤，由 `transcript_topic_service` 生成 `transcript_topics`，用于前端按主题展示转写块；它不是 `summary_service` 的前置输入。

## 结果返回

最终 `ProcessResult` 返回给前端的核心字段包括：

- `original_audio_url`：原始或标准化后的试听地址。
- `enhanced_audio_url`：增强后音频地址。
- `enhancement_visual_url`：增强前后可视化图。
- `processing_chunks`：逻辑分块计划。
- `separated_tracks`：分离轨道列表。
- `direct_asr_text` / `enhanced_asr_text`：转写文本。
- `transcript`：带时间戳、说话人和分离轨道对齐信息的片段。
- `transcript_topics`：主题分组后的转写块。
- `separation_alignment`：ASR 片段到分离轨道的能量对齐结果。
- `separation_evaluation`：有 TextGrid 参考时的事后评测。
- `summary`：会议摘要、关键词、决策和待办。
- `signal_metrics`：增强、分离、ASR、摘要、运行耗时等诊断指标。

## Fast 模式

Fast 模式用于快速评估分离模型效果：

```text
上传原始音频
-> 分离质量路由
-> separated_tracks
-> fallback summary
```

该模式不会执行增强、ASR、摘要和分离轨道对齐，返回文本会明确标记：

```text
Fast separation path: ASR skipped; quality router model separation only.
```

因此汇报完整系统能力时应使用 Full 模式；调试分离候选和模型效果时使用 Fast 模式更快。
