# 完整数据链路

本文说明当前 Web 展示和后端 `processing_mode=full` 的真实执行顺序。

## 入口

前端上传音频默认走分片上传：

```text
POST /api/upload-session
POST /api/upload-session/{upload_id}/chunk
POST /api/upload-session/{upload_id}/complete
```

小文件也可以直接走：

```text
POST /api/upload
```

两个入口最终都会进入 `process_audio_path()`。前端默认处理模式是 `full`。

```text
fast:  只做分离快评，跳过逐轨增强、ASR、主题和摘要
full:  执行当前完整展示链路
```

## Full 模式总流程

```text
原始混合音频
-> 保存到 backend/app/static/uploads
-> FFmpeg 归一化 / 简单预处理为单声道 WAV
-> 规划逻辑分块
-> 分离质量路由或 demo clean source 分离
-> 每条分离轨道单独增强
-> 每条增强轨道单独 ASR
-> 合并逐轨 transcript
-> ASR 时间戳与分离轨道能量对齐
-> TextGrid 事后验证（有参考标注时）
-> 主题提取
-> 会议摘要
-> ProcessResult 返回前端
```

关键点：混合音频不会先增强再分离。分离输入默认是 `normalized`，也就是归一化和简单处理后的混合语音。增强发生在分离之后，逐条作用于 `separated_tracks`。

## 1. 输入预处理

上传文件先被保存，再通过 `normalize_upload()` 转成稳定的单声道 WAV：

```text
raw upload
-> highpass + loudness normalize + limiter
-> normalized wav
```

这一步只用于获得稳定格式、响度和采样参数，不做神经网络降噪。

## 2. 语音分离

默认配置：

```text
SEPARATION_INPUT_SOURCE=normalized
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
SEPARATION_DEMO_CLEAN_SOURCES=true
```

如果 `SEPARATION_DEMO_CLEAN_SOURCES=true` 且文件名或样例路径能匹配 `data/near_mix_dataset_v1/manifest.jsonl`，分离阶段会直接返回 manifest 中的 clean source 轨道，用于快速展示。前端仍看到“分离轨道”；后续的逐轨增强和逐轨 ASR 仍真实执行。

如果没有匹配 clean source，则走真实分离质量路由：

```text
normalized mixture
-> Libri2Mix SepFormer
-> ClearVoice MossFormer2_SS_16K
-> ReSepFormer
-> quality score / diagnostic rerank
-> selected separated tracks
```

## 3. 逐轨增强

分离产物进入 `_enhance_separated_tracks()`。每条轨道独立调用 `enhance_uploaded_audio()`：

```text
separated track 1 -> enhancement -> enhanced track 1
separated track 2 -> enhancement -> enhanced track 2
...
```

返回给前端的 `SeparatedTrack.audio_url` 指向增强后的单轨音频；原始分离轨道保存在 `SeparatedTrack.separated_audio_url`。如果某条轨道增强失败，该轨道会回退到原分离音频，并在 `track_enhancement_status` 中标记 `fallback`。

## 4. 逐轨 ASR

ASR 不再读取整条增强混合音频，而是读取增强后的每条分离轨道：

```text
enhanced track audio
-> transcribe_audio()
-> track transcript
-> merge all track transcripts by timestamp
```

每个 track 会附带：

```text
asr_text
asr_status
transcript
primary_track_id / primary_track_label
```

为了前端可读性，连续短片段会按轨道和时间间隔合并，避免出现大量只有一两个字的小卡片。

## 5. 对齐、主题和摘要

逐轨 transcript 合并后进入：

```text
align_transcript_to_separation_tracks()
-> classify_transcript_topics()
-> generate_summary()
```

`separation_alignment` 使用 ASR 时间戳和轨道能量，把每个片段映射到主要轨道。TextGrid 只用于事后评测，不参与分离或对齐决策。

## 返回字段

核心字段：

- `original_audio_url`：原始上传音频。
- `enhanced_audio_url`：预处理后的混合音频 URL，前端标记为“预处理后混合音频”。
- `separated_tracks`：分离轨道列表。每条轨道的 `audio_url` 是增强后单轨，`separated_audio_url` 是增强前分离轨道。
- `direct_asr_text` / `enhanced_asr_text`：逐轨 ASR 合并后的文本。
- `transcript`：带时间戳、说话人和轨道信息的片段。
- `transcript_topics`：主题分组结果。
- `separation_alignment`：ASR 片段到轨道的能量对齐结果。
- `separation_evaluation`：TextGrid 事后验证结果。
- `summary`：会议摘要、关键词、决策和待办。
- `signal_metrics`：预处理、分离、逐轨增强、ASR、摘要和运行耗时等诊断指标。

## Fast 模式

Fast 模式只保留快速分离展示：

```text
原始音频
-> 分离质量路由或 demo clean source 分离
-> separated_tracks
-> fallback summary
```

该模式不会执行逐轨增强、ASR、主题、摘要和轨道对齐。展示完整系统能力时使用默认 Full 模式。
