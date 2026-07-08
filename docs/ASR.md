# ASR 转写模块

## 模块目标

ASR 模块负责把分离后的说话人轨道转成结构化中文文本，并保留时间戳、轨道标签和可合并的会议全文。它只在 `processing_mode=full` 中运行；`fast` 模式会跳过 ASR。

## 当前输入

当前 ASR 输入不是整条混合音频，而是逐条增强后的分离轨道：

```text
separated track
-> track enhancement
-> enhanced track audio
-> ASR
```

在 `SeparatedTrack` 中：

```text
separated_audio_url  增强前的分离轨道
audio_url            增强后的轨道，ASR 实际读取它
```

如果某条轨道增强失败，`audio_url` 会回退到原始分离轨道。

## 后端路径

当前 ASR 保留主路径和兜底路径：

```text
FunASR / SenseVoice + VAD + CAM++ speaker analysis
-> faster-whisper fallback
-> placeholder transcript fallback
```

默认配置示例：

```text
ASR_BACKEND=faster-whisper
ASR_MODEL=small
ASR_LANGUAGE=zh
ASR_DEVICE=auto
ASR_COMPUTE_TYPE=auto
ASR_CHUNK_SECONDS=60
ASR_MAX_CHUNKS=240
ASR_VAD_FILTER=true
```

代码默认后端是 `funasr`，但 `backend/.env.example` 示例配置为 `faster-whisper`，实际运行以本地 `backend/.env` 为准。

## 执行顺序

每条轨道单独执行：

```text
enhanced track audio
-> 选择 ASR 后端
-> 加载或复用模型缓存
-> 必要时按时间分块
-> 语音识别
-> 格式化 start/end/speaker/text
-> 写回 track transcript
```

所有轨道识别完成后：

```text
track transcripts
-> 按时间排序
-> 合并同一轨道上的连续短片段
-> 生成 merged transcript
-> 生成 enhanced_asr_text
```

这样前端不会再展示大量只有一两个字的小片段。

## 输出字段

每条 `SeparatedTrack` 会包含：

```text
asr_text       该轨道完整转写文本
asr_status     该轨道 ASR 状态
transcript     该轨道结构化片段
```

每个 transcript 片段会带上：

```text
speaker
start
end
text
primary_track_id
primary_track_label
separation_tracks
```

最终 `ProcessResult.enhanced_asr_text` 是所有轨道转写按时间合并后的会议文本。

## 与分离对齐的关系

ASR 先在每条增强轨道上得到文本和时间戳，然后 `separation_alignment_service` 会基于轨道能量把片段标记到对应轨道。TextGrid 只作为事后评测参考，不参与 ASR、分离或对齐决策。

## 失败兜底

以下情况会自动兜底：

- FunASR 模型或依赖不可用。
- faster-whisper 加载失败。
- GPU 不可用时回退 CPU。
- 某条轨道音频缺失或识别失败。

兜底时系统仍返回结构化字段，保证摘要和页面展示不中断。
