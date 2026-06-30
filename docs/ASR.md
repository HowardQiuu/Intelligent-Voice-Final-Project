# ASR 转写模块

## 模块目标

ASR 模块负责把增强后的会议音频转成结构化中文文本，并尽量保留时间戳和说话人信息。它服务于两个下游目标：

- 前端展示可读会议转写。
- 给摘要模块提供可靠文本输入。

## 最终保留路径

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

如果 FunASR 环境可用，系统可以使用中文会议路线进行识别、VAD 和说话人分段；如果不可用，则使用 faster-whisper 或本地兜底文本。

## 长音频处理

ASR 对长音频使用分块策略：

```text
ASR_MAX_SECONDS=600
ASR_CHUNK_SECONDS=60
ASR_MAX_CHUNKS=240
```

分块结果会重新拼接为带时间戳的会议转写，避免长会议一次性推理失败。

## 输出内容

ASR 模块输出：

- `enhanced_asr_text`：完整转写文本。
- `transcript`：带时间戳、说话人标签和文本的片段列表。
- `signal_metrics`：ASR 后端、设备、模型、分块状态和兜底状态。

转写结果同时供前端 Transcript 组件和摘要模块使用。

## 与分离轨道的关系

ASR 说话人分段输出的是时间轴标签，分离模块输出的是音频轨道。二者不是天然一一对应：

```text
ASR diarization: 某个时间段是谁在说话
speech separation: 某条音频轨道主要是谁的声音
```

在重叠说话场景中，同一个时间戳可以同时存在多个说话人。项目不会强制把时间轴压成“每一帧只能一个人”，而是在分离完成后，通过 `separation_alignment` 对齐层把 ASR 片段映射到已有分离轨道。这个对齐只使用 ASR 时间戳和分离轨道能量，不使用 TextGrid。

对齐后的 transcript 片段会额外包含：

```text
primary_track_id
primary_track_label
separation_tracks
```

其中 `separation_tracks` 允许一个时间段同时包含多条轨道，用于表达重叠说话。TextGrid 只在 `separation_evaluation` 中作为事后验证真值，用来比较分离结果是否接近正确说话人和真实重叠区域。

## 失败兜底

以下情况会自动兜底：

- FunASR 模型或依赖不可用。
- GPU 不可用时自动切 CPU。
- faster-whisper 加载失败。
- 音频格式异常或识别超时。

兜底时系统仍返回结构化文本，避免摘要和页面展示中断。
