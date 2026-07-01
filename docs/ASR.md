# ASR 转写模块

## 模块目标

ASR 模块负责把增强后的会议音频转成结构化中文文本，并尽量保留时间戳和说话人信息。它服务于两个下游目标：

- 前端展示可读会议转写。
- 给摘要模块提供可靠文本输入。

ASR 只在 `processing_mode=full` 的完整流程中运行；`fast` 模式会跳过 ASR，只返回分离快评结果。

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

代码默认后端是 `funasr`，但 `backend/.env.example` 当前示例配置为 `faster-whisper`，实际运行以本地 `backend/.env` 为准。如果 FunASR 环境可用，系统可以使用中文会议路线进行识别、VAD 和说话人分段；如果不可用，则使用 faster-whisper 或本地兜底文本。

## 汇报讲解：ASR 每一步具体做什么

完整 ASR 链路可以按下面顺序讲：

```text
增强后音频
-> 选择 ASR 后端
-> 加载/复用模型缓存
-> 长音频分块
-> 语音识别
-> 时间戳和说话人标签整理
-> 输出 transcript 和 enhanced_asr_text
```

### 1. 为什么 ASR 用增强后音频

ASR 需要的是尽量清晰、响度稳定的人声输入。增强模块已经做了降噪、响度归一化和限幅，所以 ASR 默认读取 `enhanced_audio_url` 对应的音频。分离模块可以使用 raw，但 ASR 更适合使用 enhanced。

### 2. FunASR / SenseVoice 路径

FunASR 路径用于中文会议场景，默认模型组合：

```text
FUNASR_MODEL=iic/SenseVoiceSmall
FUNASR_VAD_MODEL=fsmn-vad
FUNASR_PUNC_MODEL=ct-punc
FUNASR_SPK_MODEL=cam++
FUNASR_SPK_MODE=vad_segment
FUNASR_BATCH_SIZE_S=60
```

每个组件的作用：

- `SenseVoiceSmall`：执行中文语音识别，把语音转成文本。
- `fsmn-vad`：做语音活动检测，找到哪里有人声，减少静音段干扰。
- `ct-punc`：给识别文本补标点，提高摘要可读性。
- `cam++`：做说话人分析，为片段提供 speaker 标签。
- `vad_segment`：以 VAD 切出的语音段为基础做说话人分段。

FunASR 返回的 `sentence_info` 会被整理成统一 transcript：

```text
start -> 片段开始时间
end -> 片段结束时间
speaker -> 说话人 A / B / C
text -> 清理后的文本
```

如果 SenseVoice 输出 `<|...|>` 事件标签，系统会移除语言标签、标点标签等控制标签，并把有效事件标签写入 `SenseVoice事件标签` 指标。

### 3. faster-whisper fallback 路径

如果 FunASR 不可用，系统自动回退到 faster-whisper。它会根据配置选择设备和计算类型：

```text
ASR_DEVICE=auto
ASR_COMPUTE_TYPE=auto
ASR_LANGUAGE=zh
ASR_VAD_FILTER=true
ASR_BEAM_SIZE=1
ASR_BEST_OF=1
```

`auto` 会优先尝试可用 GPU；如果模型加载失败或 GPU 不可用，会回到 CPU/int8 等更稳的组合。模型对象会缓存，避免每次上传都重新加载。

faster-whisper 输出的 segment 会被转换成：

```text
start/end: 秒数格式化为 00:00
speaker: 默认“说话人”
text: segment.text
```

由于 faster-whisper 本身不提供项目所需的中文说话人分段，speaker 标签会弱于 FunASR 路径；但它能保证基础转写不中断。

### 4. 长音频分块 ASR

当音频超过：

```text
ASR_MAX_SECONDS=600
```

系统会按：

```text
ASR_CHUNK_SECONDS=60
ASR_MAX_CHUNKS=240
```

把音频切成多个 chunk。每个 chunk 单独转写，再把片段时间戳加上 chunk offset，拼回完整会议时间轴。

汇报口径：分块不是为了改变识别算法，而是为了降低长音频一次性推理的内存风险，并保留全局时间戳。

### 5. ASR 输出怎样进入后续模块

ASR 返回两类文本：

```text
enhanced_asr_text: 所有 transcript 片段拼接后的完整文本
transcript: 带 start/end/speaker/text 的结构化片段列表
```

后续用途：

- 摘要模块读取 `enhanced_asr_text` 和 `transcript`。
- 主题分组模块读取 `transcript`。
- 分离对齐模块读取 `transcript` 的时间戳，再与分离轨道能量对齐。
- 前端 Transcript 组件直接展示片段列表。

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
