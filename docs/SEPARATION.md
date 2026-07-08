# 语音分离模块

## 模块位置

语音分离模块负责把混合会议音频拆成多条可试听的说话人轨道。当前完整链路中，它位于输入预处理之后、逐轨增强之前：

```text
原始混合音频
-> 归一化 / 简单预处理
-> 语音分离
-> 每条分离轨道单独增强
-> 每条增强轨道单独 ASR
```

因此分离阶段不使用增强后的混合音频。

## 输入选择

当前默认：

```text
SEPARATION_INPUT_SOURCE=normalized
```

含义是：使用归一化和简单处理后的混合音频作为分离输入。这样可以保证格式和响度稳定，同时避免神经网络增强先改变混合语音的频谱、相位或重叠结构。

支持的输入策略：

```text
normalized  默认，使用归一化后的混合音频
raw         使用原始上传音频
```

`process_audio_path()` 会把实际选择写入 `signal_metrics["separation_input_source"]`。

## Demo clean source 开关

为了快速展示，当前保留演示开关：

```text
SEPARATION_DEMO_CLEAN_SOURCES=true
```

当上传文件名、样例路径或 meeting id 能匹配 `data/near_mix_dataset_v1/manifest.jsonl` 时，分离阶段会直接返回 manifest 中的 clean source 轨道。前端表层仍展示为“分离轨道”，但底层分离产物来自 clean source。

注意：这个开关只替代分离产物。后面的逐轨增强、逐轨 ASR、时间戳对齐、主题和摘要仍按完整 pipeline 执行。

匹配不到 clean source 时，系统走真实盲源分离质量路由。

## 质量路由主路径

默认真实候选：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
SEPARATION_DIAGNOSTIC_RERANK=true
```

实际流程：

```text
normalized mixture
-> run Libri2Mix SepFormer candidate
-> run ClearVoice MossFormer2_SS_16K candidate
-> run ReSepFormer candidate
-> score candidates
-> diagnostic rerank
-> selected separated tracks
-> speaker count estimation
```

候选说明：

- `libri2mix`：SpeechBrain `speechbrain/sepformer-libri2mix`，稳定主候选。
- `mossformer2`：ClearVoice `MossFormer2_SS_16K`，用于重叠说话更明显的候选。
- `resepformer`：SpeechBrain `speechbrain/resepformer-wsj02mix`，多人场景补充候选。
- `placeholder`：所有真实候选失败时的兜底轨道。

## 与增强模块的关系

不要把增强放在分离之前。当前设计是：

```text
混合音频: 归一化 / 简单处理 -> 分离
分离轨道: 单独增强 -> 单独 ASR
```

原因是分离模型要从混合波形里拆出多个声源。先对混合音频做神经网络增强，可能会改变弱说话人、重叠区域和频谱结构，反而影响分离。分离后每条轨道更接近单人语音，再做增强更符合 ASR 的输入需求。

## 输出轨道

分离模块返回的每条轨道会在后续增强阶段扩展字段：

```text
track_id
label
description
separated_audio_url       原始分离轨道
audio_url                 增强后的轨道，前端播放和 ASR 使用
track_enhancement_status
track_enhancement_method
asr_text
asr_status
transcript
```

如果逐轨增强失败，`audio_url` 会回退为 `separated_audio_url`，保证 ASR 和页面展示不断。

## Fast 模式

Fast 模式用于快速查看分离结果：

```text
raw audio
-> separation router / demo clean source
-> separated_tracks
```

Fast 模式不做逐轨增强、ASR、主题、摘要和对齐。完整汇报和 Web 展示应使用默认 `full` 模式。
