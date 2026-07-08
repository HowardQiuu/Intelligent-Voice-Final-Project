# 轨道增强模块

## 模块位置

当前完整链路中，增强发生在分离之后，而不是分离之前：

```text
原始混合音频
-> 归一化 / 简单预处理
-> 语音分离
-> 每条分离轨道单独增强
-> 每条增强轨道单独 ASR
```

混合音频阶段只做归一化和必要格式处理，保证分离输入稳定；神经网络降噪和响度后处理作用于分离后的单轨。

## 实际处理函数

每条分离轨道会进入 `enhance_uploaded_audio()`：

```text
separated track wav
-> 读取时长
-> 判断是否跳过增强
-> 原始轨道质量分析
-> audibility pregain
-> DeepFilterNet / ClearVoice enhancement candidate
-> quality score
-> post loudness normalization
-> enhanced track wav
```

增强成功后：

```text
SeparatedTrack.separated_audio_url = 原始分离轨道
SeparatedTrack.audio_url = 增强后的分离轨道
```

前端试听和 ASR 都读取增强后的 `audio_url`。

## 质量路由候选

默认配置：

```text
ENHANCEMENT_CANDIDATES=deepfilternet,clearvoice
DEEPFILTERNET_BACKEND=cli
CLEARVOICE_ENHANCE_MODEL=MossFormer2_SE_48K
ENHANCEMENT_PREGAIN_ENABLED=true
ENHANCEMENT_TARGET_LUFS=-18
```

候选说明：

- `DeepFilterNet`：神经网络降噪候选，适合降低背景噪声。
- `ClearVoice MossFormer2_SE_48K`：语音增强候选。这里的 `SE` 是 speech enhancement，不是分离模型的 `SS`。

`MossFormer2_SE_48K` 和 `MossFormer2_SS_16K` 的区别：

```text
MossFormer2_SE_48K: speech enhancement，输入一条语音，输出增强后一条语音
MossFormer2_SS_16K: speech separation，输入混合语音，输出多条说话人轨道
```

## 响度处理

增强前可能执行 audibility pregain，处理过轻的轨道；增强后执行响度归一化和 limiter：

```text
highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11,alimiter=limit=0.95
```

目标是让不同模型输出的单轨音频更稳定，便于试听和 ASR。

## 失败兜底

如果某条轨道增强失败：

```text
track_enhancement_status = fallback
audio_url = separated_audio_url
```

也就是说，系统会继续用原始分离轨道做 ASR，不会因为某条轨道增强失败中断完整流程。

## 前端展示

前端顶部的第二个混合音频播放器现在表示“预处理后混合音频”，不是“增强后混合音频”。真正的增强音频在每条分离轨道卡片中展示。

增强可视化模块已经从当前展示流程中移除。
