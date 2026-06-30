# 语音增强模块

## 模块目标

语音增强模块负责把上传的会议音频整理成更适合后续处理的单声道语音，并尽量提升响度、清晰度和可懂度。它位于完整流程前段：

```text
上传音频 -> 标准化 -> 语音增强 -> 分离 / ASR / 摘要
```

增强模块不改变会议语义，只改善后续模块可用的音频质量。

## 最终保留路径

当前保留的增强路径是质量路由：

```text
DeepFilterNet candidate
ClearVoice enhancement candidate
audibility pregain fallback
```

默认配置：

```text
ENHANCEMENT_CANDIDATES=deepfilternet,clearvoice
DEEPFILTERNET_BACKEND=cli
CLEARVOICE_ENHANCE_MODEL=MossFormer2_SE_48K
ENHANCEMENT_PREGAIN_ENABLED=true
ENHANCEMENT_TARGET_LUFS=-18
```

质量路由会选择评分更好的增强结果。若真实模型不可用，会进入响度预增益兜底，保证后续 ASR、分离和摘要不会中断。

## 长音频处理

长会议音频按固定窗口分块处理：

```text
ENHANCEMENT_MAX_SECONDS=300
ENHANCEMENT_CHUNK_SECONDS=60
ENHANCEMENT_MAX_CHUNKS=120
ENHANCEMENT_WORKERS=2
```

分块策略用于避免一次性把长音频送入大模型导致显存或内存不稳定。

## 输出内容

增强模块输出：

- 增强后的音频 URL。
- 原始音频与增强音频的响度/峰值指标。
- 增强前后可视化图。
- 是否使用真实增强模型或兜底策略。

前端在试听区域展示增强前后音频和可视化图，方便汇报时说明“增强不是只靠听感判断”。

## 失败兜底

以下情况会自动兜底：

- DeepFilterNet 未安装或命令不可用。
- ClearVoice 未安装或模型加载失败。
- 输入音频过长、格式异常或模型推理失败。
- 质量路由候选全部失败。

兜底后仍返回可处理音频，并在指标中记录 `fallback` 状态。
