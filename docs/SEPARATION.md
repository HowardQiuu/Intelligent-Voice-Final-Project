# 语音分离模块

## 模块目标

语音分离模块负责在多人混合或重叠说话场景中生成可试听的说话人轨道。它不是 ASR 的替代品，而是给前端试听和会议质量诊断提供多轨结果。

## 最终保留路径

当前只保留最佳主路径和兜底路径：

```text
raw mixture audio
-> quality router
-> SpeechBrain SepFormer Libri2Mix
-> ClearVoice MossFormer2_SS_16K
-> FunASR diarization gated tracks
-> placeholder fallback
```

默认配置：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_INPUT_SOURCE=raw
SEPARATION_CANDIDATES=libri2mix,mossformer2,gated
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
SEPARATION_LIBRI2MIX_BONUS=1.5
SEPARATION_DIAGNOSTIC_RERANK=true
```

说明：

- `libri2mix`：最终保留的 SpeechBrain SepFormer 候选，适合轮流说话和部分重叠样本。
- `mossformer2`：最终保留的 ClearVoice 候选，适合部分高重叠场景。
- `gated`：根据说话人分段生成门控试听轨道，是兜底路径，不宣称为严格盲源分离。
- `placeholder`：所有候选失败时的最后兜底，保证页面不崩。

旧的 WSJ02Mix 候选、外部命令适配器、ESPnet 探究脚本和 oracle 对比逻辑已经从交付版默认流程中移除。

## 质量路由

质量路由会运行最终候选，并根据音频质量、轨道数量、轨道相关性、重叠诊断和模型偏置选择结果。当前最佳探究结论是：

```text
SEPARATION_CANDIDATES=libri2mix,mossformer2,gated
```

这个组合在本地 20 条中文两说话人合成基准上取得：

```text
平均 SI-SDRi = 22.078 dB
平均 STOI = 0.928
失败数 = 0
fallback 数 = 0
```

## 与 FunASR / VAD / 说话人分段的关系

FunASR、VAD 和说话人分段负责生成带时间戳和说话人标签的转写。分离模块使用这些信息作为兜底门控轨道来源，但真实盲源分离主要由 `libri2mix` 和 `mossformer2` 候选完成。

汇报时可以这样表述：

```text
ASR 主路径负责识别和说话人分段；
分离主路径负责输出可试听多轨；
当盲源分离不可用时，系统使用说话人分段门控轨道兜底。
```

## 输出内容

分离模块输出：

- `method`：最终选择的分离方法。
- `status`：成功、门控兜底或 placeholder 兜底状态。
- `track_count`：输出轨道数。
- `tracks`：每条轨道的试听 URL、标签和说明。
- `metrics`：质量路由评分和诊断信息。

## 失败兜底

以下情况会进入兜底路径：

- SpeechBrain 或 ClearVoice 未安装。
- 模型权重缺失。
- CUDA/CPU 推理失败。
- 输入音频过长或格式异常。
- 候选没有输出有效轨道。

优先兜底到 `gated`，最后兜底到 `placeholder`。
