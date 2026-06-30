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
-> SpeechBrain ReSepFormer WSJ02Mix
-> placeholder fallback
```

默认配置：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_INPUT_SOURCE=raw
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
SEPARATION_LIBRI2MIX_BONUS=1.5
SEPARATION_DIAGNOSTIC_RERANK=true
SEPARATION_RECURSIVE_EXPANSION=true
SEPARATION_RECURSIVE_MODE=direct_split
```

说明：

- `libri2mix`：最终保留的 SpeechBrain SepFormer 候选，适合轮流说话和部分重叠样本。
- `mossformer2`：最终保留的 ClearVoice 候选，适合部分高重叠场景。
- `resepformer`：本地可用的 SpeechBrain ReSepFormer 候选；多人目标数 >=3 时会获得额外路由偏置。
- `placeholder`：所有候选失败时的最后兜底，保证页面不崩。

旧的 WSJ02Mix 候选、外部命令适配器、ESPnet 探究脚本和 oracle 对比逻辑已经从交付版默认流程中移除。

## 质量路由

质量路由会运行最终候选，并根据音频质量、轨道数量、轨道相关性、重叠诊断和模型偏置选择结果。当前最佳探究结论是：

```text
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
```

这个组合在本地 20 条中文两说话人合成基准上取得：

```text
平均 SI-SDRi = 22.078 dB
平均 STOI = 0.928
失败数 = 0
fallback 数 = 0
```

## 与 FunASR / VAD / 说话人分段的关系

FunASR、VAD 和说话人分段负责生成带时间戳和说话人标签的转写。分离模块可用这些信息估计目标说话人数并辅助评分，但不会用它们生成门控假分离轨道；真实盲源分离由 `libri2mix`、`mossformer2` 和 `resepformer` 候选完成。

汇报时可以这样表述：

```text
ASR 主路径负责识别和说话人分段；
分离主路径负责输出可试听多轨；
当盲源分离不可用时，系统只返回 placeholder 兜底，不使用说话人分段门控轨道冒充分离。
```

## 输出内容

分离模块输出：

- `method`：最终选择的分离方法。
- `status`：成功或 placeholder 兜底状态。
- `track_count`：输出轨道数。
- `tracks`：每条轨道的试听 URL、标签和说明。
- `metrics`：质量路由评分和诊断信息。
- `separation_alignment`：ASR 时间段与已生成分离轨道的能量对齐结果，不使用 TextGrid。
- `separation_evaluation`：TextGrid 事后验证结果，只用于评估分离效果。

## ASR 分段与分离轨道对齐

分离模型存在输出排列不确定性，因此 `track 1` 不能天然等于 `说话人 A`。项目新增了独立对齐层：

```text
ASR transcript timestamps
+ separated track audio
-> segment-to-track energy alignment
```

这一步只读取当前 pipeline 已经产生的 ASR 时间戳和分离轨道音频。它不会读取 TextGrid，也不会用 TextGrid 改变分离后端选择、轨道标签或 ASR 分段。

默认设置：

```text
ASR_SEPARATION_ALIGNMENT_ACTIVE_RATIO=0.35
ASR_SEPARATION_ALIGNMENT_ACTIVE_FLOOR=0.00001
```

对齐层会在每个 ASR 片段的时间范围内计算各分离轨道的 RMS 能量，并写回：

```text
primary_track_id
primary_track_label
separation_tracks
```

如果同一个时间戳上多个分离轨道都有明显能量，`separation_tracks` 会保留多个轨道，表示该片段可能存在重叠说话。

## TextGrid 事后验证

TextGrid 只作为验证真值使用。若额外放入 AliMeeting far/near 原始包，系统可按文件名查找对应 TextGrid，并在分离结束后把分离结果与 TextGrid 参考进行比较。TextGrid 不进入分离、质量路由、轨道重命名或 ASR 对齐。当前精简交付版不再保留原始 AliMeeting 包，只保留 `data/near_mix_dataset_v1` 作为 close-talk near-mix 验证数据集。

默认设置：

```text
SEPARATION_EVAL_TRANSCRIBE_TRACKS=auto
SEPARATION_EVAL_MAX_TRANSCRIBE_SECONDS=120
```

系统会在找到 TextGrid 且音频不超过 120 秒时，对每条分离轨道再跑一次 ASR，并把轨道文本与 TextGrid 中各参考说话人的文本做字符级相似度匹配，从而判断：

```text
track_1 更接近哪个参考说话人
track_2 更接近哪个参考说话人
TextGrid 中真实重叠区域是否被保留下来
```

如果处理的是完整长会议，`auto` 默认不会额外转写每条分离轨道，以免耗时过高。需要强制验证时可设为 `SEPARATION_EVAL_TRANSCRIBE_TRACKS=true`；如果未开启轨道转写，评测层仍会给出参考说话人数和重叠比例，但不会假称已经完成文本验证。

## 失败兜底

以下情况会进入兜底路径：

- SpeechBrain 或 ClearVoice 未安装。
- 模型权重缺失。
- CUDA/CPU 推理失败。
- 输入音频过长或格式异常。
- 候选没有输出有效轨道。

只兜底到 `placeholder`，不再把说话人分段门控轨道作为分离结果。
