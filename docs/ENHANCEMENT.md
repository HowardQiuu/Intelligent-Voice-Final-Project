# 语音增强模块

## 模块位置

语音增强只在完整处理模式中运行：

```text
上传音频
-> 标准化为单声道 WAV
-> 语音增强
-> ASR / 摘要
```

当前完整链路中，ASR 使用增强后的音频；分离模块默认使用原始混合音频作为输入，除非 `SEPARATION_INPUT_SOURCE=enhanced`。因此增强的主要价值是提升转写可懂度、试听质量和摘要输入质量，而不是强制改变分离输入。

## 实际处理链路

上传音频进入 `enhance_uploaded_audio()` 后，实际步骤是：

```text
normalized wav
-> 读取时长
-> 判断是否跳过增强
-> 原始音频质量分析
-> audibility pregain
-> 增强质量路由
-> 后置响度归一化
-> 质量评分和指标返回
```

跳过增强的条件包括 `DEEPFILTERNET_BACKEND=off/disabled/placeholder/skip/none`，或设置了 `ENHANCEMENT_SKIP_SECONDS` 且音频超过该时长。跳过时会直接返回输入音频 URL，并在指标里标记 skipped。

## 汇报讲解：每一步具体做什么

完整增强链路可以按下面顺序讲：

```text
标准化 WAV
-> audibility pregain
-> DeepFilterNet / ClearVoice 候选增强
-> quality score 选择最优候选
-> post loudness normalization
-> 输出增强音频和指标
```

### 1. audibility pregain

`audibility pregain` 是模型前的可听度预处理，目标是处理“原始录音太轻”的情况。它不是降噪模型，而是把过低的语音先抬到更适合模型和 ASR 的响度范围，避免后续模型把低能量语音当成背景。

系统先分析原始音频质量：

- `rms_dbfs`：整体均方根能量，表示平均响度。
- `peak_dbfs`：峰值电平，表示最大瞬时幅度。
- `silent_ratio`：按 16000 个采样点为一块统计，块 RMS 小于等于 `0.006` 的比例。
- `clipping_ratio`：绝对幅度大于等于 `0.98` 的采样比例，用来判断是否削波。
- `spectral_centroid_hz`：频谱质心，用来粗略判断频率能量是否落在语音常见范围。

触发 pregain 的条件：

```text
rms_dbfs <= -34
或
peak_dbfs <= -24 且 rms_dbfs <= -28
```

推荐增益：

```text
gain_db = min(ENHANCEMENT_MAX_GAIN_DB, ENHANCEMENT_TARGET_LUFS - rms_dbfs)
默认最大增益 = 24 dB
默认目标响度 = -18
```

实际 FFmpeg 滤波链：

```text
highpass=f=80,
volume={gain_db}dB,
acompressor=threshold=-28dB:ratio=3:attack=5:release=80,
loudnorm=I=-18:TP=-2:LRA=8,
alimiter=limit=0.95
```

可以这样解释：

- `highpass=f=80`：去掉 80 Hz 以下低频轰鸣和直流漂移。
- `volume={gain_db}dB`：把过轻的语音先抬升。
- `acompressor`：压缩动态范围，避免某些突出的音节被放得过响。
- `loudnorm`：按目标响度做归一化，同时限制 True Peak 到 `-2 dB`。
- `alimiter=limit=0.95`：最后限幅，避免增益后出现削波。

### 2. DeepFilterNet candidate

DeepFilterNet 是一个神经网络语音增强候选，主要目标是抑制平稳或非平稳背景噪声，保留人声主体。项目支持两种运行方式：

```text
DEEPFILTERNET_BACKEND=cli     调用官方命令行
DEEPFILTERNET_BACKEND=source  直接加载 df.enhance API，并缓存模型
```

在汇报中可以说：DeepFilterNet 负责做“神经网络降噪”，输入是 pregain 后的单声道 WAV，输出是去噪后的 WAV。它不会做说话人分离，也不会改变会议文本语义，只是尽量降低背景噪声、改善语音清晰度。

如果音频超过 `ENHANCEMENT_MAX_SECONDS=300`，DeepFilterNet 会分块执行：

```text
按 60s 切块 -> 每块单独增强 -> 拼接回完整音频
```

这样做是为了避免长会议一次性送入模型导致显存或内存不稳定。

### 3. ClearVoice enhancement candidate

ClearVoice enhancement 是另一个增强候选，当前默认模型是：

```text
CLEARVOICE_ENHANCE_MODEL=MossFormer2_SE_48K
```

这里的 `SE` 表示 speech enhancement。它同样做语音增强/降噪，不是分离模块里的 `MossFormer2_SS_16K`。汇报时要区分：

```text
MossFormer2_SE_48K: speech enhancement，用于增强
MossFormer2_SS_16K: speech separation，用于分离
```

ClearVoice 候选会生成自己的增强音频，然后和 DeepFilterNet 的结果一起进入质量评分。这样系统不是固定相信某一个模型，而是让两个增强结果通过统一指标竞争。

### 4. quality score

`quality score` 是增强候选选择分数。它不是主观听感分，而是由音频统计指标计算出来。

基础音频质量分 `score_audio_quality(candidate)` 从 55 分开始：

```text
RMS 过低:
  rms_dbfs < -36      -25
  -36 <= rms_dbfs < -30  -12
RMS 过高:
  rms_dbfs > -10      -10
RMS 合理:
  其他范围             +12

峰值过低:
  peak_dbfs < -24     -12

静音比例:
  silent_ratio > 0.65 -18
  silent_ratio < 0.45 +8

削波比例:
  clipping_ratio > 0.01  -25
  clipping_ratio > 0.002 -10

频谱质心:
  300 Hz <= centroid <= 4200 Hz  +7
```

基础分会限制在 `0-100`。然后增强候选分数继续加入“相对原音频是否变好”的因素：

```text
candidate_score =
  score_audio_quality(candidate)
  + clamp(candidate.rms_dbfs - original.rms_dbfs, -12, 18)
  - 15  如果 candidate 的 clipping_ratio 比 original 多 0.002 以上
  + 最多 8 分  如果 candidate 的 silent_ratio 比 original 更低
```

最终分数限制在 `0-120`。因此它偏好：

- 平均响度更合适的结果。
- 静音/低能量比例更低的结果。
- 没有新增削波的结果。
- 频谱能量落在语音常见范围的结果。

汇报时可以说：质量路由不是简单比较“声音更大”，因为如果声音变大但发生削波，系统会扣分；如果候选把有效语音压没了、静音比例升高，也不会被选中。

### 5. post loudness normalization

候选模型输出后，系统还会做一次后置响度归一化：

```text
highpass=f=80,
loudnorm=I=-18:TP=-2:LRA=11,
alimiter=limit=0.95
```

这一步解决的是“不同增强模型输出响度不一致”的问题。模型 A 可能输出很轻，模型 B 可能输出很响，如果直接给 ASR 或前端试听，会影响稳定性和主观比较。因此后处理把候选统一到接近 `-18 LUFS` 的目标响度，并用 limiter 避免过载。

它和 pregain 的区别：

```text
audibility pregain: 模型前，把太轻的输入抬起来
post loudness normalization: 模型后，把不同候选输出拉到统一响度
```

## 质量路由候选

默认配置：

```text
ENHANCEMENT_CANDIDATES=deepfilternet,clearvoice
DEEPFILTERNET_BACKEND=cli
CLEARVOICE_ENHANCE_MODEL=MossFormer2_SE_48K
ENHANCEMENT_PREGAIN_ENABLED=true
ENHANCEMENT_TARGET_LUFS=-18
```

质量路由逐个尝试候选：

```text
DeepFilterNet
ClearVoice MossFormer2_SE_48K
```

每个候选会先生成去噪音频，再执行后置响度归一化，然后通过 `audio_quality_service.score_enhancement_candidate()` 计算分数。系统选择分数最高的候选，并把每个候选的状态写入 `signal_metrics`。

如果 `QUALITY_ROUTER_ENABLED=false`，增强模块只跑 DeepFilterNet 路径，不再比较多个候选。

## DeepFilterNet 路径

DeepFilterNet 支持两种后端：

```text
DEEPFILTERNET_BACKEND=cli
DEEPFILTERNET_BACKEND=source
```

`cli` 后端通过官方命令行处理音频；`source` 后端直接加载 `df.enhance`，并缓存模型对象以减少重复加载成本。模型目录可通过 `DEEPFILTERNET_SOURCE_DIR` 和 `DEEPFILTERNET_MODEL_DIR` 指定。

长音频会进入分块增强：

```text
ENHANCEMENT_MAX_SECONDS=300
ENHANCEMENT_CHUNK_SECONDS=60
ENHANCEMENT_MAX_CHUNKS=120
ENHANCEMENT_WORKERS=2
```

当音频超过 `ENHANCEMENT_MAX_SECONDS` 时，系统把音频切成多个 WAV chunk，分别增强后再拼接。CLI 后端支持按 `ENHANCEMENT_WORKERS` 并行处理分块。

## ClearVoice 路径

ClearVoice 候选通过 `clearvoice.ClearVoice` 加载：

```text
CLEARVOICE_ENHANCE_MODEL=MossFormer2_SE_48K
CLEARVOICE_ENHANCE_USE_CUDA=auto
```

运行结果会复制到 uploads 目录，并作为增强候选参与质量评分。如果 ClearVoice 依赖、模型或推理失败，该候选只会被标记为 skipped，不会中断完整流程。

## 响度处理与可视化

增强前会执行 audibility pregain，避免原始音频过轻导致模型和 ASR 难以处理。增强后会执行：

```text
highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11,alimiter=limit=0.95
```

完整流程随后调用 `generate_enhancement_visual()`，生成增强前后波形包络、噪声底、能量和清晰度代理指标。前端用 `enhancement_visual_url` 展示这张图。

## 输出字段

增强模块返回：

- `original_audio_url`：标准化后的原始音频 URL。
- `enhanced_audio_url`：被选中的增强音频 URL；跳过或兜底时可能等于原始音频。
- `method`：增强方法、候选分数和响度归一化说明。
- `metrics`：候选状态、选中分数、响度处理、质量指标和分块并行状态。

这些字段会合并进最终 `ProcessResult.signal_metrics`。

## 失败兜底

以下情况不会让完整流程崩溃：

- DeepFilterNet 未安装、命令不可用或模型目录缺失。
- ClearVoice 未安装、模型加载失败或推理失败。
- FFmpeg 不可用导致分块或后处理失败。
- 输入音频过长、格式异常或所有候选失败。

候选全部失败时，系统返回 audibility pregain 后的音频，并标记：

```text
Audibility pregain fallback (no enhancement candidate succeeded)
```

如果 `process_audio_path()` 捕获到增强阶段 `RuntimeError`，会进一步退回标准化音频，保证 ASR、分离和摘要仍有输入。
