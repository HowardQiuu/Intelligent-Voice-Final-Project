# 语音分离模块

## 模块位置

语音分离模块负责把混合会议音频拆成可试听的多条说话人轨道。它服务于前端试听、重叠说话展示、轨道质量诊断和后续 ASR 片段对齐。

当前项目有两种调用方式：

```text
fast 模式: 上传音频 -> 分离质量路由 -> separated_tracks
full 模式: 完整链路中 ASR 后运行分离，再做 ASR 片段到分离轨道的对齐
```

Fast 模式只用于快速评估分离效果，不运行增强、ASR、摘要和对齐。

## 输入选择

完整链路会先做音频标准化和增强，但分离模块默认使用原始混合音频：

```text
SEPARATION_INPUT_SOURCE=raw
```

原因是 SpeechBrain 和 ClearVoice 的盲源分离模型更接近原始混合音频的训练分布。可选值包括：

```text
raw        使用原始上传音频，当前默认
normalized 使用标准化 WAV
enhanced   使用增强后音频
```

`process_audio_path()` 会把实际选择写入 `signal_metrics["separation_input_source"]`。

## 质量路由主路径

默认配置：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
SEPARATION_LIBRI2MIX_BONUS=1.5
SEPARATION_RESEPFORMER_MULTISPEAKER_BONUS=4.0
SEPARATION_DIAGNOSTIC_RERANK=true
```

实际链路：

```text
selected source audio
-> run libri2mix candidate
-> run mossformer2 candidate
-> run resepformer candidate
-> score candidates
-> diagnostic rerank
-> optional selected SpeechBrain refinement
-> selected tracks
-> speaker count estimation
```

候选说明：

- `libri2mix`：SpeechBrain `speechbrain/sepformer-libri2mix`，当前有默认分数加成，作为稳定主候选。
- `mossformer2`：ClearVoice `MossFormer2_SS_16K`，在重叠诊断明显时可获得额外加成。
- `resepformer`：SpeechBrain `speechbrain/resepformer-wsj02mix`，当预计说话人数不少于 3 时会获得多人场景加成。
- `placeholder`：所有真实候选失败时的兜底轨道。

旧的外部命令适配、ESPnet 探索脚本和 oracle 对比不属于当前交付主链路。

## 汇报讲解：分离每一步具体做什么

完整分离链路可以按下面顺序讲：

```text
选择分离输入
-> 运行多个盲源分离候选
-> 必要时分块或递归扩展
-> 对候选结果计算 quality score
-> 根据诊断指标重排
-> 对选中的 SpeechBrain 结果做后处理
-> 估计说话人数
-> 与 ASR 片段做能量对齐
```

### 1. 为什么分离默认用 raw

当前默认：

```text
SEPARATION_INPUT_SOURCE=raw
```

这是因为分离模型的目标是从“混合波形”中拆出多个声源。增强模型会改变频谱、相位和能量分布，有时会抹掉弱说话人或改变重叠区域，对盲源分离反而不利。因此当前完整链路是：

```text
增强音频 -> 给 ASR 和摘要使用
原始混合音频 -> 给分离模型使用
```

如果汇报时被问到“为什么不先增强再分离”，可以回答：系统保留了 `SEPARATION_INPUT_SOURCE=enhanced` 的选项，但默认选择 raw，是为了让分离输入更接近模型训练时的混合语音分布，减少增强对多说话人结构的破坏。

### 2. Libri2Mix SepFormer candidate

`libri2mix` 调用 SpeechBrain 的：

```text
speechbrain/sepformer-libri2mix
```

SepFormer 是基于 Transformer 的时域语音分离模型。它直接输入混合语音波形，输出多条估计的说话人波形。项目里它是稳定主候选，所以有默认加分：

```text
SEPARATION_LIBRI2MIX_BONUS=1.5
```

汇报口径：Libri2Mix SepFormer 负责主路径盲源分离，特点是稳定、可复现，适合作为默认候选。

### 3. ClearVoice MossFormer2 separation candidate

`mossformer2` 调用 ClearVoice 的：

```text
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
```

这里的 `SS` 表示 speech separation。它和增强模块的 `MossFormer2_SE_48K` 不是同一个任务：

```text
MossFormer2_SE_48K: 语音增强，输出一条更干净的语音
MossFormer2_SS_16K: 语音分离，输出多条说话人轨道
```

MossFormer2 在重叠说话更明显时可能更合适，因此代码里有两类加分：

```text
基础模型加分: +8
transcript 重叠比例超过阈值时: 额外 overlap boost
诊断重排满足条件时: 额外 diagnostic bonus
```

### 4. ReSepFormer candidate

`resepformer` 调用：

```text
speechbrain/resepformer-wsj02mix
```

它作为额外 SpeechBrain 候选参与质量路由。当系统有期望说话人数，且预计至少 3 个说话人时，ReSepFormer 会获得多人场景偏置：

```text
SEPARATION_RESEPFORMER_MULTISPEAKER_BONUS=4.0
```

汇报口径：ReSepFormer 是补充分离候选，用来提高多人或复杂场景下的备选覆盖。

## 候选评分依据

`separate_with_quality_router()` 会对每个候选生成的结果打分。评分主要参考：

- 候选是否成功输出有效轨道。
- 输出轨道数是否接近期望说话人数。
- 轨道之间的相关性和能量平衡。
- 分离轨道相加后与混合音频的相似程度。
- transcript 中估算的重叠说话比例。
- `libri2mix`、`mossformer2`、`resepformer` 的模型偏置加分。

诊断重排开启时，系统还会比较候选的混合重建相关性、轨道重叠比例和 SpeechBrain 轨道相关性。如果 MossFormer2 更符合高重叠场景，会给它附加诊断加分。

### quality score 具体怎么算

分离候选的初始分数来自 `_score_separation_result()`：

```text
score = 25 + min(30, track_count * 10)
```

也就是先给一个基础分，再根据输出轨道数加分，最多加 30 分。随后把每条轨道当作普通音频计算 `score_audio_quality()`，取平均后乘以 `0.35` 加到候选分：

```text
score += average(track_audio_quality_score) * 0.35
```

后续加减项：

```text
placeholder 或 fallback: -30
method 包含 mossformer2: +8
method 包含 speechbrain: +4
method 包含 libri2mix: +SEPARATION_LIBRI2MIX_BONUS
resepformer 且 expected_speakers == 3: +SEPARATION_RESEPFORMER_MULTISPEAKER_BONUS
输出轨道数少于 expected_speakers: 按缺失比例最多扣 45
mossformer2 且 transcript 重叠比例超过阈值: overlap boost
```

最终分数限制在 `0-100`。因此它不是只看模型名字，而是综合考虑：

- 是否真的输出了轨道。
- 轨道数是否足够。
- 每条轨道自身的响度、静音、削波和频谱质量是否合理。
- 当前文本时间轴是否显示明显重叠说话。
- 某些模型在特定场景下的经验偏置。

### transcript overlap ratio

完整链路中 ASR 先运行，所以分离可以拿到 transcript 时间戳。系统把不同说话人的时间段合并成事件线，计算“同一时间有两个及以上说话人活动”的时长比例：

```text
overlap_ratio = overlap_seconds / total_meeting_span
```

这个比例不直接生成分离轨道，只辅助质量路由。如果重叠比例较高，MossFormer2 会获得额外加分，因为它在部分高重叠场景下更可能保留重叠说话结构。

### diagnostic rerank 具体看什么

`SEPARATION_DIAGNOSTIC_RERANK=true` 时，系统会读取候选前两条轨道和原始混合音频，计算诊断指标：

- `sum_mix_correlation`：两条分离轨道相加后，与原混合音频的相关性。越高表示重建混合音频越接近。
- `sum_mix_residual_ratio`：轨道相加与原混合音频之间的残差能量比例。越低表示遗漏越少。
- `inter_track_correlation`：两条分离轨道之间的相关性。过高说明两条轨道可能太像，分离不彻底。
- `track_energy_balance`：两条轨道能量较小值 / 较大值。越接近 1 表示能量更平衡。
- `track_overlap_ratio`：按 25 ms 帧、10 ms hop 计算，两条轨道同时活跃的比例。

MossFormer2 的诊断加分条件：

```text
sum_mix_correlation >= MOSSFORMER2_DIAGNOSTIC_MIXCORR_THRESHOLD
track_overlap_ratio >= MOSSFORMER2_DIAGNOSTIC_TRACK_OVERLAP_THRESHOLD
SpeechBrain inter_track_correlation <= MOSSFORMER2_DIAGNOSTIC_SPEECHBRAIN_CORR_MAX
```

默认阈值：

```text
MOSSFORMER2_DIAGNOSTIC_MIXCORR_THRESHOLD=0.986
MOSSFORMER2_DIAGNOSTIC_TRACK_OVERLAP_THRESHOLD=0.34
MOSSFORMER2_DIAGNOSTIC_SPEECHBRAIN_CORR_MAX=-0.0015
MOSSFORMER2_DIAGNOSTIC_BONUS=8.0
```

汇报口径：诊断重排的作用是避免只靠模型固定优先级，而是检查候选输出是否真的能重建混合音频、轨道之间是否足够独立、是否保留了重叠说话。

## 长音频与递归扩展

SpeechBrain 分离支持分块处理，避免长音频一次性推理失败：

```text
SEPARATION_MAX_SECONDS=60
SEPARATION_CHUNK_SECONDS=60
SEPARATION_MAX_CHUNKS=120
SEPARATION_CHUNK_ALIGNMENT_SIMILARITY_FLOOR=0.15
```

分块后系统会按轨道特征把不同 chunk 的输出轨道对齐，再拼回完整轨道。

当预计说话人数多于候选直接输出轨道数时，可以启用递归盲源扩展：

```text
SEPARATION_RECURSIVE_EXPANSION=true
SEPARATION_RECURSIVE_MODE=direct_split
SEPARATION_RECURSIVE_PARENT_SELECTION=complexity
SEPARATION_RECURSIVE_MAX_TRACKS=6
SEPARATION_RECURSIVE_MAX_STEPS=4
```

递归扩展会选择复杂度较高的轨道继续拆分，直到达到目标轨道数或不再有可靠拆分。

### 自动递归扩展才是分离过程里的“轨道数估计”

如果没有配置 `expected_speakers`，代码不会先运行 `speaker_count_estimation_service` 来决定人数，而是进入自动递归扩展：

```text
基础分离结果
-> 判断哪条轨道还像混合轨道
-> 尝试把这条轨道再次分成 2 条
-> 评估这次拆分是否可信
-> 可信就接受，轨道数 +1
-> 继续下一轮，直到没有可信拆分或达到上限
```

所以你汇报里如果说“系统自动估计应该分出几条轨道”，更准确指的是这一层：

```text
auto recursive blind expansion
```

它不是后面的 `speaker_count_estimation` 诊断，而是在分离过程中通过“能不能继续可信拆分”来自适应决定最终轨道数。

默认上限来自：

```text
SEPARATION_AUTO_RECURSIVE_MAX_TRACKS
SEPARATION_RECURSIVE_MAX_TRACKS=6
SEPARATION_RECURSIVE_MAX_STEPS=4
SEPARATION_AUTO_RECURSIVE_MAX_DEPTH
```

也就是说，自动扩展不会无限拆。它受到最大轨道数、最大递归步数和单条轨道最大递归深度限制。

### 自动递归扩展如何判断“还要不要拆”

自动递归扩展每一轮先选择一个父轨道。默认策略是：

```text
SEPARATION_RECURSIVE_PARENT_SELECTION=complexity
```

也就是优先挑复杂度最高的轨道。复杂度高通常表示这条轨道里可能还混着多个说话人。如果改成 `energy`，则优先挑 RMS 能量最高的轨道。

选中父轨道后，系统会把父轨道再次送进当前分离模型，得到两个 child tracks。然后 `_score_recursive_split()` 判断这次拆分是否可信，核心指标有四个：

```text
energy_balance
active_min
parent_correlation
child_correlation
```

含义：

- `energy_balance`：两个子轨道能量是否均衡，计算为较小能量 / 较大能量。太低说明拆出来一条几乎没声音，像假拆分。
- `active_min`：两个子轨道中较低的活跃比例。太低说明至少一条子轨道没有足够有效语音。
- `parent_correlation`：两个子轨道相加后和父轨道的相关性。越高表示拆分后还能重建原父轨道。
- `child_correlation`：两个子轨道之间的相关性。越高说明两条子轨道太像，可能没有真正分开。

综合质量分：

```text
quality =
  0.45 * energy_balance
  + 0.30 * active_min
  + 0.15 * parent_correlation
  + 0.10 * (1 - child_correlation)
```

只有同时满足下面条件，这次拆分才会被接受：

```text
quality >= SEPARATION_AUTO_RECURSIVE_MIN_QUALITY
energy_balance >= SEPARATION_AUTO_RECURSIVE_MIN_ENERGY_BALANCE
active_min >= SEPARATION_AUTO_RECURSIVE_MIN_ACTIVE_RATIO
child_correlation <= SEPARATION_AUTO_RECURSIVE_MAX_CHILD_CORRELATION
```

如果接受：

```text
父轨道 -> 两条子轨道
总轨道数 +1
继续下一轮
```

如果拒绝：

```text
保留当前轨道数
尝试下一个候选父轨道
如果没有任何可信拆分，就停止扩展
```

因此，当前输出 4 条轨道时，通常可以解释为：系统不是预先知道有 4 个说话人，而是在递归过程中连续找到了可信的可拆分轨道，接受了若干次拆分，最终得到 4 条分离轨道。

### 分块分离具体做什么

当音频超过 `SEPARATION_MAX_SECONDS=60` 时，SpeechBrain 候选会分块执行。每个 chunk 都会输出一组轨道，但不同 chunk 的轨道顺序可能会交换，所以系统会提取轨道特征，计算 chunk 间相似度，把同一个说话人的轨道对齐到同一条长轨道，再拼接。

这一层解决两个问题：

- 长音频不会一次性占满显存或内存。
- 分块后轨道顺序不稳定的问题会通过相似度对齐缓解。

### 分块后不同 chunk 的说话人如何匹配

这是分离汇报里的重点。SpeechBrain 对每个 chunk 都可能输出：

```text
chunk 1: track_1 = A, track_2 = B
chunk 2: track_1 = B, track_2 = A
```

如果直接按轨道编号拼接，就会把不同说话人的声音拼到同一条轨道里。因此项目做了 chunk 间轨道匹配，核心方法是“音频签名相似度匹配”，不依赖文本，也不依赖 TextGrid。

整体流程：

```text
1. 每个 chunk 独立分离，得到若干 chunk track
2. 给每条 chunk track 提取 spectral signature
3. 第一个 chunk 建立 speaker prototype
4. 后续 chunk 枚举所有轨道排列
5. 选择与 prototype 平均相似度最高的排列
6. 按匹配后的 speaker_index 分组
7. 每个 speaker_index 的 chunk track 拼接成完整长轨道
```

#### 第一个 chunk 如何建立原型

第一个 chunk 没有历史参考，所以按模型输出顺序建立原型：

```text
speaker_1 prototype = chunk_1 track_1 signature
speaker_2 prototype = chunk_1 track_2 signature
```

这些 prototype 代表“当前认为的说话人轨道特征”。后续 chunk 都要和这些 prototype 对齐。

#### 每条轨道的 spectral signature 怎么提取

对每条分离轨道，系统会提取一个向量签名：

```text
track wav
-> 转单声道
-> 按 64 ms 左右分帧
-> hop = frame / 2
-> 计算每帧 RMS
-> 只保留活跃帧
-> 对活跃帧做 Hann window + FFT
-> 取 log spectrum
-> 压缩成 24 个频带均值
-> 加上 RMS、过零率、谱质心等统计量
-> log1p 压缩
-> L2 归一化为单位向量
```

关键细节：

- 帧长：`max(256, sample_rate * 0.064)`。
- hop：`max(128, frame / 2)`。
- 活跃帧阈值：`max(percentile_75_rms * 0.35, 1e-5)`。
- 频谱：对活跃帧做 `rfft`，取幅度后 `log1p`。
- 频带：把频谱均分成 24 个 band，取每个 band 的平均值。
- 额外特征：RMS 均值、RMS 标准差、过零率均值、过零率标准差、谱质心均值、谱质心标准差。
- 最后做 L2 归一化，因此后面可以用点积当余弦相似度。

这样做的直觉是：同一个说话人的音色、频谱分布、能量变化、过零率和谱质心在相邻 chunk 中通常更相似；不同说话人则会有差异。

#### 后续 chunk 如何找最优排列

假设每个 chunk 输出 2 条轨道，后续 chunk 有两种可能排列：

```text
排列 1: speaker_1 <- chunk track_1, speaker_2 <- chunk track_2
排列 2: speaker_1 <- chunk track_2, speaker_2 <- chunk track_1
```

如果输出 3 条轨道，就枚举 3! 种排列。系统会对每个排列计算：

```text
similarity = dot(prototype_signature, current_track_signature)
order_score = 当前排列下所有 speaker 的 similarity 平均值
```

然后选择 `order_score` 最大的排列。由于签名已经归一化，点积就是余弦相似度，范围大致在 `-1` 到 `1`。

如果最佳平均相似度低于：

```text
SEPARATION_CHUNK_ALIGNMENT_SIMILARITY_FLOOR=0.15
```

系统认为当前 chunk 的签名不可靠，就退回模型原始顺序，避免错误匹配造成更坏的拼接。

#### prototype 如何更新

匹配成功后，prototype 不会被当前 chunk 完全覆盖，而是做指数平滑：

```text
new_prototype = previous_prototype * 0.8 + current_signature * 0.2
```

这样做的意义是：第一个 chunk 的说话人特征作为主参考，但后续 chunk 也可以逐步修正原型，适应同一说话人在不同时间段音量、语速、频谱状态的变化。

#### 最终如何拼接

匹配后，系统按 `speaker_index` 把各 chunk 的轨道放入同一组：

```text
speaker_1: chunk1_trackA + chunk2_trackA + chunk3_trackA -> speaker_1_chunked.wav
speaker_2: chunk1_trackB + chunk2_trackB + chunk3_trackB -> speaker_2_chunked.wav
```

拼接后输出的指标包括：

```text
chunk_track_alignment=spectral_signature_matching
chunk_track_alignment_mean_similarity
chunk_track_alignment_min_similarity
```

汇报时可以总结为：分块分离后，项目不是简单按 `track 1/track 2` 拼接，而是为每个轨道提取频谱签名，通过全排列搜索找到和历史说话人原型最相似的排列，再拼接成长轨道。

### 递归扩展具体做什么

普通两人分离模型通常输出 2 条轨道。如果预计说话人更多，系统可以把“仍然很复杂的轨道”再次送入分离模型：

```text
混合音频 -> 轨道 A / 轨道 B
轨道 A 仍像混合音频 -> 再分成 A1 / A2
```

选择父轨道时默认使用 `complexity` 策略，也就是优先选择频谱/能量变化更复杂、更像混合了多人声音的轨道。递归扩展不使用参考真值，也不读取 TextGrid。

## 分离后的说话人数诊断

这里的 `speaker_count_estimation_service` 不是上面那个“自动决定拆成几条轨道”的机制，而是在分离完成后执行的诊断模块。它用来回答“已经生成的这些轨道更像几个真实说话人”。它不会读取 TextGrid，也不会把估计结果反过来伪造分离轨道。

默认开关：

```text
SEPARATION_SPEAKER_COUNT_ESTIMATION=true
SPEAKER_EMBEDDING_BACKEND=ecapa
SPEAKER_EMBEDDING_STRONG_REQUIRED=true
SPEAKER_CLUSTER_THRESHOLD=0.55
SPEAKER_COUNT_MIN_TRACK_QUALITY=0.80
SPEAKER_COUNT_MAX_TRACKS=64
```

需要区分三件事：

- `expected_speakers`：质量路由里的期望说话人数，来自 `SEPARATION_EXPECTED_SPEAKERS`、`SEPARATION_MIN_TRACKS` 或函数参数。
- `auto recursive expansion`：没有明确期望人数时，通过“是否还能可信拆分轨道”来自适应增加轨道数。
- `speaker_count_estimation`：分离结束后的轨道聚类诊断结果，默认不反向改变本次分离候选。

### 分离后诊断总流程

```text
separated_tracks
-> 读取每条轨道音频
-> 计算轨道质量
-> 过滤低质量轨道
-> 提取 speaker embedding
-> 按 embedding 相似度聚类
-> 得到 estimated_speaker_count
-> 输出每条轨道的质量、cluster、global_speaker_id
```

### 1. 读取轨道并计算基础特征

系统最多读取：

```text
SPEAKER_COUNT_MAX_TRACKS=64
```

每条轨道会转成单声道，并计算：

- `duration_seconds`：轨道时长。
- `rms`：整条轨道平均能量。
- `active_ratio`：活跃帧比例。
- `clipping_ratio`：削波采样比例。
- `max_abs_correlation`：它和其他轨道之间最大的波形相关性。

活跃帧计算方式：

```text
frame = max(256, sample_rate * 0.064)
hop = max(128, frame / 2)
active_threshold = max(percentile_75_frame_rms * 0.25, 1e-5)
active_ratio = active_frames / total_frames
```

`active_ratio` 高，说明这条轨道中有足够多的有效语音；如果轨道基本是静音，就不应该被当成有效说话人。

### 2. 轨道质量分怎么计算

每条轨道会得到 `quality_score`，范围 `0-1`：

```text
quality =
  0.30 * activity_score
  + 0.25 * rms_score
  + 0.30 * isolation_score
  + 0.10 * duration_score
  + 0.05 * clipping_score
```

各项含义：

```text
activity_score = min(1, active_ratio / 0.25)
rms_score = min(1, rms / 0.015)
isolation_score = 1 - min(0.75, max_abs_correlation) / 0.75
duration_score = min(1, duration_seconds / 1.0)
clipping_score = 1 - min(0.02, clipping_ratio) / 0.02
```

解释：

- `activity_score`：轨道里是否有足够语音活动。
- `rms_score`：轨道整体能量是否足够。
- `isolation_score`：轨道和其他轨道是否不像。相关性越高，越可能是重复或串音，分数越低。
- `duration_score`：时长太短的轨道不可靠。
- `clipping_score`：削波越多，轨道越不可靠。

只有满足下面条件的轨道才进入说话人聚类：

```text
quality_score >= SPEAKER_COUNT_MIN_TRACK_QUALITY
并且
speaker embedding 提取成功
```

默认质量阈值是 `0.80`。这一步很重要，因为它会过滤静音轨道、严重串音轨道、太短的假轨道和异常削波轨道。

### 3. speaker embedding 如何提取

系统支持三类后端：

```text
ecapa          SpeechBrain ECAPA speaker embedding，默认
campp          FunASR CAM++ speaker embedding
local_spectral 本地频谱特征兜底
```

默认：

```text
SPEAKER_EMBEDDING_BACKEND=ecapa
SPEAKER_EMBEDDING_STRONG_REQUIRED=true
```

当 `strong required` 打开时，只接受 ECAPA 或 CAM++ 这类强说话人嵌入；本地频谱特征属于 degraded fallback，不会在强约束下使用。

ECAPA 路径会先从轨道中抽取最有语音活动的片段：

```text
按 0.5s 窗口、0.25s hop 扫描
计算每窗 RMS
阈值 = percentile_70_rms * 0.45
保留高能活动窗口
最多保留约 SPEAKER_EMBEDDING_MAX_ACTIVE_SECONDS 秒，默认 12 秒
送入 ECAPA 编码器
得到 embedding 并 L2 归一化
```

这样做是为了让 embedding 更集中在有效人声上，而不是把静音、噪声或长尾背景也送进说话人模型。

### 4. 如何根据 embedding 聚类

每条 accepted track 都有一个归一化 embedding。两个轨道的相似度用点积计算：

```text
similarity = dot(embedding_a, embedding_b)
```

因为 embedding 已经 L2 归一化，所以这个点积可以理解为余弦相似度。

聚类方法是凝聚式聚类：

```text
初始时每条轨道单独一个 cluster
循环寻找两个最相似的 cluster
如果最大相似度 >= SPEAKER_CLUSTER_THRESHOLD，就合并
否则停止
```

默认阈值：

```text
SPEAKER_CLUSTER_THRESHOLD=0.55
```

cluster 间相似度取两个 cluster 中任意轨道 embedding 相似度的最大值。最终 cluster 数就是原始全局说话人数估计：

```text
raw_global_estimated_speaker_count = len(clusters)
```

### 5. window consensus 是什么

如果轨道 ID 中带有窗口前缀，例如：

```text
w001_track_1
w001_track_2
w002_track_1
w002_track_2
```

系统会按窗口分别聚类，得到 `window_estimated_speaker_counts`。如果窗口内估计和全局聚类数量不一致，系统会优先采用窗口估计中的最大值：

```text
estimated = max(window_estimated_speaker_counts)
```

实际代码里如果 `raw_count` 和 `max_window_count` 不一致，会优先使用 `max_window_count`，并把来源标记为：

```text
global_count_source=window_consensus
```

这用于处理长音频分块或窗口化轨道中，全局 embedding 聚类可能把跨窗口轨道过度合并的问题。

### 6. single-window tracks 特例

如果没有窗口前缀、轨道数量不多，并且所有 accepted track 质量都足够高，系统会保留“每条轨道就是一个说话人”的结果：

```text
global_count_source=single_window_tracks
estimated_speaker_count=len(accepted_tracks)
```

这个特例是为了避免单个短样本中，不同说话人的 embedding 因数据太少被过度合并。

### 7. 稳定性如何计算

系统还会输出 `cluster_stability`，用于说明这个人数估计靠不靠谱。稳定性由四部分组成：

```text
stability =
  0.35 * min_track_quality
  + 0.25 * separation_margin
  + 0.25 * embedding_coverage
  + 0.15 * cluster_score
```

含义：

- `min_track_quality`：被接受轨道里的最低质量分，越高越稳定。
- `separation_margin`：不同 cluster 之间越不相似，稳定性越高。
- `embedding_coverage`：有 embedding 的轨道比例。
- `cluster_score`：cluster 内平均质量和相似度。

最终输出会进入：

```text
speaker_count_estimation
signal_metrics["estimated_speaker_count"]
signal_metrics["speaker_count_cluster_stability"]
signal_metrics["speaker_count_global_cluster_summary"]
```

### 8. 汇报口径

可以这样讲：

```text
自动递归扩展负责在分离过程中决定是否继续增加轨道；
分离后的说话人数诊断不直接相信轨道数，而是先过滤掉低质量轨道，
再用 ECAPA/CAM++ 说话人嵌入判断哪些轨道属于同一个人，
最后通过聚类数量得到 estimated_speaker_count。
```

这两个机制解决的问题不同：自动递归扩展决定“分离时要不要继续拆”，分离后诊断判断“拆出来的轨道像不像真实说话人”。

## 后处理

当前保留的后处理包括：

```text
SEPARATION_STFT_MASK_REFINEMENT=true
SEPARATION_LOW_OVERLAP_LEAKAGE_SUPPRESSION=true
SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION=true
SEPARATION_MIXTURE_CONSISTENCY=false
```

这些步骤用于减少串音、改善轨道掩膜和控制残差。`SEPARATION_MIXTURE_CONSISTENCY` 默认关闭，避免过度投影破坏已选轨道听感。

### STFT mask refinement

STFT mask refinement 是对选中 SpeechBrain 结果的频域细化。它会：

```text
原始混合音频 -> STFT 频谱
各分离轨道 -> STFT 幅度谱
根据各轨道幅度谱计算 soft mask
mask * mixture_spec -> 重新合成每条轨道
```

关键参数：

```text
SEPARATION_STFT_MASK_N_FFT=1024
SEPARATION_STFT_MASK_HOP=128
SEPARATION_STFT_MASK_POWER=1.5
SEPARATION_STFT_MASK_LIMIT=0.99
```

可以这样解释：模型先给出初始分离轨道，后处理再用原始混合频谱作为约束，让每个频点更多分配给能量更强的那条轨道，从而减少串音。

### low-overlap leakage suppression

这一步用于处理“本来不是重叠说话，但弱轨道里还残留另一个人的声音”的情况。

系统先计算两条轨道的活动重叠比例。如果重叠比例高，说明可能真的有重叠说话，就不强行压制；如果重叠比例低，则按帧比较各轨道 RMS：

```text
如果某一帧里轨道 A 比轨道 B 强至少 3 dB，
且该帧能量高于 -45 dB，
就把输掉的轨道乘以 0.1
```

默认配置：

```text
SEPARATION_LOW_OVERLAP_THRESHOLD=0.2
SEPARATION_LOW_OVERLAP_DOMINANCE_DB=3.0
SEPARATION_LOW_OVERLAP_LOSER_GAIN=0.1
SEPARATION_LOW_OVERLAP_ACTIVE_FLOOR_DB=-45
```

汇报口径：它不是删除重叠说话，而是在“低重叠场景”下压低泄漏串音。

### SpeechBrain residual projection

SpeechBrain residual projection 会计算：

```text
residual = 原始混合音频 - 所有分离轨道相加
```

然后把平均残差按一定比例加回每条轨道：

```text
projected_track = track + amount * residual / track_count
```

默认：

```text
SEPARATION_SPEECHBRAIN_RESIDUAL_PROJECTION_AMOUNT=1.0
```

它的目的不是重新分离，而是减少分离轨道相加后与原混合音频之间的能量缺口，让输出不要丢失太多原始语音成分。

## 与 ASR 的关系

ASR 和分离解决的是不同问题：

```text
ASR diarization: 某个时间段是谁在说话
speech separation: 某条音频轨道主要是谁的声音
```

完整链路中，ASR 先输出带时间戳的 transcript，分离质量路由可用它估计重叠比例。分离完成后，`separation_alignment_service` 再计算每个 ASR 片段在各分离轨道上的 RMS 能量，并写回：

```text
primary_track_id
primary_track_label
separation_tracks
```

如果一个 ASR 片段同时在多条分离轨道上有明显能量，`separation_tracks` 会保留多个轨道，用来表达重叠说话。这个对齐过程不使用 TextGrid。

### ASR 片段到轨道的能量对齐

分离模型输出的 `track 1`、`track 2` 没有天然身份顺序。项目用能量对齐来解释“某句 ASR 文本主要落在哪条分离轨道”：

```text
读取 ASR 片段 start/end
-> 在同一时间范围内读取每条分离轨道
-> 计算该时间窗 RMS 能量
-> 选择能量最高的轨道作为 primary_track
-> 如果多条轨道都超过活动阈值，记录为 separation_tracks
```

默认活动阈值：

```text
ASR_SEPARATION_ALIGNMENT_ACTIVE_RATIO=0.35
ASR_SEPARATION_ALIGNMENT_ACTIVE_FLOOR=0.00001
```

也就是说，某轨道在当前片段里的能量如果达到最高能量的一定比例，并且高于最小能量地板，就会被视为该片段相关轨道。

## TextGrid 事后验证

TextGrid 只作为事后验证真值使用，不参与分离候选选择、轨道重命名或 ASR 对齐。

默认设置：

```text
SEPARATION_EVAL_TRANSCRIBE_TRACKS=auto
SEPARATION_EVAL_MAX_TRANSCRIBE_SECONDS=120
```

当找到参考标注且音频不超过限制时，系统可以对每条分离轨道再跑一次 ASR，并与 TextGrid 中的参考说话人文本做字符相似度匹配。长会议默认不额外转写每条轨道，避免评测耗时过高。

## 输出字段

分离模块返回：

- `method`：选中的分离方法和后处理说明。
- `status`：成功、递归扩展、自动扩展或 placeholder 状态。
- `track_count`：输出轨道数。
- `tracks`：每条轨道的 `track_id`、`label`、`audio_url` 和说明。
- `metrics`：候选评分、诊断指标、后处理指标和质量路由选择结果。
- `speaker_count_estimation`：基于轨道和说话人嵌入估计的说话人数信息。

完整 `ProcessResult` 还会包含：

- `separation_alignment`：ASR 片段到分离轨道的能量对齐。
- `separation_evaluation`：TextGrid 事后验证结果。

## 失败兜底

以下情况会进入兜底：

- SpeechBrain、ClearVoice 或模型权重不可用。
- CUDA/CPU 推理失败。
- 输入音频格式异常或分块失败。
- 候选没有输出有效轨道。
- 所有候选评分都不可用。

兜底只返回 `placeholder` 轨道，保证页面不崩；系统不会把 ASR 说话人分段门控轨道冒充为真实分离结果。
