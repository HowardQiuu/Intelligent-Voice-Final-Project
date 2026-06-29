# 语音分离模块接入说明

本文档说明项目中“语音增强 -> 语音分离 -> ASR 转写 -> 摘要生成”的语音分离接口设计。

## 1. 当前流程

完整处理链路为：

```text
会议音频输入
-> 语音增强
-> 语音分离
-> 自动语音识别
-> 摘要生成
-> 会议纪要输出
```

当前版本提供两种后端：

- `placeholder`：默认后端，复用增强后音频作为一条“会议语音轨道”，保证 Demo 稳定。
- `speechbrain`：调用 SpeechBrain SepFormer 预训练模型，输出多个说话人音轨。

## 2. 代码位置

```text
backend/app/services/separation_service.py
```

核心函数：

```python
separate_demo_audio(case_id, enhanced_audio_url)
separate_uploaded_audio(enhanced_audio_url)
```

后续可以在这里接入：

- SepFormer
- Conv-TasNet
- Demucs
- Asteroid
- 说话人条件分离模型
- 自研语音分离模型

## 3. 安装真实分离模型依赖

基础后端依赖不会强制安装 PyTorch 和 SpeechBrain，避免普通 Demo 启动过慢。需要真实语音分离时，额外执行：

```powershell
backend\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-separation.txt
```

Windows note: the backend uses SpeechBrain `LocalStrategy.COPY` to avoid symlink permission errors during first model download.

首次运行 `speechbrain` 后端时会下载模型权重，耗时取决于网络环境。

## 4. 配置 `.env`

在 `backend/.env` 中加入：

```text
SEPARATION_BACKEND=speechbrain
SEPARATION_MODEL=speechbrain/sepformer-wsj02mix
SEPARATION_DEVICE=auto
SEPARATION_MAX_SECONDS=60
SEPARATION_CHUNK_SECONDS=60
SEPARATION_MAX_CHUNKS=120
ENHANCEMENT_MAX_SECONDS=300
ENHANCEMENT_CHUNK_SECONDS=60
ENHANCEMENT_MAX_CHUNKS=120
ENHANCEMENT_WORKERS=2
```

如果希望保持稳定演示模式：

```text
SEPARATION_BACKEND=placeholder
```

## 5. 返回结构

语音分离统一返回：

```json
{
  "method": "Demo speech separation placeholder",
  "track_count": "1",
  "tracks": [
    {
      "track_id": "noisy_meeting_speaker_mix",
      "label": "降噪后会议语音轨道",
      "audio_url": "/static/audio/noisy_meeting_enhanced.wav",
      "description": "演示模式复用增强后音频，后续可替换为真实说话人分离模型输出。"
    }
  ]
}
```

如果真实模型输出多个说话人音轨，可以返回：

```json
{
  "method": "SepFormer speaker separation",
  "track_count": "2",
  "tracks": [
    {
      "track_id": "speaker_1",
      "label": "说话人1",
      "audio_url": "/static/uploads/example_speaker_1.wav",
      "description": "分离出的第一位说话人音轨。"
    },
    {
      "track_id": "speaker_2",
      "label": "说话人2",
      "audio_url": "/static/uploads/example_speaker_2.wav",
      "description": "分离出的第二位说话人音轨。"
    }
  ]
}
```

## 6. 后端接口

完整流水线接口会自动包含分离结果：

```text
POST /api/process-demo/{case_id}
POST /api/upload
POST /api/upload-session
POST /api/upload-session/{upload_id}/chunk
POST /api/upload-session/{upload_id}/complete
POST /api/process-local-file
```

单独调试语音分离模块可以使用：

```text
POST /api/separate-demo/{case_id}
POST /api/separate-upload
```

示例：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/separate-demo/noisy_meeting"
```

## 7. 失败兜底

以下情况会自动回退到 `placeholder`：

- 未安装 `speechbrain`、`torch` 或 `torchaudio`
- 模型下载失败
- 推理异常
- 音频文件不存在或格式不兼容
- CPU 推理过慢或音频过长
- 上传音频超过 `SEPARATION_MAX_SECONDS` 时，按 `SEPARATION_CHUNK_SECONDS` 分块调用 SepFormer，再按说话人轨道拼接输出
- 上传音频超过 `ENHANCEMENT_MAX_SECONDS` 时，按 `ENHANCEMENT_CHUNK_SECONDS` 分块调用 DeepFilterNet，增强后拼接为一条完整音频，再进入分离/ASR/摘要流程

回退时接口仍会返回 `tracks`，并在指标中显示：

```text
分离算法：Placeholder fallback
分离状态：SpeechBrain failed: ...
```

## 8. 分块处理与增强可视化

长会议音频不会一次性送入重模型。后端会先读取音频时长，然后按配置生成分块计划：

```text
CHUNK_SECONDS=60
CHUNK_OVERLAP_SECONDS=5
```

接口会返回：

```text
processing_chunks
```

每个分块包含 `chunk_id`、`start`、`end`、`duration_seconds`、`status` 和 `description`。当前版本已经在后端使用分块策略执行长音频增强、长音频分离和长音频 ASR，并在模型不可用或单块失败时兜底。

语音增强会额外生成可视化图片：

```text
enhancement_visual_url
```

前端会直接展示增强前后的能量包络图，并在处理指标中显示：

```text
原始平均能量
增强后平均能量
平均能量变化
原始峰值
增强后峰值
```

这样汇报时可以用“图 + 数据”说明增强效果，而不是只凭听感描述。

## 9. 前端展示

前端读取完整处理结果中的：

```text
result.separated_tracks
```

并在“增强前后试听”区域展示“语音分离轨道”。如果后续模型输出多个音轨，前端会按列表直接展示。

## 10. 汇报时的表述建议

可以这样介绍：

```text
系统流程中在语音增强之后增加了语音分离模块。
当前演示版本默认使用占位分离轨道保证稳定；如果安装 SpeechBrain 依赖并配置 speechbrain 后端，系统会调用 SepFormer 预训练模型输出多个说话人音轨。
后端已经预留了独立分离接口和完整流水线中的 separated_tracks 字段，因此后续替换为更适合会议场景的分离模型时不需要重写前端和主流程。
```
# 当前版本说明：默认不是 SepFormer

完整会议提取 pipeline 当前默认不再优先调用 SpeechBrain SepFormer。实际顺序是：

```text
DeepFilterNet 增强
-> FunASR/SenseVoice 中文ASR
-> fsmn-vad + cam++ 说话人分段
-> build_speaker_tracks_from_transcript 生成说话人轨道
-> 摘要、主题分类、质量评分
```

也就是说，`result.separated_tracks` 在完整 pipeline 中默认来自说话人时间段轨道，描述字段会包含：

```text
FunASR speaker diarization gated track
```

SpeechBrain SepFormer 仍然保留，但只在以下情况使用：

- 调用独立分离接口 `/api/separate-demo/{case_id}` 或 `/api/separate-upload`；
- 或在完整 pipeline 中说话人轨道生成失败后回到 `separate_uploaded_audio`，并且显式设置了 `SEPARATION_BACKEND=speechbrain`。

如果没有显式设置，`SEPARATION_BACKEND` 默认是 `placeholder`，不会自动优先调用 SepFormer。
