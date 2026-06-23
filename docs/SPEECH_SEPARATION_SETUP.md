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

当前版本优先保证课程 Demo 稳定，因此语音分离模块使用占位实现：复用增强后音频作为一条“会议语音轨道”。后续接入真实分离模型时，只需要替换服务层实现，接口形状可以保持不变。

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

## 3. 返回结构

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

## 4. 后端接口

完整流水线接口会自动包含分离结果：

```text
POST /api/process-demo/{case_id}
POST /api/upload
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

## 5. 前端展示

前端读取完整处理结果中的：

```text
result.separated_tracks
```

并在“增强前后试听”区域展示“语音分离轨道”。如果后续模型输出多个音轨，前端会按列表直接展示。

## 6. 汇报时的表述建议

可以这样介绍：

```text
系统流程中在语音增强之后增加了语音分离模块。
当前演示版本为了保证稳定性，使用占位分离轨道展示接口形状；后续可以接入 SepFormer、Conv-TasNet 或 Demucs 等真实语音分离模型。
后端已经预留了独立分离接口和完整流水线中的 separated_tracks 字段，因此模型替换时不需要重写前端和主流程。
```
