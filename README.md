# 智能会议语音分离与转写系统 Demo

这是一个面向课程展示的 Web Demo，用于演示：

`会议音频输入 -> 语音增强 -> 说话人处理 -> 自动语音识别 -> 概要生成 -> 会议纪要输出`

当前版本优先保证课堂演示稳定，内置 3 组样例会议缓存结果。上传音频也可以进入同一条展示链路，后续可逐步替换为真实模型。

## 目录结构

```text
smart-meeting-demo/
  backend/   FastAPI 后端，提供音频、处理流程和缓存结果接口
  frontend/  React + Vite 前端，展示系统流程和处理结果
```

## 启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

后端健康检查：

```text
http://127.0.0.1:8000/api/health
```

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

默认访问：

```text
http://127.0.0.1:5173
```

## 已实现功能

- 选择 3 个内置会议样例并运行完整流程。
- 展示处理链路进度：音频输入、语音增强、说话人处理、ASR、摘要。
- 播放原始音频和增强后音频。
- 对比直接转写和增强后转写。
- 展示带时间戳与说话人标签的转写片段。
- 输出会议主题、关键词、摘要、关键决策和待办事项。
- 支持上传 `.wav/.mp3/.m4a/.aac/.flac/.ogg` 音频进入演示流程。

## 后续替换真实模型的位置

- 语音增强：`backend/app/services/enhancement_service.py`
  - 当前已实现语音增强流水线：优先调用 DeepFilterNet 去噪；若未安装 DeepFilterNet，则使用 ffmpeg 频谱降噪兜底；随后执行轻量 dereverb-lite 后处理。
  - DeepFilterNet 是可选依赖，需要真实模型去噪时手动安装：
    `python -m pip install deepfilternet`
  - 当前 dereverb-lite 不是完整 WPE/DNN 去混响模型，而是通过高通/低通、残余噪声抑制、动态归一化和压缩提升远场语音清晰度。
- ASR 转写：`backend/app/services/asr_service.py`
  - 可接入 faster-whisper、本地 Whisper 或云端 ASR。
- 摘要生成：`backend/app/services/summary_service.py`
  - 可接入大语言模型 API，生成会议摘要、决策和待办事项。
- 样例缓存：`backend/app/data/demo_results.json`
  - 可替换为真实会议样例的预生成转写和摘要结果。

## 小组分工建议

- 前端与交互：优化页面布局、进度动画、音频波形对比。
- 语音增强：接入真实增强模型，输出增强前后音频。
- ASR 与说话人处理：接入 Whisper/说话人分段模型，输出时间戳。
- 摘要生成：优化关键词、会议摘要、决策和待办事项格式。

## 展示建议

课堂展示时优先使用“带噪会议片段”，依次说明：

1. 原始会议音频存在噪声和混响。
2. 语音增强后听感更清晰。
3. 增强后转写结果更稳定。
4. 系统最终生成结构化会议纪要。
