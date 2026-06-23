# 智能会议语音分离与转写系统 Demo

这是一个面向课程展示的 Web Demo，用于演示：

`会议音频输入 -> 语音增强 -> 语音分离 -> 自动语音识别 -> 概要生成 -> 会议纪要输出`

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

如需启用大语言模型摘要生成，复制并填写后端配置文件：

```powershell
Copy-Item backend\.env.example backend\.env
```

然后在 `backend/.env` 中填写 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_TIMEOUT_SECONDS`。未配置 `LLM_API_KEY` 或模型调用失败时，系统会自动使用缓存摘要，保证课堂演示稳定。完整教程见 `docs/LLM_SUMMARY_SETUP.md`。

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
- 展示处理链路进度：音频输入、语音增强、语音分离、ASR、摘要。
- 播放原始音频和增强后音频。
- 展示可替换的语音分离轨道接口和演示轨道。
- 对比直接转写和增强后转写。
- 展示带时间戳与说话人标签的转写片段。
- 输出会议主题、关键词、摘要、关键决策和待办事项。
- 支持上传 `.wav/.mp3/.m4a/.aac/.flac/.ogg` 音频进入演示流程。

## 文档索引

- 大语言模型摘要生成配置：`docs/LLM_SUMMARY_SETUP.md`
- DeepFilterNet 语音增强配置：`docs/DEEPFILTERNET_SETUP.md`
- 语音分离模块接入：`docs/SPEECH_SEPARATION_SETUP.md`

## 后续替换真实模型的位置

- 语音增强：`backend/app/services/enhancement_service.py`
  - 当前已实现语音增强流水线：优先调用 DeepFilterNet 去噪；若未安装 DeepFilterNet，则使用 ffmpeg 频谱降噪兜底；随后执行轻量 dereverb-lite 后处理。
  - DeepFilterNet 是可选依赖，需要真实模型去噪时手动安装：
    `python -m pip install deepfilternet`
  - 当前 dereverb-lite 不是完整 WPE/DNN 去混响模型，而是通过高通/低通、残余噪声抑制、动态归一化和压缩提升远场语音清晰度。
- 语音分离：`backend/app/services/separation_service.py`
  - 当前提供稳定的演示占位接口，返回统一的 `separated_tracks` 结构。
  - 后续可替换为 SepFormer、Conv-TasNet、Demucs 或说话人条件分离模型，输出多个说话人独立音轨。
  - 独立接口：`POST /api/separate-demo/{case_id}` 和 `POST /api/separate-upload`，便于先单独调试分离模块，再接入完整流水线。
- ASR 转写：`backend/app/services/asr_service.py`
  - 可接入 faster-whisper、本地 Whisper 或云端 ASR。
- 摘要生成：`backend/app/services/summary_service.py`
  - 当前已接入 OpenAI-Compatible Chat Completions API，可通过 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_TIMEOUT_SECONDS` 配置 DeepSeek、OpenAI、通义兼容接口或本地兼容服务。
  - Prompt 输入包含会议名称、增强后 ASR 文本、带时间戳和说话人标签的转写片段。
  - 模型输出被约束为结构化 JSON：`title`、`keywords`、`abstract`、`decisions`、`action_items`。
  - 后端会校验 JSON 字段，缺失列表会自动补默认值；未配置 Key、网络错误、超时或模型输出异常时回退到缓存摘要。
- 样例缓存：`backend/app/data/demo_results.json`
  - 可替换为真实会议样例的预生成转写和摘要结果。

## 摘要生成模块说明

摘要生成模块负责把 ASR 与说话人分段结果转换为结构化会议纪要，是系统从“语音识别”走向“会议内容理解”的核心部分。课堂展示时可以重点说明：

1. 输入不只是普通文本，而是包含时间戳和说话人标签的会议转写。
2. Prompt 要求模型输出固定 JSON，方便前端直接展示主题、关键词、摘要、决策和待办事项。
3. 系统在处理指标中展示摘要来源、模型名称和调用状态，能够区分“LLM API 生成”和“缓存兜底”。
4. 失败兜底机制保证即使 API Key 未配置或网络异常，Demo 仍可稳定运行。

## 小组分工建议

- 前端与交互：优化页面布局、进度动画、音频波形对比。
- 语音增强：接入真实增强模型，输出增强前后音频。
- 语音分离与 ASR：接入说话人分离模型、Whisper/说话人分段模型，输出独立音轨和时间戳转写。
- 摘要生成：优化关键词、会议摘要、决策和待办事项格式。

## 展示建议

课堂展示时优先使用“带噪会议片段”，依次说明：

1. 原始会议音频存在噪声和混响。
2. 语音增强后听感更清晰。
3. 语音分离模块预留多个说话人轨道输出，便于后续替换真实模型。
4. 增强和分离后的转写结果更稳定。
5. 系统最终生成结构化会议纪要。
