# 智能会议语音分离与转写系统 Demo

这是一个面向智能语音课程结课展示的 Web Demo，用于演示从会议音频到结构化会议纪要的完整链路：

```text
会议音频输入 -> 语音增强 -> 分块处理 -> 语音分离 -> 自动语音识别 -> 摘要生成 -> 会议纪要输出
```

项目当前目标是“稳定可演示 + 接口可替换”。短音频可以接入 SpeechBrain SepFormer 做真实分离；长会议音频会生成分块计划，语音增强采用分块 DeepFilterNet 推理后拼接，避免 30 分钟以上音频一次性占满内存。

## 一键启动

在项目根目录运行：

```powershell
.\start_project.cmd
```

脚本会自动清理旧的 `8000` / `5173` 监听进程，然后启动：

- FastAPI 后端：`http://127.0.0.1:8000`
- Vite 前端：`http://127.0.0.1:5173`

停止服务：

```powershell
.\stop_project.cmd
```

后端健康检查：

```text
http://127.0.0.1:8000/api/health
```

## 目录结构

```text
Intelligent-Voice-Final-Project/
  backend/
    app/
      main.py                 FastAPI 路由入口
      models.py               API 数据结构
      data/                   内置样例与缓存结果
      services/               后端功能模块
      static/                 音频、上传结果、可视化图片
    tests/                    后端单元测试
    requirements.txt          基础依赖
    requirements-separation.txt  可选语音分离依赖

  frontend/
    src/
      App.jsx                 前端主页面状态与交互
      api.js                  API 请求封装
      components/             展示组件
      styles.css              页面样式

  docs/                       模块说明文档
  scripts/stop_ports.ps1      端口清理脚本
  start_project.cmd           一键启动前后端
  stop_project.cmd            一键停止前后端
```

更详细的模块说明见：

```text
docs/PROJECT_STRUCTURE.md
```

## 后端模块

后端入口：

```text
backend/app/main.py
```

只负责 FastAPI 初始化、静态文件挂载、路由注册和基础参数校验。

核心服务：

- `pipeline_service.py`：完整流水线编排，连接增强、分块、分离、ASR 兜底和摘要。
- `audio_service.py`：音频目录、上传归一化、时长读取、静态 URL 解析。
- `enhancement_service.py`：语音增强和长音频分块增强策略。
- `chunking_service.py`：长会议音频分块计划。
- `separation_service.py`：SpeechBrain SepFormer 分离与 placeholder 兜底。
- `asr_service.py`：ASR 步骤说明和上传音频转写兜底。
- `summary_service.py`：OpenAI-compatible LLM 摘要生成与缓存兜底。
- `visualization_service.py`：增强前后能量包络 SVG 图生成。
- `demo_cache.py`：内置样例和缓存结果读取。

## 前端模块

前端入口：

```text
frontend/src/main.jsx
```

页面编排：

```text
frontend/src/App.jsx
frontend/src/api.js
```

展示组件：

- `Pipeline.jsx`：处理链路。
- `AudioCompare.jsx`：增强试听、增强可视化、分离轨道、分块计划。
- `Transcript.jsx`：带时间戳转写。
- `Summary.jsx`：会议纪要和处理指标。
- `EmptyState.jsx`：初始和加载状态。

## 大文件处理

页面主入口“上传音频”现在使用分块上传，不再一次性把完整音频塞进浏览器和 FastAPI 内存。默认每块 `4 MB`，流程为：

```text
创建上传会话 -> 前端 File.slice 分块 -> 后端分片落盘 -> 合并文件 -> 进入处理流水线
```

相关接口：

```text
POST /api/upload-session
POST /api/upload-session/{upload_id}/chunk
POST /api/upload-session/{upload_id}/complete
```

分块大小可通过环境变量调整：

```text
UPLOAD_CHUNK_MB=4
```

页面侧边栏仍保留“高级兜底：本地大文件路径”入口。它适合浏览器上传被系统策略拦截、文件位于后端同一台机器、或需要完全绕过浏览器传输时使用，例如：

```text
C:\workshop\school lesson\speech signal processing\final_work\train_S\wav\20200623_S_R001S01C01.flac
```

课堂演示优先使用“上传音频”按钮；路径入口只是备选方案。

长音频保护策略：

- 超过 `ENHANCEMENT_MAX_SECONDS=300` 秒：按 `ENHANCEMENT_CHUNK_SECONDS=60` 分块调用 DeepFilterNet，增强后再拼接。
- 超过 `SEPARATION_MAX_SECONDS=60` 秒：按 `SEPARATION_CHUNK_SECONDS=60` 分块调用 SpeechBrain SepFormer，按说话人轨道拼接。
- 超过 `ASR_MAX_SECONDS=600` 秒：按 `ASR_CHUNK_SECONDS=60` 分块调用 faster-whisper，并合并全局时间戳。
- 模型不可用或单块推理失败时，才回退到兜底分离轨道/兜底转写，保证页面不崩。

分块配置：

```text
CHUNK_SECONDS=60
CHUNK_OVERLAP_SECONDS=5
```

## 语音增强可视化

系统会为增强前后音频生成 SVG 图片，并返回：

```text
enhancement_visual_url
```

前端会显示增强前后的能量包络图，同时在“处理指标”中展示：

- 原始平均能量
- 增强后平均能量
- 平均能量变化
- 原始峰值
- 增强后峰值

这样展示时可以用“图 + 数据”说明增强效果。

## 语音分离

短音频可使用 SpeechBrain SepFormer：

```text
SEPARATION_BACKEND=speechbrain
SEPARATION_MODEL=speechbrain/sepformer-wsj02mix
SEPARATION_DEVICE=auto
SEPARATION_MAX_SECONDS=60
SEPARATION_CHUNK_SECONDS=60
SEPARATION_MAX_CHUNKS=120
```

安装可选依赖：

```powershell
backend\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-separation.txt
```

如果模型不可用、音频过长或推理失败，系统会自动回退到 placeholder 轨道，保证页面不崩。

## ASR 转写

上传音频、本地文件和分块上传完成后会优先调用本地 `faster-whisper` 生成转写。默认使用 CPU + int8：

```text
ASR_BACKEND=faster-whisper
ASR_MODEL=small
ASR_DEVICE=auto
ASR_COMPUTE_TYPE=auto
ASR_LANGUAGE=zh
ASR_MAX_SECONDS=600
ASR_CHUNK_SECONDS=60
ASR_MAX_CHUNKS=240
ASR_VAD_FILTER=true
```

安装可选依赖：

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-asr.txt
```

如果未安装模型依赖、首次下载模型失败、音频超过 `ASR_MAX_SECONDS`，系统会自动回退到演示转写，不影响摘要和页面展示。详细说明见 `docs/ASR_SETUP.md`。

## 摘要生成

摘要模块支持 OpenAI-compatible Chat Completions API，可接 DeepSeek、OpenAI、通义兼容接口或本地兼容服务。

配置文件：

```text
backend/.env
```

常用变量：

```text
LLM_ENABLED=false
LLM_API_KEY=填写你的 API Key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=20
```

稳定演示模式默认 `LLM_ENABLED=false`，直接使用缓存/兜底摘要。需要展示真实 LLM 摘要时，再改为 `LLM_ENABLED=true` 并配置 Key。

详细教程见：

```text
docs/LLM_SUMMARY_SETUP.md
```

## 常用测试命令

后端测试：

```powershell
backend\.venv\Scripts\python.exe -m unittest backend.tests.test_summary_service backend.tests.test_separation_service backend.tests.test_enhancement_service backend.tests.test_chunking_visualization backend.tests.test_upload_session_service backend.tests.test_asr_service
```

后端编译检查：

```powershell
backend\.venv\Scripts\python.exe -m compileall backend\app
```

前端构建：

```powershell
cd frontend
npm.cmd run build
```

## 文档索引

- 项目结构：`docs/PROJECT_STRUCTURE.md`
- LLM 摘要配置：`docs/LLM_SUMMARY_SETUP.md`
- DeepFilterNet 增强配置：`docs/DEEPFILTERNET_SETUP.md`
- SpeechBrain 分离配置：`docs/SPEECH_SEPARATION_SETUP.md`
- faster-whisper 转写配置：`docs/ASR_SETUP.md`

## 课堂展示建议

推荐展示顺序：

1. 运行内置“带噪会议片段”，说明完整链路。
2. 展示增强前后试听和增强可视化图。
3. 展示分块处理计划，说明长会议音频如何避免内存爆掉。
4. 展示分离轨道，说明短音频可接入真实 SepFormer，长音频稳定兜底。
5. 展示结构化会议纪要：主题、关键词、摘要、关键决策、待办事项。
