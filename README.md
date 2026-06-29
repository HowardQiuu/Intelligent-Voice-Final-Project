# 智能会议语音分离与转写系统 Demo

这是一个面向智能语音课程结课展示的 Web Demo，用于演示从会议音频到结构化会议纪要的完整链路：

```text
会议音频输入 -> 语音增强 -> 分块处理 -> 语音分离 -> 自动语音识别 -> 摘要生成 -> 会议纪要输出
```

项目当前目标是“稳定可演示 + 接口可替换”。完整会议提取主流程现在优先采用中文会议路线：DeepFilterNet 增强后调用 FunASR/SenseVoice + VAD + CAM++ 生成带说话人标签的中文转写，再根据说话人时间段生成可试听轨道和质量评分。SpeechBrain SepFormer 保留为独立实验分离后端；faster-whisper 保留为 FunASR 不可用时的 ASR 回退。

## 一键安装并启动

首次运行根据系统选择：

```bash
# macOS / Linux / Git Bash
bash install.sh
```

```powershell
# Windows PowerShell / CMD
.\install_project.cmd
```

脚本会自动完成：

- 创建 `backend/.venv` Python 虚拟环境
- 安装后端基础依赖 `backend/requirements.txt`
- 安装前端依赖 `frontend/package-lock.json`
- 清理旧的 `8000` / `5173` 监听进程
- 启动 FastAPI 后端和 Vite 前端

启动后打开：

```text
http://127.0.0.1:5173
```

默认安装的是稳定演示所需的基础环境。真实 ASR、语音分离、DeepFilterNet 降噪依赖体积较大，可按需安装：

```bash
bash install.sh --with-asr
bash install.sh --with-separation
bash install.sh --with-deepfilter
bash install.sh --full
bash install.sh --full --download-models
```

Windows 下对应命令为：

```powershell
.\install_project.cmd --with-asr
.\install_project.cmd --with-separation
.\install_project.cmd --with-deepfilter
.\install_project.cmd --full
.\install_project.cmd --full --download-models
```

只安装依赖、不启动服务：

```bash
bash install.sh --no-start
```

```powershell
.\install_project.cmd --no-start
```

`install.sh` 启动后按 `Ctrl+C` 可停止本次启动的前后端服务；后端日志在 `.runtime/backend.log`。Windows 脚本会复用下面的 `start_project.cmd`，停止服务可运行 `.\stop_project.cmd`。

多人协作时，推荐先阅读：

```text
docs/TEAM_SETUP.md
```

## Windows 一键启动

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
  scripts/download_models.py  可选模型权重预下载/预热脚本
  scripts/stop_ports.ps1      端口清理脚本
  install.sh                  一键安装依赖并启动前后端
  install_project.cmd         Windows 一键安装依赖并启动前后端
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
- `separation_service.py`：说话人时间段轨道生成、可选 SpeechBrain SepFormer 分离与 placeholder 兜底。
- `asr_service.py`：ASR 步骤说明和上传音频转写兜底。
- `summary_service.py`：OpenAI-compatible LLM 摘要生成与缓存兜底。
- `transcript_topic_service.py`：按时间块组织转写，并用 LLM 或兜底逻辑生成主题分组。
- `upload_session_service.py`：分块上传会话、分片落盘和合并。
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
- `Summary.jsx`：会议纪要。
- `ProcessingDiagnostics.jsx`：后端耗时、模型状态和兜底原因等处理诊断。
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

前端默认使用 `VITE_UPLOAD_CHUNK_MB=4` 作为兜底分片大小；如果后端创建上传会话时返回了 `chunk_size_bytes`，页面会优先使用后端值。

页面侧边栏仍保留“高级兜底：本地大文件路径”入口。它适合浏览器上传被系统策略拦截、文件位于后端同一台机器、或需要完全绕过浏览器传输时使用，例如：

```text
C:\workshop\school lesson\speech signal processing\final_work\train_S\wav\20200623_S_R001S01C01.flac
```

课堂演示优先使用“上传音频”按钮；路径入口只是备选方案。

长音频保护策略：

- 超过 `ENHANCEMENT_MAX_SECONDS=300` 秒：按 `ENHANCEMENT_CHUNK_SECONDS=60` 分块调用 DeepFilterNet，增强后再拼接。
- 完整 pipeline 默认先执行 FunASR/SenseVoice + VAD + CAM++，再根据转写中的说话人时间段生成说话人轨道。
- 只有在独立分离接口或显式配置 `SEPARATION_BACKEND=speechbrain` 时，才调用 SpeechBrain SepFormer。
- FunASR 不可用时，上传流程会自动回退到 faster-whisper；超过 `ASR_MAX_SECONDS=600` 秒时 faster-whisper 回退路径按 `ASR_CHUNK_SECONDS=60` 分块转写。
- 如需演示时跳过 DeepFilterNet，可设置 `DEEPFILTERNET_BACKEND=off`；如需仅对超长音频跳过增强，可设置 `ENHANCEMENT_SKIP_SECONDS`。
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

## 处理诊断

后端会把各阶段耗时和模型状态写入 `signal_metrics`，前端 `ProcessingDiagnostics.jsx` 会集中展示。常见字段包括：

```text
runtime_normalize_seconds
runtime_chunk_plan_seconds
runtime_enhancement_seconds
runtime_visual_seconds
runtime_separation_seconds
runtime_asr_seconds
runtime_summary_seconds
runtime_topic_seconds
runtime_total_seconds
```

这些指标适合在调试或汇报时说明瓶颈位置，例如 DeepFilterNet 增强、SpeechBrain 分离、ASR 转写或 LLM 主题分类分别耗时多少，以及当前结果是否来自真实模型或兜底路径。

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

上传音频、本地文件和分块上传完成后会优先调用 FunASR/SenseVoice 生成中文会议转写、VAD 分段和 CAM++ 说话人标签；如果 FunASR 不可用，则自动回退到本地 `faster-whisper`。推荐配置：

```text
ASR_BACKEND=funasr
FUNASR_MODEL=iic/SenseVoiceSmall
FUNASR_VAD_MODEL=fsmn-vad
FUNASR_SPK_MODEL=cam++
FUNASR_DEVICE=auto
FASTER_WHISPER_MODEL=small
ASR_DEVICE=auto
ASR_COMPUTE_TYPE=auto
ASR_LANGUAGE=zh
ASR_MAX_SECONDS=600
ASR_CHUNK_SECONDS=60
ASR_MAX_CHUNKS=240
ASR_VAD_FILTER=true
ASR_BEAM_SIZE=1
ASR_BEST_OF=1
ASR_CPU_THREADS=0
ASR_NUM_WORKERS=1
ASR_CONDITION_ON_PREVIOUS_TEXT=false
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
LLM_TOPIC_WINDOW_SECONDS=120
LLM_TOPIC_MAX_BLOCKS=80
```

稳定演示推荐设置 `LLM_ENABLED=false`，直接使用缓存/兜底摘要和兜底主题分组。`backend/.env.example` 展示的是真实 LLM 接入形态；如果没有配置 `LLM_API_KEY` 或接口失败，系统仍会自动回退。需要展示真实 LLM 摘要和主题分组时，设置 `LLM_ENABLED=true` 并配置 Key。

详细教程见：

```text
docs/LLM_SUMMARY_SETUP.md
```

## 常用测试命令

后端测试：

```powershell
backend\.venv\Scripts\python.exe -m unittest backend.tests.test_audio_service backend.tests.test_summary_service backend.tests.test_separation_service backend.tests.test_enhancement_service backend.tests.test_chunking_visualization backend.tests.test_upload_session_service backend.tests.test_asr_service
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
- 团队环境复现：`docs/TEAM_SETUP.md`

## 课堂展示建议

推荐展示顺序：

1. 运行内置“带噪会议片段”，说明完整链路。
2. 展示增强前后试听和增强可视化图。
3. 展示分块处理计划，说明长会议音频如何避免内存爆掉。
4. 展示说话人轨道，说明完整 pipeline 优先使用 FunASR/SenseVoice + VAD + CAM++；SpeechBrain SepFormer 是可选实验后端。
5. 展示结构化会议纪要：主题、关键词、摘要、关键决策、待办事项。
# 中文会议 Pipeline 升级说明

新的默认设计面向中文课堂/会议展示：

```text
DeepFilterNet 增强 -> FunASR/SenseVoice 中文ASR -> fsmn-vad + cam++ 说话人分段 -> 说话人轨道 -> 主题转写 -> 会议纪要 -> 质量评分
```

核心创新点：

- 中文会议自适应路由：FunASR 不可用时自动回退到 faster-whisper 或演示兜底。
- 说话人轨道：根据说话人时间段生成可试听轨道，说明来源为 `FunASR speaker diarization gated track`。
- 会议质量评估：输出语音覆盖率、疑似重叠比例、检测说话人数和 0-100 会议提取质量评分。
- 可解释诊断：前端展示主处理后端、中文ASR模型、说话人分段模型和路由说明。

详细说明见 `docs/CHINESE_MEETING_PIPELINE.md`。
