# 团队环境复现说明

本文档用于让新的维护者把本项目配置到和当前主开发环境一致的能力边界：前后端可启动、可选真实 ASR、可选 SpeechBrain 语音分离、可选 DeepFilterNet 增强、可选 LLM 摘要。

## 1. 从零安装并启动

Windows:

```powershell
copy backend\.env.example backend\.env
.\install_project.cmd --full --download-models
```

macOS / Linux / Git Bash:

```bash
cp backend/.env.example backend/.env
bash install.sh --full --download-models
```

完成后打开：

```text
http://127.0.0.1:5173
```

如果暂时不需要真实模型，只想先把项目跑起来：

```powershell
.\install_project.cmd
```

```bash
bash install.sh
```

## 2. 当前推荐配置

`backend/.env.example` 已经给出当前项目推荐配置。需要真实摘要时，把 `LLM_API_KEY` 改成自己的 Key；没有 Key 时可以设置：

```text
LLM_ENABLED=false
```

当前主开发环境使用的非密钥配置如下：

```text
LLM_MODEL=deepseek-chat
LLM_ENABLED=true
LLM_TOPIC_WINDOW_SECONDS=120
LLM_TOPIC_MAX_BLOCKS=80
ENHANCEMENT_MAX_SECONDS=300
ENHANCEMENT_CHUNK_SECONDS=60
ENHANCEMENT_MAX_CHUNKS=120
ENHANCEMENT_WORKERS=2
ENHANCEMENT_SKIP_SECONDS=0
DEEPFILTERNET_BACKEND=cli
CHUNK_SECONDS=60
CHUNK_OVERLAP_SECONDS=5
SEPARATION_BACKEND=speechbrain
SEPARATION_MODEL=speechbrain/sepformer-wsj02mix
SEPARATION_DEVICE=auto
SEPARATION_MAX_SECONDS=60
SEPARATION_CHUNK_SECONDS=60
SEPARATION_MAX_CHUNKS=120
ASR_BACKEND=faster-whisper
ASR_MODEL=small
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
UPLOAD_CHUNK_MB=4
```

前端也可以按需配置：

```text
VITE_API_BASE=http://127.0.0.1:8000
VITE_UPLOAD_CHUNK_MB=4
VITE_MAX_BROWSER_UPLOAD_MB=120
```

## 3. 权重和缓存在哪里

仓库不会提交模型权重，原因是它们体积较大，并且 `backend/models/` 已被 `.gitignore` 忽略。

当前模型来源：

- `faster-whisper`：`--download-models --with-asr` 会下载 `ASR_MODEL` 指定的模型，默认 `small`，项目本地缓存目录为 `backend/models/faster-whisper/small`。
- `SpeechBrain SepFormer`：`--download-models --with-separation` 会下载 `speechbrain/sepformer-wsj02mix`，项目本地缓存目录为 `backend/models/speechbrain/sepformer-wsj02mix`。
- `DeepFilterNet`：`--download-models --with-deepfilter` 会运行一次 CLI 预热。默认 `DEEPFILTERNET_BACKEND=cli`，通常不需要手动管理 `.ckpt`。只有切到 `DEEPFILTERNET_BACKEND=source` 时，才需要配置 `DEEPFILTERNET_SOURCE_DIR` 和 `DEEPFILTERNET_MODEL_DIR`。
- `LLM`：不下载本地权重，调用 OpenAI-compatible API。需要每个人自己配置 Key。

单独预下载模型：

```powershell
backend\.venv\Scripts\python.exe scripts\download_models.py --asr --separation --deepfilter
```

```bash
backend/.venv/bin/python scripts/download_models.py --asr --separation --deepfilter
```

只下载指定模型：

```powershell
backend\.venv\Scripts\python.exe scripts\download_models.py --asr --asr-model small
backend\.venv\Scripts\python.exe scripts\download_models.py --separation --separation-model speechbrain/sepformer-wsj02mix
```

## 4. 无网络或下载慢时

可以从已经配置好的机器复制以下目录给同伴：

```text
backend/models/faster-whisper/small
backend/models/speechbrain/sepformer-wsj02mix
```

复制后保持 `.env` 中：

```text
ASR_MODEL=small
SEPARATION_MODEL=speechbrain/sepformer-wsj02mix
```

应用会优先使用 `backend/models/faster-whisper/small` 里的本地 Whisper 模型；SpeechBrain 会使用 `backend/models/speechbrain/sepformer-wsj02mix`。

不要提交以下文件：

```text
backend/.env
backend/models/
frontend/node_modules/
frontend/dist/
```

## 5. 常见命令

安装全部可选模型并启动：

```powershell
.\install_project.cmd --full --download-models
```

只安装依赖和下载模型，不启动：

```powershell
.\install_project.cmd --full --download-models --no-start
```

启动已有环境：

```powershell
.\start_project.cmd
```

停止服务：

```powershell
.\stop_project.cmd
```

后端健康检查：

```text
http://127.0.0.1:8000/api/health
```

## 6. 给 Codex 的本地配置提示词

同伴可以在 Codex 中打开项目根目录，然后粘贴下面这段：

```text
请帮我把这个项目配置到团队推荐环境。

要求：
1. 先阅读 README.md、docs/TEAM_SETUP.md、backend/.env.example。
2. 如果 backend/.env 不存在，从 backend/.env.example 复制一份，但不要替我填写真实 API Key。
3. 在 Windows 下优先运行 .\install_project.cmd --full --download-models --no-start；在 Bash 环境下运行 bash install.sh --full --download-models --no-start。
4. 安装完成后检查 backend/models/faster-whisper/small 和 backend/models/speechbrain/sepformer-wsj02mix 是否存在。
5. 最后启动项目，并告诉我前端地址、后端健康检查地址，以及哪些模型成功下载，哪些失败或回退。
6. 不要提交 backend/.env、backend/models、frontend/node_modules、frontend/dist。
```

如果只是想快速跑通，不下载真实模型，可以把第 3 条改成：

```text
运行 .\install_project.cmd --no-start 或 bash install.sh --no-start。
```
