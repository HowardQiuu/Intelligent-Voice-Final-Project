# ASR 转写模块配置

本文档说明如何启用项目中的真实 ASR 转写模块。当前完整 pipeline 默认优先使用 `FunASR/SenseVoice + fsmn-vad + cam++`，输出中文转写、时间戳和说话人标签；如果 FunASR 不可用，再自动回退到 `faster-whisper`，最后回退到演示转写，保证课堂展示稳定。

## 安装依赖

在项目根目录运行：

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-asr.txt
```

如果本机有 `scripts/run_backend.cmd` 中配置的 `voice-final-py311` conda 环境，后端会优先使用它。FunASR 在 Python 3.11 环境更稳；Python 3.13 下部分依赖可能需要本地 C++ 编译工具链，容易安装失败。

首次运行会下载 Whisper 模型，建议提前在网络稳定时完成。CPU 演示推荐从 `small` 或 `base` 开始。

FunASR/SenseVoice 还需要 ModelScope 模型缓存。项目默认使用 `backend/.runtime/modelscope`，避免 Windows 用户目录全局缓存出现权限、文件锁或 SSL 下载问题。如果全局缓存 `C:\Users\<你>\.cache\modelscope\hub\models\iic` 中已经有模型，可以复制到项目缓存的 `backend/.runtime/modelscope/models/iic` 下。

## 环境变量

可在 `backend/.env` 或启动脚本中配置：

```text
ASR_BACKEND=funasr
FUNASR_MODEL=iic/SenseVoiceSmall
FUNASR_VAD_MODEL=fsmn-vad
FUNASR_PUNC_MODEL=ct-punc
FUNASR_SPK_MODEL=cam++
FUNASR_SPK_MODE=vad_segment
FUNASR_DEVICE=auto
FUNASR_MODELSCOPE_FILE_LOCK=false
FUNASR_MODELSCOPE_CACHE=
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

常用选择：

- `ASR_BACKEND=funasr`：默认中文会议主路径，调用 FunASR/SenseVoice、VAD 和 CAM++。
- `ASR_BACKEND=faster-whisper`：跳过 FunASR，直接使用原来的 faster-whisper 路径。
- `ASR_BACKEND=placeholder`：强制使用演示转写，不加载模型。
- `FUNASR_MODEL=iic/SenseVoiceSmall`：中文会议主路径 ASR 模型。
- `FUNASR_SPK_MODEL=cam++`：说话人分段/聚类模型。
- `FUNASR_SPK_MODE=vad_segment`：按 VAD 语音段分配说话人，更适合远场长会议；默认 `punc_segment` 在部分 SenseVoice 长音频上可能无法产生稳定句子时间戳。
- `FUNASR_MODELSCOPE_FILE_LOCK=false`：Windows 上 ModelScope 文件锁可能卡住，课堂演示建议保持 false。
- `FUNASR_MODELSCOPE_CACHE`：可选，指定 ModelScope 缓存目录；留空时使用 `backend/.runtime/modelscope`。
- `FASTER_WHISPER_MODEL=tiny/base/small`：FunASR 失败后的 Whisper 回退模型。
- `ASR_MAX_SECONDS=600`：超过该时长进入分块转写，不再直接跳过真实 ASR。
- `ASR_CHUNK_SECONDS=60`：每个 ASR 分块的窗口长度。
- `ASR_MAX_CHUNKS=240`：最大分块数量，防止超长音频无限占用机器。
- `ASR_VAD_FILTER=true`：启用静音过滤，减少无效片段。
- `ASR_BEAM_SIZE=1` / `ASR_BEST_OF=1`：控制 faster-whisper 解码搜索规模。课堂 CPU 演示推荐保持 1，优先保证速度。
- `ASR_CPU_THREADS=0`：使用 faster-whisper 默认线程策略；需要限制 CPU 占用时可改成固定线程数。
- `ASR_NUM_WORKERS=1`：模型推理 worker 数量。单机演示通常保持 1，避免内存占用过高。
- `ASR_CONDITION_ON_PREVIOUS_TEXT=false`：分块转写时默认不让上一块文本强影响下一块，减少长音频中错误累积。

## 流水线行为

- 内置 demo 样例继续使用缓存转写，保证演示稳定。
- 上传音频、分块上传完成、本地文件处理会优先识别增强后主音频。
- 当前版本优先用 FunASR/CAM++ 识别说话人标签，转写中的 speaker 会稳定显示为“说话人 A/B/C”。如果回退到 faster-whisper 或演示数据，则使用已有转写或兜底 speaker 标签。

返回给前端的字段保持不变：

```text
direct_asr_text
enhanced_asr_text
transcript
signal_metrics
```

处理指标会显示：

```text
ASR 后端
ASR 模型
ASR 设备
ASR 状态
ASR 语言
ASR 分段数
ASR 分块窗口
```

同时会返回运行耗时指标，例如 `runtime_asr_seconds` 和 `runtime_total_seconds`，前端会在处理诊断区域展示。

## 失败兜底

以下情况会自动使用演示兜底，不影响页面和摘要展示：

- 未安装 `faster-whisper`
- `ASR_BACKEND` 仍被 `.env` 写成 `faster-whisper`
- 未在后端实际使用的 Python 环境中安装 `funasr` / `modelscope`
- FunASR 导入时找不到可执行 `ffmpeg`
- ModelScope 全局缓存权限异常、文件锁卡住或 SSL 下载失败
- 模型下载失败
- 音频分块数量超过 `ASR_MAX_CHUNKS`
- 音频格式异常或推理报错
- 模型返回空结果

课堂展示时，如果机器性能有限，可以先使用：

```text
ASR_BACKEND=placeholder
```

需要展示真实转写时，再改回：

```text
ASR_BACKEND=funasr
FUNASR_MODEL=iic/SenseVoiceSmall
FUNASR_VAD_MODEL=fsmn-vad
FUNASR_SPK_MODEL=cam++
FUNASR_SPK_MODE=vad_segment
```
