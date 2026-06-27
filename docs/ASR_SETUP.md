# ASR 转写模块配置

本文档说明如何启用项目中的真实 ASR 转写模块。当前实现优先使用 `faster-whisper`，失败时自动回退到演示转写，保证课堂展示稳定。

## 安装依赖

在项目根目录运行：

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-asr.txt
```

首次运行会下载 Whisper 模型，建议提前在网络稳定时完成。CPU 演示推荐从 `small` 或 `base` 开始。

## 环境变量

可在 `backend/.env` 或启动脚本中配置：

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
ASR_BEAM_SIZE=1
ASR_BEST_OF=1
ASR_CPU_THREADS=0
ASR_NUM_WORKERS=1
ASR_CONDITION_ON_PREVIOUS_TEXT=false
```

常用选择：

- `ASR_BACKEND=placeholder`：强制使用演示转写，不加载模型。
- `ASR_MODEL=tiny/base/small`：模型越大效果通常越好，但 CPU 越慢。
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
- 首版不强制识别分离后的多条说话人轨道；speaker 标签统一为“说话人”，后续可接说话人分段或多轨 ASR。

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
ASR_BACKEND=faster-whisper
ASR_MODEL=base
```
