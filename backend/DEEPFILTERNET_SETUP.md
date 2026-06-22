# DeepFilterNet 接入说明

本项目的上传音频增强模块现在保留两种 DeepFilterNet 后端：

- `cli`：调用官方命令行工具 `deepFilter` / `deep-filter`，这是当前默认方案，推荐课程 Demo 使用。
- `source`：直接调用官方源码中的 Python 推理接口，并加载预训练权重。这是进阶方案，需要处理 Rust / libDF / pyDF 编译依赖。

## 推荐方案：官方 CLI

课程 Demo 推荐使用 CLI 方式。它仍然使用 DeepFilterNet 官方模型和预训练权重，只是我们通过官方封装好的命令行入口调用，避免在项目里直接处理 Rust 编译链。

安装：

```powershell
cd F:\homework\3.2\voice_signal_process\Final\smart-meeting-demo\backend
pip install deepfilternet
```

检查命令是否可用：

```powershell
deepFilter --help
# 或
deep-filter --help
```

启动后端：

```powershell
cd F:\homework\3.2\voice_signal_process\Final\smart-meeting-demo\backend
$env:DEEPFILTERNET_BACKEND="cli"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

如果不设置 `DEEPFILTERNET_BACKEND`，系统默认也是 `cli`。

上传音频时，后端流程为：

```text
上传音频
-> normalize_upload 统一音频格式
-> 调用 deepFilter / deep-filter
-> 输出 DeepFilterNet 增强后的音频
```

## 保留方案：源码 + 预训练权重

如果后续想展示“直接使用源码加载模型权重”，可以使用 `source` 后端。

推荐目录结构：

```text
backend/
  vendor/
    DeepFilterNet/
      DeepFilterNet/
        df/
          enhance.py
      models/
        DeepFilterNet3/
          DeepFilterNet3/
            config.ini
            checkpoints/
              model_120.ckpt.best
```

如果源码和模型不放在默认位置，可以指定环境变量：

```powershell
$env:DEEPFILTERNET_SOURCE_DIR="D:\path\to\DeepFilterNet\DeepFilterNet"
$env:DEEPFILTERNET_MODEL_DIR="D:\path\to\DeepFilterNet\models\DeepFilterNet3\DeepFilterNet3"
$env:DEEPFILTERNET_BACKEND="source"
```

源码方式会调用：

```python
from df.enhance import enhance, init_df, load_audio, save_audio
```

但这种方式通常还需要安装 `torch`、`torchaudio`、`loguru`，并构建 `libdf/pyDF`。如果没有 Rust 编译环境，容易卡在依赖安装上，因此不建议作为课程 Demo 的默认运行方式。

## 当前代码选择

当前默认后端：

```text
DEEPFILTERNET_BACKEND=cli
```

代码位置：

```text
backend/app/services/enhancement_service.py
```

其中：

- `denoise_audio_with_cli()`：官方 CLI 调用方式。
- `denoise_audio_with_source()`：源码 + 权重调用方式，保留备用。
- `denoise_audio()`：根据 `DEEPFILTERNET_BACKEND` 选择后端，默认走 CLI。

## 汇报时的表述建议

可以这样表述：

> 系统在上传音频后调用 DeepFilterNet 官方预训练模型进行单通道语音增强。工程实现上，为了保证课程 Demo 的稳定性，我们采用官方 CLI 作为推理入口；同时保留了基于源码和预训练权重的调用接口，便于后续进一步研究模型结构与推理过程。
