# 项目结构说明

## 一键启动

在项目根目录运行：

```powershell
.\start_project.cmd
```

停止前后端：

```powershell
.\stop_project.cmd
```

## 后端模块

```text
backend/app/main.py
```

FastAPI 应用入口，只负责：

- 创建应用
- 注册 CORS / static
- 提供 API 路由
- 做简单参数校验

```text
backend/app/models.py
```

Pydantic 数据结构，包括处理结果、分离轨道、分块计划、会议纪要等。

```text
backend/app/services/
```

后端功能模块：

- `pipeline_service.py`：完整处理流水线编排，连接增强、分块、分离、ASR 兜底和摘要。
- `audio_service.py`：音频目录、上传归一化、音频时长读取、静态 URL 解析。
- `enhancement_service.py`：语音增强与长音频分块增强策略。
- `chunking_service.py`：长会议音频分块计划。
- `separation_service.py`：默认根据 FunASR/CAM++ 说话人时间段生成可试听轨道；SpeechBrain SepFormer 保留为可选实验分离后端。
- `asr_service.py`：ASR 流程步骤和上传音频转写兜底。
- `summary_service.py`：OpenAI-compatible LLM 摘要生成与缓存兜底。
- `transcript_topic_service.py`：将转写整理成主题时间块，支持 LLM 分类和本地兜底。
- `upload_session_service.py`：分块上传会话、分片落盘、合并与清理。
- `visualization_service.py`：增强前后能量包络 SVG 图生成。
- `demo_cache.py`：内置样例和缓存结果读取。

## 前端模块

```text
frontend/src/main.jsx
```

React 入口，只负责挂载 `App`。

```text
frontend/src/App.jsx
```

页面状态和交互编排：

- 加载样例列表
- 运行样例
- 上传音频
- 处理本地大文件路径
- 分发结果给展示组件

```text
frontend/src/api.js
```

前端 API 封装和基础配置。

```text
frontend/src/components/
```

页面展示组件：

- `Pipeline.jsx`：处理链路
- `AudioCompare.jsx`：增强试听、增强可视化、分离轨道、分块计划
- `Transcript.jsx`：带时间戳转写
- `Summary.jsx`：会议纪要
- `ProcessingDiagnostics.jsx`：后端耗时、模型状态和兜底原因等处理诊断
- `EmptyState.jsx`：初始/加载状态

```text
frontend/src/styles.css
```

页面样式。

## 大文件处理建议

主上传入口已经改为分块上传：前端使用 `File.slice` 将音频切成小块，后端将分片落盘并在完成后合并，再进入增强、分离、转写和摘要流水线。默认分片大小为 `UPLOAD_CHUNK_MB=4`。

页面侧边栏的“高级兜底：本地大文件路径”仍保留，用于浏览器上传受限或文件就在后端机器上的特殊情况。

## 运行诊断与主题转写

完整处理结果会在 `signal_metrics` 中返回阶段耗时和模型状态，例如 `runtime_enhancement_seconds`、`runtime_asr_seconds`、`runtime_summary_seconds`、`runtime_topic_seconds` 和 `runtime_total_seconds`。前端通过 `ProcessingDiagnostics.jsx` 展示这些字段，便于判断当前耗时集中在哪个阶段。

转写展示优先使用后端返回的 `transcript_topics`。如果启用了 LLM 且配置了 Key，主题分组由 `transcript_topic_service.py` 调用 OpenAI-compatible API 生成；如果没有 Key、关闭了 `LLM_ENABLED` 或调用失败，则使用本地时间块兜底，页面仍然可以正常展示。
