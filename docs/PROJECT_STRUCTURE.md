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
- `separation_service.py`：SpeechBrain SepFormer 分离与 placeholder 兜底。
- `asr_service.py`：ASR 流程步骤和上传音频转写兜底。
- `summary_service.py`：OpenAI-compatible LLM 摘要生成与缓存兜底。
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
- `Summary.jsx`：会议纪要和处理指标
- `EmptyState.jsx`：初始/加载状态

```text
frontend/src/styles.css
```

页面样式。

## 大文件处理建议

主上传入口已经改为分块上传：前端使用 `File.slice` 将音频切成小块，后端将分片落盘并在完成后合并，再进入增强、分离、转写和摘要流水线。默认分片大小为 `UPLOAD_CHUNK_MB=4`。

页面侧边栏的“高级兜底：本地大文件路径”仍保留，用于浏览器上传受限或文件就在后端机器上的特殊情况。
