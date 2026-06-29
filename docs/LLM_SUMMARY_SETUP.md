# 大语言模型摘要生成配置教程

本文档用于配置项目中的“概要生成 -> 会议纪要输出”模块。配置完成后，后端会把会议转写文本发送给 OpenAI-Compatible 大语言模型 API，生成结构化会议纪要。

## 1. 功能说明

摘要生成模块位于：

```text
backend/app/services/summary_service.py
```

它会读取以下输入：

- 会议名称
- 增强后的 ASR 转写文本
- 带时间戳和说话人标签的转写片段

然后要求大模型输出固定 JSON：

```json
{
  "title": "会议标题",
  "keywords": ["关键词1", "关键词2"],
  "abstract": "100-200字会议摘要",
  "decisions": ["关键决策1", "关键决策2"],
  "action_items": ["待办事项1", "待办事项2"]
}
```

前端会直接展示这些字段：会议主题、关键词、摘要、关键决策和待办事项。

## 2. 安装后端依赖

进入后端目录：

```powershell
cd backend
```

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

其中 `httpx` 用于请求 OpenAI-Compatible Chat Completions API。

## 3. 推荐配置：DeepSeek

DeepSeek 提供 OpenAI-Compatible 接口，配置最简单，适合课程 Demo。本项目推荐把参数写在 `backend/.env` 文件里，这样每次启动后端时会自动读取，不需要反复输入环境变量命令。

先复制模板：

```powershell
Copy-Item backend\.env.example backend\.env
```

然后打开 `backend/.env`，填写：

```text
LLM_ENABLED=true
LLM_API_KEY=这里填写你的 DeepSeek API Key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=20
LLM_TOPIC_WINDOW_SECONDS=120
LLM_TOPIC_MAX_BLOCKS=80
```

然后启动后端：

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

注意：`backend/.env` 已加入 `.gitignore`，不要把真实 API Key 提交到 Git 仓库。

## 4. 其他模型配置示例

### OpenAI

```text
LLM_ENABLED=true
LLM_API_KEY=你的 OpenAI API Key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=20
```

### 本地 OpenAI-Compatible 服务

如果本地模型服务兼容 `/chat/completions`：

```text
LLM_ENABLED=true
LLM_API_KEY=local-key
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=local-chat-model
LLM_TIMEOUT_SECONDS=60
```

## 5. 转写主题分组

同一组 LLM 配置也会用于 `transcript_topic_service.py`。它会把 ASR 结果按时间窗口整理为主题块，再请求 OpenAI-compatible API 输出主题标题、摘要和对应时间块。

可调参数：

```text
LLM_TOPIC_WINDOW_SECONDS=120
LLM_TOPIC_MAX_BLOCKS=80
```

- `LLM_TOPIC_WINDOW_SECONDS`：每个候选主题时间块的窗口大小，最小值为 30 秒。
- `LLM_TOPIC_MAX_BLOCKS`：发送给模型的最大时间块数量，最小值为 8，避免超长会议 prompt 过大。

`LLM_ENABLED=false` 会同时关闭真实摘要和真实主题分类，系统会使用缓存/兜底摘要以及本地时间块主题分组。没有配置 `LLM_API_KEY` 或接口调用失败时，也会自动回退，不影响页面展示。

## 6. 启动前端并验证

另开一个 PowerShell 窗口：

```powershell
cd frontend
npm install
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

推荐选择“带噪会议片段”，点击“运行样例”。

如果 LLM 调用成功，页面右下角“处理指标”中应看到类似内容：

```text
摘要生成：LLM API
摘要模型：deepseek-chat
摘要状态：结构化 JSON 生成成功
转写主题来源：LLM API
转写主题数量：...
```

如果没有配置 API Key，或接口调用失败，会看到类似内容：

```text
摘要生成：缓存兜底
摘要模型：deepseek-chat
摘要状态：未配置 LLM_API_KEY
转写主题来源：本地兜底
```

或：

```text
摘要状态：LLM 调用失败：ConnectTimeout
```

这表示系统自动回退到了缓存摘要，Demo 仍然可以正常展示。

## 7. 快速接口测试

后端启动后，可以直接访问健康检查：

```text
http://127.0.0.1:8000/api/health
```

也可以用 PowerShell 调用样例处理接口：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/process-demo/noisy_meeting"
```

返回结果中的关键字段：

```text
summary.title
summary.keywords
summary.abstract
summary.decisions
summary.action_items
signal_metrics.摘要生成
signal_metrics.摘要模型
signal_metrics.摘要状态
signal_metrics.转写主题来源
signal_metrics.转写主题数量
```

## 8. 常见问题

### 1. 页面显示“缓存兜底”

可能原因：

- 没有设置 `LLM_API_KEY`
- API Key 填错
- `LLM_BASE_URL` 填错
- 当前网络无法访问模型 API
- 模型返回内容不是合法 JSON

处理方式：

```powershell
Get-Content backend\.env
```

确认配置文件存在且字段完整后，重新启动后端。

### 2. 修改 `.env` 后没有生效

后端只在启动时读取 `backend/.env`。修改配置后，需要停止并重新启动 FastAPI 后端。

### 3. 模型返回格式不稳定

后端已经要求模型只输出 JSON，并会尝试从代码块或普通文本中提取 JSON。若仍失败，系统会回退缓存摘要，不影响课堂演示。

### 4. 请求超时

可以适当调大超时时间：

```text
LLM_TIMEOUT_SECONDS=60
```

把这一行写入 `backend/.env` 后重启后端。

## 9. 课堂展示讲解词

可以这样介绍摘要生成模块：

```text
我们的摘要模块使用 OpenAI-Compatible 大语言模型接口。
后端把增强后的 ASR 文本，以及带时间戳和说话人标签的转写片段，组织成 Prompt 发送给模型。
模型必须返回固定 JSON，包括会议标题、关键词、摘要、关键决策和待办事项。
为了保证课堂演示稳定，我们设计了失败兜底机制：如果没有配置 API Key、网络异常或模型输出格式错误，系统会自动使用缓存摘要。
页面上的处理指标会显示摘要来源、模型名称和调用状态，因此可以清楚看到当前结果是由 LLM API 生成，还是由缓存兜底生成。
转写区域会按主题展示时间块；如果开启 LLM 主题分类，主题标题和摘要由模型生成，如果关闭或失败，则使用本地兜底分组。
```

## 10. 最小演示流程

1. 启动后端前配置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。
2. 启动 FastAPI 后端。
3. 启动 React 前端。
4. 选择“带噪会议片段”。
5. 点击“运行样例”。
6. 展示会议纪要输出和处理指标中的摘要生成状态。
