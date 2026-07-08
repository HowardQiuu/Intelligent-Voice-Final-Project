# Team Setup

## 环境准备

推荐使用 Windows PowerShell，在项目根目录执行：

```powershell
.\install_project.cmd --full --download-models
```

如果只安装依赖不启动服务：

```powershell
.\install_project.cmd --full --download-models --no-start
```

手动启动：

```powershell
.\start_project.cmd
```

访问：

```text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:8000
```

## 本地配置

复制配置模板：

```powershell
Copy-Item backend\.env.example backend\.env
```

当前展示版核心配置：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_INPUT_SOURCE=normalized
SEPARATION_DEMO_CLEAN_SOURCES=true
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
SEPARATION_RECURSIVE_EXPANSION=true
SEPARATION_RECURSIVE_MODE=direct_split
```

当前 pipeline 是：

```text
混合音频归一化 / 简单预处理
-> 分离
-> 分离轨道逐条增强
-> 增强轨道逐条 ASR
-> 对齐 / 主题 / 摘要
```

`SEPARATION_DEMO_CLEAN_SOURCES=true` 用于快速展示 near-mix 数据：匹配 manifest 时直接返回 clean source 作为分离轨道；匹配不到时走真实分离模型。

如需 LLM 摘要，在 `backend/.env` 中配置：

```text
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

不要提交真实密钥。

## 必要权重

交付版保留的权重目录：

```text
backend/models/
backend/checkpoints/
```

推荐预下载：

```powershell
backend\.venv\Scripts\python.exe scripts\download_models.py --separation --separation-model speechbrain/sepformer-libri2mix
```

ClearVoice / MossFormer2 权重由 ClearVoice 按需读取 `backend/checkpoints`。

## 常用测试

后端完整测试：

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover tests
```

前端构建：

```powershell
cd frontend
npm run build
```

## 故障排查

如果前端无法访问：

```powershell
.\stop_project.cmd
.\start_project.cmd
```

如果分离模型失败：

- 确认 `speechbrain`、`clearvoice`、`torch`、`torchaudio` 已安装。
- 确认 `SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer`。
- 查看后端返回的 `分离状态` 和 `quality_router_*` 指标。
- 模型不可用时只允许进入 `placeholder` 兜底。

如果展示结果需要接近 clean source：

- 使用 `data/near_mix_dataset_v1` 中 manifest 可匹配的样例或上传文件名。
- 保持 `SEPARATION_DEMO_CLEAN_SOURCES=true`。
- 记住该开关只替代分离产物，后续逐轨增强和逐轨 ASR 仍执行。
