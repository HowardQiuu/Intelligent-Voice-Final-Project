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

核心最终配置：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_INPUT_SOURCE=raw
SEPARATION_CANDIDATES=libri2mix,mossformer2,gated
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
```

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

## 清理后不再保留的内容

以下内容属于探究阶段产物，不作为交付版流程的一部分：

- `.runtime/separation_eval` 离线评测缓存。
- `.runtime/listening_test` 临时试听样本。
- `scripts/eval_separation` 评测和外部模型适配脚本。
- external separation command 入口。
- ESPnet / TF-GridNet 外部适配实验记录。
- 多余模块文档和探究报告。

## 常用测试

后端关键测试：

```powershell
backend\.venv\Scripts\python.exe -m unittest backend.tests.test_audio_quality_service backend.tests.test_enhancement_service backend.tests.test_separation_service backend.tests.test_asr_service backend.tests.test_summary_service backend.tests.test_pipeline_service
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
- 确认 `SEPARATION_CANDIDATES=libri2mix,mossformer2,gated`。
- 查看后端返回的 `分离状态` 和 `quality_router_*` 指标。
- 模型不可用时允许进入 `gated` 或 `placeholder` 兜底。
