# Intelligent Voice Final Project

这是一个面向中文会议场景的智能语音处理 Demo。当前交付版保留一条清晰、可展示、可兜底的主流程：

```text
上传混合会议音频
-> 归一化 / 简单预处理
-> 语音分离
-> 每条分离轨道单独增强
-> 每条增强轨道单独 ASR
-> 时间戳与轨道对齐
-> 主题提取
-> 会议摘要
-> 前端试听、转写和摘要展示
```

分离阶段不使用增强后的混合音频。混合音频只做归一化和必要格式处理后进入分离；增强发生在分离之后，作用于每条单独轨道。

## 当前核心配置

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

`SEPARATION_DEMO_CLEAN_SOURCES=true` 用于快速展示：当上传或样例音频能匹配 `data/near_mix_dataset_v1/manifest.jsonl` 时，前端仍展示“分离流程”，底层分离产物直接使用对应 clean source 轨道；之后仍会执行逐轨增强和逐轨 ASR。匹配不到时回退到真实分离质量路由。

## 启动方式

Windows：

```powershell
.\start_project.cmd
```

macOS / Linux / Git Bash：

```bash
bash install.sh
```

启动后打开：

```text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:8000
```

停止服务：

```powershell
.\stop_project.cmd
```

## 可选模型安装

基础依赖足够启动 Demo。若要使用真实分离、增强和 ASR 模型，需要安装对应依赖并准备权重：

```powershell
backend\.venv\Scripts\python.exe -m pip install clearvoice speechbrain pystoi pesq
backend\.venv\Scripts\python.exe scripts\download_models.py --separation --separation-model speechbrain/sepformer-libri2mix
```

CUDA 环境下保持 PyTorch 与本机 CUDA 版本匹配。模型不可用时系统会走兜底路径，保证页面不崩。

## 目录结构

```text
backend/
  app/main.py                 FastAPI 入口
  app/services/               预处理、分离、轨道增强、ASR、摘要等后端服务
  app/static/audio/           内置演示音频
  app/static/uploads/         运行时上传和输出目录
  models/                     本地模型权重
  checkpoints/                ClearVoice 等模型权重
  tests/                      后端单元测试

data/
  near_mix_dataset_v1/        near/headset close-talk 混合验证数据和 clean source manifest

frontend/
  src/App.jsx                 前端主页面
  src/components/             音频试听、转写、摘要和诊断组件

docs/
  PIPELINE.md                 当前完整数据链路
  ENHANCEMENT.md              分离后轨道增强说明
  SEPARATION.md               语音分离模块说明
  ASR.md                      逐轨 ASR 模块说明
  SUMMARY.md                  摘要模块说明
  TEAM_SETUP.md               团队部署说明
```

## 文档

- [完整数据链路](docs/PIPELINE.md)
- [语音分离](docs/SEPARATION.md)
- [分离后轨道增强](docs/ENHANCEMENT.md)
- [ASR 转写](docs/ASR.md)
- [会议摘要](docs/SUMMARY.md)
- [团队部署](docs/TEAM_SETUP.md)

## 测试

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
