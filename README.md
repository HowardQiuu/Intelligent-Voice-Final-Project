# Intelligent Voice Final Project

这是一个面向中文会议场景的智能语音处理 Demo。当前交付版已经收敛为最小稳定流程：语音增强、语音分离、ASR 转写、会议摘要四个模块，加上必要的兜底路径，避免保留探究阶段的实验入口和评测缓存。

## 最终主流程

```text
会议音频上传
-> 音频标准化
-> 语音增强
-> ASR 转写
-> 最佳分离路径：Libri2Mix SepFormer / ClearVoice MossFormer2 / ReSepFormer 质量路由
-> ASR 片段与分离轨道对齐
-> 主题整理与会议摘要
-> 前端试听、转写和摘要展示
```

当前最佳分离配置：

```text
QUALITY_ROUTER_ENABLED=true
SEPARATION_INPUT_SOURCE=raw
SEPARATION_CANDIDATES=libri2mix,mossformer2,resepformer
SEPARATION_MODEL=speechbrain/sepformer-libri2mix
MOSSFORMER2_SEPARATION_MODEL=MossFormer2_SS_16K
SEPARATION_RECURSIVE_EXPANSION=true
SEPARATION_RECURSIVE_MODE=direct_split
```

其中 `libri2mix`、`mossformer2` 和 `resepformer` 是最终保留的真实盲源分离候选；模型不可用或输入不适合分离时只允许退回 `placeholder`，不再使用说话人分段门控轨道冒充分离结果。

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
http://127.0.0.1:5173
```

后端接口：

```text
http://127.0.0.1:8000
```

停止服务：

```powershell
.\stop_project.cmd
```

## 可选模型安装

基础依赖足够启动 Demo。若要使用最佳效果，需要安装对应模型依赖并准备权重：

```powershell
backend\.venv\Scripts\python.exe -m pip install clearvoice speechbrain pystoi pesq
backend\.venv\Scripts\python.exe scripts\download_models.py --separation --separation-model speechbrain/sepformer-libri2mix
```

CUDA 环境下保持 PyTorch 与本机 CUDA 版本匹配。若模型加载失败，系统会自动回退到兜底轨道。

## 目录结构

```text
backend/
  app/main.py                 FastAPI 入口
  app/services/               增强、分离、ASR、摘要等后端服务
  app/static/audio/           内置演示音频
  app/static/uploads/         运行时上传和输出目录
  models/                     本地模型权重
  checkpoints/                ClearVoice 等模型权重
  tests/                      后端单元测试

data/
  near_mix_dataset_v1/        最终保留的 near/headset close-talk 混合验证数据集

frontend/
  src/App.jsx                 前端主页面
  src/components/             音频试听、转写、摘要和诊断组件

docs/
  PIPELINE.md                 新音频完整数据链路
  ENHANCEMENT.md              语音增强模块说明
  SEPARATION.md               语音分离模块说明
  ASR.md                      ASR 模块说明
  SUMMARY.md                  摘要模块说明
  TEAM_SETUP.md               团队部署说明

scripts/
  create_near_mix_dataset.py  从 AliMeeting near 原始包复现生成 near-mix 数据集
  download_models.py          必要模型下载和预热
  run_backend.cmd             启动后端
  run_frontend.cmd            启动前端
  stop_ports.ps1              清理端口
```

## 文档

最终文档：

- [新音频完整数据链路](docs/PIPELINE.md)
- [语音增强](docs/ENHANCEMENT.md)
- [语音分离](docs/SEPARATION.md)
- [ASR 转写](docs/ASR.md)
- [会议摘要](docs/SUMMARY.md)
- [团队部署](docs/TEAM_SETUP.md)

## 测试

运行后端关键测试：

```powershell
backend\.venv\Scripts\python.exe -m unittest backend.tests.test_audio_quality_service backend.tests.test_enhancement_service backend.tests.test_separation_service backend.tests.test_asr_service backend.tests.test_summary_service backend.tests.test_pipeline_service
```

前端构建：

```powershell
cd frontend
npm run build
```

## 清理原则

本交付版已经移除探究阶段的评测缓存、外部模型适配实验、候选 oracle 对比脚本和非最终文档。保留内容只服务于当前最佳运行路径、必要兜底路径和团队复现。
