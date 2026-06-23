@echo off
setlocal

cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" (
  echo backend\.venv was not found.
  echo Please create/install the backend environment first.
  pause
  exit /b 1
)

rem Stable classroom demo mode:
rem - Keep real SpeechBrain separation enabled.
rem - Disable external LLM calls so FastAPI stays responsive.
set LLM_ENABLED=false
set ENHANCEMENT_MAX_SECONDS=300
set CHUNK_SECONDS=60
set CHUNK_OVERLAP_SECONDS=5
set SEPARATION_BACKEND=speechbrain
set SEPARATION_MODEL=speechbrain/sepformer-wsj02mix
set SEPARATION_DEVICE=cpu
set SEPARATION_MAX_SECONDS=60

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
