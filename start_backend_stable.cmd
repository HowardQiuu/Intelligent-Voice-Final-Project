@echo off
setlocal

cd /d "%~dp0backend"

set "CONDA_BACKEND_PY=%USERPROFILE%\.conda\envs\voice-final-py311\python.exe"
set "CONDA_BACKEND_SCRIPTS=%USERPROFILE%\.conda\envs\voice-final-py311\Scripts"

if exist "%CONDA_BACKEND_PY%" (
  set "BACKEND_PY=%CONDA_BACKEND_PY%"
  set "PATH=%CONDA_BACKEND_SCRIPTS%;%PATH%"
) else if exist ".venv\Scripts\python.exe" (
  set "BACKEND_PY=.venv\Scripts\python.exe"
  set "PATH=%CD%\.venv\Scripts;%PATH%"
) else (
  echo backend Python environment was not found.
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

"%BACKEND_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
