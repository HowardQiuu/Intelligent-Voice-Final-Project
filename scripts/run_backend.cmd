@echo off
setlocal

set "ROOT=%~dp0.."
set "BACKEND=%ROOT%\backend"

cd /d "%BACKEND%"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] backend\.venv was not found.
  echo Run backend environment setup first, then start_project.cmd again.
  exit /b 1
)

set LLM_ENABLED=false
set ENHANCEMENT_MAX_SECONDS=300
set CHUNK_SECONDS=60
set CHUNK_OVERLAP_SECONDS=5
set SEPARATION_BACKEND=speechbrain
set SEPARATION_MODEL=speechbrain/sepformer-wsj02mix
set SEPARATION_DEVICE=cpu
set SEPARATION_MAX_SECONDS=60

echo FastAPI backend starting at http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
