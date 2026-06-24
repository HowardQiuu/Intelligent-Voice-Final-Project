@echo off
setlocal

set "ROOT=%~dp0.."
set "BACKEND=%ROOT%\backend"
set "CONDA_BACKEND_PY=%USERPROFILE%\.conda\envs\voice-final-py311\python.exe"
set "CONDA_BACKEND_SCRIPTS=%USERPROFILE%\.conda\envs\voice-final-py311\Scripts"

cd /d "%BACKEND%"

if exist "%CONDA_BACKEND_PY%" (
  set "BACKEND_PY=%CONDA_BACKEND_PY%"
  set "PATH=%CONDA_BACKEND_SCRIPTS%;%PATH%"
) else if exist ".venv\Scripts\python.exe" (
  set "BACKEND_PY=.venv\Scripts\python.exe"
  set "PATH=%CD%\.venv\Scripts;%PATH%"
) else (
  echo [ERROR] backend Python environment was not found.
  echo Expected either:
  echo   %CONDA_BACKEND_PY%
  echo   %CD%\.venv\Scripts\python.exe
  exit /b 1
)

echo FastAPI backend starting at http://127.0.0.1:8000
"%BACKEND_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
