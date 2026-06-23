@echo off
setlocal

set "ROOT=%~dp0"

echo [1/3] Cleaning old backend/frontend processes on ports 8000 and 5173...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_ports.ps1"

echo [2/3] Starting FastAPI backend at http://127.0.0.1:8000 ...
start "voice-backend" /min cmd /k ""%ROOT%scripts\run_backend.cmd""

echo [3/3] Starting Vite frontend at http://127.0.0.1:5173 ...
start "voice-frontend" /min cmd /k ""%ROOT%scripts\run_frontend.cmd""

echo.
echo Project is starting. Open:
echo   http://127.0.0.1:5173
echo.
echo To stop both services, run:
echo   stop_project.cmd
echo.
pause
