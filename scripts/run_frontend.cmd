@echo off
setlocal

set "ROOT=%~dp0.."
set "FRONTEND=%ROOT%\frontend"

cd /d "%FRONTEND%"

if not exist "node_modules" (
  echo [ERROR] frontend\node_modules was not found.
  echo Run npm install in frontend first, then start_project.cmd again.
  exit /b 1
)

echo Vite frontend starting at http://127.0.0.1:5173
npm.cmd run dev -- --host 127.0.0.1 --port 5173
