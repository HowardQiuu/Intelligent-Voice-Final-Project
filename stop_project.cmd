@echo off
setlocal

echo Stopping services on ports 8000 and 5173...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_ports.ps1"

echo Done.
