@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"
set "CONDA_BACKEND_PY=%USERPROFILE%\.conda\envs\voice-final-py311\python.exe"
if exist "%CONDA_BACKEND_PY%" (
  set "VENV_PY=%CONDA_BACKEND_PY%"
) else (
  set "VENV_PY=%BACKEND%\.venv\Scripts\python.exe"
)
set "INSTALL_ASR=0"
set "INSTALL_SEPARATION=0"
set "INSTALL_DEEPFILTER=0"
set "INSTALL_CLEARVOICE=0"
set "START_APP=1"
set "USAGE_EXIT=1"
set "DOWNLOAD_MODELS=0"

:parse_args
if "%~1"=="" goto after_args
if "%~1"=="--no-start" (
  set "START_APP=0"
  shift
  goto parse_args
)
if "%~1"=="--with-asr" (
  set "INSTALL_ASR=1"
  shift
  goto parse_args
)
if "%~1"=="--with-separation" (
  set "INSTALL_SEPARATION=1"
  shift
  goto parse_args
)
if "%~1"=="--with-deepfilter" (
  set "INSTALL_DEEPFILTER=1"
  shift
  goto parse_args
)
if "%~1"=="--with-clearvoice" (
  set "INSTALL_CLEARVOICE=1"
  shift
  goto parse_args
)
if "%~1"=="--download-models" (
  set "DOWNLOAD_MODELS=1"
  shift
  goto parse_args
)
if "%~1"=="--full" (
  set "INSTALL_ASR=1"
  set "INSTALL_SEPARATION=1"
  set "INSTALL_DEEPFILTER=1"
  set "INSTALL_CLEARVOICE=1"
  shift
  goto parse_args
)
if "%~1"=="-h" (
  set "USAGE_EXIT=0"
  goto usage
)
if "%~1"=="--help" (
  set "USAGE_EXIT=0"
  goto usage
)
echo [ERROR] Unknown option: %~1
goto usage

:usage
echo Usage: install_project.cmd [options]
echo.
echo Options:
echo   --no-start          Install dependencies only, do not start the app.
echo   --with-asr          Install faster-whisper for real ASR.
echo   --with-separation   Install SpeechBrain separation; keep CUDA PyTorch if available.
echo   --with-deepfilter   Install DeepFilterNet CLI for real denoising.
echo   --with-clearvoice   Install ClearVoice FRCRN/MossFormer2 optional models.
echo   --download-models    Download/warm up selected optional models after installing dependencies.
echo   --full              Install all optional model dependencies.
exit /b %USAGE_EXIT%

:after_args
echo [1/5] Preparing backend virtual environment...
if not exist "%CONDA_BACKEND_PY%" if not exist "%VENV_PY%" (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.11 -m venv "%BACKEND%\.venv" 2>nul || py -3 -m venv "%BACKEND%\.venv"
  ) else (
    python -m venv "%BACKEND%\.venv"
  )
)

if not exist "%VENV_PY%" (
  echo [ERROR] Could not create backend virtual environment.
  echo Please install Python 3.10+ and rerun this script.
  exit /b 1
)

echo [2/5] Installing backend dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install -r "%BACKEND%\requirements.txt"
if errorlevel 1 exit /b 1

if "%INSTALL_ASR%"=="1" (
  echo [2/5] Installing optional ASR dependencies...
  "%VENV_PY%" -m pip install -r "%BACKEND%\requirements-asr.txt"
  if errorlevel 1 exit /b 1
)

if "%INSTALL_SEPARATION%"=="1" (
  echo [2/5] Installing optional speech separation dependencies...
  "%VENV_PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
  if errorlevel 1 (
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 exit /b 1
  ) else (
    echo [2/5] CUDA PyTorch already available; keeping existing GPU build.
  )
  "%VENV_PY%" -m pip install -r "%BACKEND%\requirements-separation.txt"
  if errorlevel 1 exit /b 1
)

if "%INSTALL_DEEPFILTER%"=="1" (
  echo [2/5] Installing optional DeepFilterNet dependency...
  "%VENV_PY%" -m pip install deepfilternet
  if errorlevel 1 exit /b 1
)

if "%INSTALL_CLEARVOICE%"=="1" (
  echo [2/5] Installing optional ClearVoice dependency...
  "%VENV_PY%" -m pip install clearvoice
  if errorlevel 1 exit /b 1
  "%VENV_PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [2/5] Restoring CUDA PyTorch after ClearVoice dependency resolution...
    "%VENV_PY%" -m pip install --force-reinstall torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
    "%VENV_PY%" -m pip install --force-reinstall torchvision==0.26.0 --no-deps --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
  )
)

if "%DOWNLOAD_MODELS%"=="1" (
  set "MODEL_ARGS="
  if "%INSTALL_ASR%"=="1" set "MODEL_ARGS=!MODEL_ARGS! --asr"
  if "%INSTALL_SEPARATION%"=="1" set "MODEL_ARGS=!MODEL_ARGS! --separation"
  if "%INSTALL_DEEPFILTER%"=="1" set "MODEL_ARGS=!MODEL_ARGS! --deepfilter"
  if "!MODEL_ARGS!"=="" (
    echo [WARN] --download-models was set, but no optional model dependency was selected.
    echo [WARN] Use --with-asr, --with-separation, --with-deepfilter, or --full.
  ) else (
    echo [2/5] Downloading/warming selected model weights...
    "%VENV_PY%" "%ROOT%scripts\download_models.py" !MODEL_ARGS!
    if errorlevel 1 exit /b 1
  )
)

echo [3/5] Installing frontend dependencies...
cd /d "%FRONTEND%"
if exist package-lock.json (
  npm.cmd ci
) else (
  npm.cmd install
)
if errorlevel 1 exit /b 1

cd /d "%ROOT%"

if "%START_APP%"=="0" (
  echo [DONE] Dependencies installed. Start later with: start_project.cmd
  exit /b 0
)

echo [4/5] Cleaning old backend/frontend processes...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_ports.ps1"

echo [5/5] Starting project...
call "%ROOT%start_project.cmd"
