#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
RUNTIME_DIR="$ROOT/.runtime"

WITH_ASR=0
WITH_SEPARATION=0
WITH_DEEPFILTER=0
START_APP=1
DOWNLOAD_MODELS=0

usage() {
  cat <<'EOF'
Usage: bash install.sh [options]

Options:
  --no-start          Install dependencies only, do not start the app.
  --with-asr          Install faster-whisper for real ASR.
  --with-separation   Install PyTorch CPU wheels and SpeechBrain separation.
  --with-deepfilter   Install DeepFilterNet CLI for real denoising.
  --download-models    Download/warm up selected optional models after installing dependencies.
  --full              Install all optional model dependencies.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start)
      START_APP=0
      ;;
    --with-asr)
      WITH_ASR=1
      ;;
    --with-separation)
      WITH_SEPARATION=1
      ;;
    --with-deepfilter)
      WITH_DEEPFILTER=1
      ;;
    --download-models)
      DOWNLOAD_MODELS=1
      ;;
    --full)
      WITH_ASR=1
      WITH_SEPARATION=1
      WITH_DEEPFILTER=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
  elif command_exists python3; then
    echo "python3"
  elif command_exists python; then
    echo "python"
  else
    echo "[ERROR] Python 3.10+ was not found. Install Python, then rerun this script." >&2
    exit 1
  fi
}

find_venv_python() {
  if [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
    echo "$VENV_DIR/Scripts/python.exe"
  elif [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "$VENV_DIR/bin/python"
  else
    echo "[ERROR] Could not find Python inside $VENV_DIR" >&2
    exit 1
  fi
}

find_npm() {
  if command_exists npm; then
    echo "npm"
  elif command_exists npm.cmd; then
    echo "npm.cmd"
  else
    echo "[ERROR] npm was not found. Install Node.js 18+, then rerun this script." >&2
    exit 1
  fi
}

prepend_venv_path() {
  if [[ -d "$VENV_DIR/Scripts" ]]; then
    export PATH="$VENV_DIR/Scripts:$PATH"
  elif [[ -d "$VENV_DIR/bin" ]]; then
    export PATH="$VENV_DIR/bin:$PATH"
  fi
}

stop_ports() {
  echo "[4/6] Cleaning old listeners on ports 8000 and 5173..."
  if command_exists powershell.exe && [[ -f "$ROOT/scripts/stop_ports.ps1" ]]; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT/scripts/stop_ports.ps1" >/dev/null || true
    return
  fi

  for port in 8000 5173; do
    if command_exists lsof; then
      while IFS= read -r pid; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
      done < <(lsof -ti tcp:"$port" 2>/dev/null || true)
    elif command_exists fuser; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
  done
}

mkdir -p "$RUNTIME_DIR"

PYTHON_BIN="$(find_python)"
NPM_BIN="$(find_npm)"

echo "[1/6] Preparing backend virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$(find_venv_python)"
prepend_venv_path

echo "[2/6] Installing backend dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$BACKEND_DIR/requirements.txt"

if [[ "$WITH_ASR" -eq 1 ]]; then
  echo "[2/6] Installing optional ASR dependencies..."
  "$VENV_PYTHON" -m pip install -r "$BACKEND_DIR/requirements-asr.txt"
fi

if [[ "$WITH_SEPARATION" -eq 1 ]]; then
  echo "[2/6] Installing optional speech separation dependencies..."
  "$VENV_PYTHON" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  "$VENV_PYTHON" -m pip install -r "$BACKEND_DIR/requirements-separation.txt"
fi

if [[ "$WITH_DEEPFILTER" -eq 1 ]]; then
  echo "[2/6] Installing optional DeepFilterNet dependency..."
  "$VENV_PYTHON" -m pip install deepfilternet
fi

if [[ "$DOWNLOAD_MODELS" -eq 1 ]]; then
  model_args=()
  [[ "$WITH_ASR" -eq 1 ]] && model_args+=("--asr")
  [[ "$WITH_SEPARATION" -eq 1 ]] && model_args+=("--separation")
  [[ "$WITH_DEEPFILTER" -eq 1 ]] && model_args+=("--deepfilter")
  if [[ "${#model_args[@]}" -eq 0 ]]; then
    echo "[WARN] --download-models was set, but no optional model dependency was selected."
    echo "[WARN] Use --with-asr, --with-separation, --with-deepfilter, or --full."
  else
    echo "[2/6] Downloading/warming selected model weights..."
    "$VENV_PYTHON" "$ROOT/scripts/download_models.py" "${model_args[@]}"
  fi
fi

echo "[3/6] Installing frontend dependencies..."
cd "$FRONTEND_DIR"
if [[ -f package-lock.json ]]; then
  "$NPM_BIN" ci
else
  "$NPM_BIN" install
fi
cd "$ROOT"

if [[ "$START_APP" -eq 0 ]]; then
  echo "[DONE] Dependencies installed. Start later with: bash install.sh"
  exit 0
fi

stop_ports

echo "[5/6] Starting FastAPI backend at http://127.0.0.1:8000 ..."
export LLM_ENABLED="${LLM_ENABLED:-false}"
export ENHANCEMENT_MAX_SECONDS="${ENHANCEMENT_MAX_SECONDS:-300}"
export CHUNK_SECONDS="${CHUNK_SECONDS:-60}"
export CHUNK_OVERLAP_SECONDS="${CHUNK_OVERLAP_SECONDS:-5}"
if [[ "$WITH_SEPARATION" -eq 1 ]]; then
  export SEPARATION_BACKEND="${SEPARATION_BACKEND:-speechbrain}"
  export SEPARATION_DEVICE="${SEPARATION_DEVICE:-cpu}"
else
  export SEPARATION_BACKEND="${SEPARATION_BACKEND:-placeholder}"
fi

(
  cd "$BACKEND_DIR"
  "$VENV_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) >"$RUNTIME_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[6/6] Starting Vite frontend at http://127.0.0.1:5173 ..."
echo
echo "Open: http://127.0.0.1:5173"
echo "Backend log: $RUNTIME_DIR/backend.log"
echo "Press Ctrl+C to stop services started by this script."
echo

cd "$FRONTEND_DIR"
"$NPM_BIN" run dev -- --host 127.0.0.1 --port 5173
