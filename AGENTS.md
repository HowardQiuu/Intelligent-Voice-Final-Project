# Repository Guidelines

## Project Structure & Module Organization

This repository is a smart meeting voice-processing demo with a Python FastAPI backend and React/Vite frontend.

- `backend/app/main.py` defines the FastAPI entrypoint and routes.
- `backend/app/models.py` contains API data models.
- `backend/app/services/` holds audio normalization, enhancement, chunking, ASR, separation, transcript grouping, summaries, diagnostics, and cache logic.
- `backend/app/data/` and `backend/app/static/audio/` contain demo data and audio assets.
- `backend/tests/` contains backend unit tests and smoke checks.
- `frontend/src/App.jsx`, `frontend/src/api.js`, and `frontend/src/components/` implement the browser UI.
- `docs/` contains setup and architecture notes; `scripts/` contains launch, port cleanup, and model download helpers.

## Build, Test, and Development Commands

- `.\install_project.cmd` installs dependencies and starts both services on Windows.
- `bash install.sh` does the same for macOS/Linux/Git Bash.
- `.\start_project.cmd` starts backend `http://127.0.0.1:8000` and frontend `http://127.0.0.1:5173`.
- `.\stop_project.cmd` stops the local project services.
- `cd frontend && npm run dev` starts only the Vite frontend.
- `cd frontend && npm run build` creates a production frontend build.
- `cd backend && .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000` starts only the backend.
- `cd backend && .\.venv\Scripts\python.exe -m unittest discover tests` runs the backend test suite.

## Coding Style & Naming Conventions

Use 4-space indentation in Python and keep service functions focused on one pipeline responsibility. Name Python modules and tests with `snake_case`, for example `summary_service.py` and `test_summary_service.py`. React uses ES modules, function components, hooks, and PascalCase component files such as `AudioCompare.jsx`. Keep shared API calls in `frontend/src/api.js`.

## Testing Guidelines

Backend tests use Python `unittest` with selective mocking for external tools such as FFmpeg, ASR, model inference, and LLM calls. Add or update tests in `backend/tests/` when service behavior, fallback logic, request handling, or diagnostics change. Prefer deterministic fixtures and bundled demo data over live model or network calls.

## Commit & Pull Request Guidelines

Recent history uses concise imperative commits, sometimes with conventional prefixes such as `docs:` and `feat:`. Keep commits scoped, for example `feat: surface diarization diagnostics` or `docs: update setup notes`. Pull requests should describe the user-visible change, list tested commands, mention required model or environment setup, and include screenshots for UI changes.

## Security & Configuration Tips

Copy configuration from `backend/.env.example`; do not commit real `.env` secrets or API keys. Optional ASR, separation, DeepFilterNet, and model downloads can be large, so document new dependencies in `README.md` or `docs/` and keep graceful fallback behavior intact.
