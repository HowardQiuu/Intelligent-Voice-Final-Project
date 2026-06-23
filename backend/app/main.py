from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import (
    LocalFileRequest,
    ProcessResult,
    UploadSessionCompleteRequest,
    UploadSessionCreateRequest,
    UploadSessionResponse,
)
from .services.audio_service import UPLOAD_DIR, ensure_audio_dirs, ensure_demo_audios
from .services.demo_cache import get_case, load_demo_cases
from .services.enhancement_service import enhance_demo_audio
from .services.pipeline_service import (
    process_audio_path,
    process_demo_case,
    save_upload_stream,
    separate_uploaded_path,
    stage_local_file,
)
from .services.separation_service import separate_demo_audio
from .services.upload_session_service import (
    complete_upload_session,
    create_upload_session,
    save_upload_chunk,
)


ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

app = FastAPI(title="Smart Meeting Speech Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_audio_dirs()
ensure_demo_audios([case["id"] for case in load_demo_cases()])
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "demo-cache"}


@app.get("/api/demo-cases")
def demo_cases() -> list[dict]:
    return load_demo_cases()


@app.post("/api/process-demo/{case_id}", response_model=ProcessResult)
def process_demo(case_id: str) -> ProcessResult:
    try:
        return process_demo_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Demo case not found") from exc


@app.post("/api/separate-demo/{case_id}")
def separate_demo(case_id: str) -> dict:
    try:
        get_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Demo case not found") from exc

    audio = enhance_demo_audio(case_id)
    separation = separate_demo_audio(case_id, audio["enhanced_audio_url"])
    return {
        "case_id": case_id,
        "enhanced_audio_url": audio["enhanced_audio_url"],
        "separation": separation,
    }


@app.post("/api/upload", response_model=ProcessResult)
async def upload_audio(file: UploadFile = File(...)) -> ProcessResult:
    suffix = _validate_audio_suffix(file.filename or "meeting.wav")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    await save_upload_stream(file, raw_path)
    return process_audio_path(raw_path, file.filename or raw_path.name, case_id="upload")


@app.post("/api/upload-session", response_model=UploadSessionResponse)
def create_chunked_upload(request: UploadSessionCreateRequest) -> UploadSessionResponse:
    suffix = _validate_audio_suffix(request.filename)
    session = create_upload_session(request.filename, request.size_bytes, suffix)
    return UploadSessionResponse(**session)


@app.post("/api/upload-session/{upload_id}/chunk")
async def upload_chunk(upload_id: str, index: int, file: UploadFile = File(...)) -> dict:
    try:
        return await save_upload_chunk(upload_id, index, file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Upload session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload-session/{upload_id}/complete", response_model=ProcessResult)
def complete_chunked_upload(upload_id: str, request: UploadSessionCompleteRequest) -> ProcessResult:
    try:
        raw_path, display_name = complete_upload_session(upload_id, request.total_chunks)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Upload session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return process_audio_path(raw_path, request.filename or display_name, case_id="upload")


@app.post("/api/process-local-file", response_model=ProcessResult)
def process_local_file(request: LocalFileRequest) -> ProcessResult:
    source_path = Path(request.path).expanduser()
    _validate_local_audio_file(source_path)
    staged_path = stage_local_file(source_path)
    return process_audio_path(staged_path, source_path.name, case_id="local-file")


@app.post("/api/separate-upload")
async def separate_upload(file: UploadFile = File(...)) -> dict:
    suffix = _validate_audio_suffix(file.filename or "meeting.wav")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    await save_upload_stream(file, raw_path)
    return separate_uploaded_path(raw_path, file.filename or raw_path.name)


def _validate_audio_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    return suffix


def _validate_local_audio_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Local audio file not found")
    _validate_audio_suffix(path.name)
