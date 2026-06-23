from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import ProcessResult
from .services.asr_service import build_pipeline_steps, fallback_upload_result
from .services.audio_service import UPLOAD_DIR, audio_url, ensure_audio_dirs, ensure_demo_audios, normalize_upload
from .services.demo_cache import get_case, get_result, load_demo_cases
from .services.enhancement_service import enhance_demo_audio, enhance_uploaded_audio
from .services.summary_service import fallback_summary, generate_summary


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
        case = get_case(case_id)
        cached = get_result(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Demo case not found") from exc

    audio = enhance_demo_audio(case_id)
    summary_result = generate_summary(
        transcript=cached["transcript"],
        case_name=case["name"],
        enhanced_asr_text=cached["enhanced_asr_text"],
        fallback=cached["summary"],
    )
    signal_metrics = {
        **cached["signal_metrics"],
        **summary_result.metrics,
    }
    return ProcessResult(
        case_id=case_id,
        case_name=case["name"],
        original_audio_url=audio["original_audio_url"],
        enhanced_audio_url=audio["enhanced_audio_url"],
        direct_asr_text=cached["direct_asr_text"],
        enhanced_asr_text=cached["enhanced_asr_text"],
        signal_metrics=signal_metrics,
        steps=build_pipeline_steps(cache_mode=True),
        transcript=cached["transcript"],
        summary=summary_result.summary,
    )


@app.post("/api/upload", response_model=ProcessResult)
async def upload_audio(file: UploadFile = File(...)) -> ProcessResult:
    suffix = Path(file.filename or "meeting.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with raw_path.open("wb") as f:
        f.write(await file.read())

    normalized = normalize_upload(raw_path, raw_path.stem)
    try:
        audio = enhance_uploaded_audio(normalized)
    except RuntimeError as exc:
        audio = {
            "original_audio_url": audio_url(normalized),
            "enhanced_audio_url": audio_url(normalized),
            "method": f"上传增强兜底：{exc}",
        }

    fallback = fallback_upload_result(file.filename or raw_path.name)
    signal_metrics = {
        **fallback["signal_metrics"],
        "增强算法": audio["method"],
    }
    summary_result = generate_summary(
        transcript=fallback["transcript"],
        case_name=file.filename or "上传会议音频",
        enhanced_asr_text=fallback["enhanced_asr_text"],
        fallback=fallback_summary(),
    )
    signal_metrics = {
        **signal_metrics,
        **summary_result.metrics,
    }
    return ProcessResult(
        case_id="upload",
        case_name=file.filename or "上传会议音频",
        original_audio_url=audio["original_audio_url"],
        enhanced_audio_url=audio["enhanced_audio_url"],
        direct_asr_text=fallback["direct_asr_text"],
        enhanced_asr_text=fallback["enhanced_asr_text"],
        signal_metrics=signal_metrics,
        steps=build_pipeline_steps(cache_mode=False),
        transcript=fallback["transcript"],
        summary=summary_result.summary,
    )
