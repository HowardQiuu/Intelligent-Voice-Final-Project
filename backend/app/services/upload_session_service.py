from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from .audio_service import UPLOAD_DIR


CHUNK_UPLOAD_DIR = UPLOAD_DIR / "chunk_uploads"
STREAM_BYTES = 1024 * 1024


def default_chunk_size_bytes() -> int:
    return int(os.getenv("UPLOAD_CHUNK_MB", "4")) * 1024 * 1024


def create_upload_session(filename: str, size_bytes: int, suffix: str) -> dict:
    chunk_size = default_chunk_size_bytes()
    total_chunks = max(1, math.ceil(size_bytes / chunk_size))
    upload_id = uuid.uuid4().hex
    session_dir = _session_dir(upload_id)
    (session_dir / "parts").mkdir(parents=True, exist_ok=False)
    _write_metadata(
        session_dir,
        {
            "upload_id": upload_id,
            "filename": Path(filename).name or f"meeting{suffix}",
            "suffix": suffix,
            "size_bytes": size_bytes,
            "chunk_size_bytes": chunk_size,
            "total_chunks": total_chunks,
            "received": [],
        },
    )
    return {
        "upload_id": upload_id,
        "chunk_size_bytes": chunk_size,
        "total_chunks": total_chunks,
    }


async def save_upload_chunk(upload_id: str, index: int, file: UploadFile) -> dict:
    session_dir = _existing_session_dir(upload_id)
    metadata = _read_metadata(session_dir)
    total_chunks = int(metadata["total_chunks"])
    if index < 0 or index >= total_chunks:
        raise ValueError("chunk index out of range")

    part_path = session_dir / "parts" / f"{index:06d}.part"
    with part_path.open("wb") as f:
        while True:
            chunk = await file.read(STREAM_BYTES)
            if not chunk:
                break
            f.write(chunk)

    received = set(int(item) for item in metadata.get("received", []))
    received.add(index)
    metadata["received"] = sorted(received)
    _write_metadata(session_dir, metadata)
    return {
        "upload_id": upload_id,
        "index": index,
        "received_chunks": len(received),
        "total_chunks": total_chunks,
    }


def complete_upload_session(upload_id: str, total_chunks: int) -> tuple[Path, str]:
    session_dir = _existing_session_dir(upload_id)
    metadata = _read_metadata(session_dir)
    expected_total = int(metadata["total_chunks"])
    if total_chunks != expected_total:
        raise ValueError("total chunks mismatch")

    parts_dir = session_dir / "parts"
    missing = [
        index
        for index in range(expected_total)
        if not (parts_dir / f"{index:06d}.part").exists()
    ]
    if missing:
        raise ValueError(f"missing chunks: {missing[:5]}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{metadata['suffix']}"
    with output_path.open("wb") as output:
        for index in range(expected_total):
            part_path = parts_dir / f"{index:06d}.part"
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output, length=STREAM_BYTES)

    display_name = str(metadata.get("filename") or output_path.name)
    shutil.rmtree(session_dir, ignore_errors=True)
    return output_path, display_name


def _session_dir(upload_id: str) -> Path:
    if not upload_id or any(char not in "0123456789abcdef" for char in upload_id):
        raise ValueError("invalid upload id")
    return CHUNK_UPLOAD_DIR / upload_id


def _existing_session_dir(upload_id: str) -> Path:
    session_dir = _session_dir(upload_id)
    if not session_dir.exists() or not session_dir.is_dir():
        raise FileNotFoundError("upload session not found")
    return session_dir


def _metadata_path(session_dir: Path) -> Path:
    return session_dir / "metadata.json"


def _read_metadata(session_dir: Path) -> dict:
    return json.loads(_metadata_path(session_dir).read_text(encoding="utf-8"))


def _write_metadata(session_dir: Path, metadata: dict) -> None:
    _metadata_path(session_dir).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
