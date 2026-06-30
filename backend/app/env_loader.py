from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_backend_env(backend_dir: Path) -> bool:
    return load_dotenv(backend_dir / ".env", encoding="utf-8-sig")
