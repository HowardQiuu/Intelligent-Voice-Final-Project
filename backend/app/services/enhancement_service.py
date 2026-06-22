from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .audio_service import UPLOAD_DIR, audio_url, generate_demo_audio

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_VENDOR_DIR = BACKEND_DIR / "vendor" / "DeepFilterNet"
DEFAULT_SOURCE_DIR = PROJECT_VENDOR_DIR / "DeepFilterNet"
DEFAULT_MODEL_DIR = PROJECT_VENDOR_DIR / "models" / "DeepFilterNet3"


def _normalize_model_dir(model_dir: Path) -> Path:
    if (model_dir / "config.ini").exists():
        return model_dir
    nested = [p for p in model_dir.rglob("config.ini") if p.is_file()]
    if nested:
        return nested[0].parent
    return model_dir


def enhance_demo_audio(case_id: str) -> dict[str, str]:
    original = generate_demo_audio(case_id, noisy=True)
    enhanced = generate_demo_audio(case_id, noisy=False)
    return {
        "original_audio_url": audio_url(original),
        "enhanced_audio_url": audio_url(enhanced),
        "method": "Demo cached enhancement",
    }


def _resolve_deepfilternet_source_dir() -> Path:
    source_dir = Path(os.getenv("DEEPFILTERNET_SOURCE_DIR", str(DEFAULT_SOURCE_DIR))).resolve()
    if not (source_dir / "df" / "enhance.py").exists():
        if importlib.util.find_spec("df.enhance") is not None:
            return Path()
        raise RuntimeError(
            "DeepFilterNet source code not found. Set DEEPFILTERNET_SOURCE_DIR to the official "
            "DeepFilterNet/DeepFilterNet directory that contains df/enhance.py, or install the "
            "official package with: python -m pip install deepfilternet."
        )
    return source_dir


def _resolve_deepfilternet_model_dir() -> Path | None:
    configured = os.getenv("DEEPFILTERNET_MODEL_DIR")
    if configured:
        model_dir = Path(configured).resolve()
        if not model_dir.exists():
            raise RuntimeError(f"DeepFilterNet model directory not found: {model_dir}")
        return _normalize_model_dir(model_dir)

    if DEFAULT_MODEL_DIR.exists():
        return _normalize_model_dir(DEFAULT_MODEL_DIR)

    models_dir = PROJECT_VENDOR_DIR / "models"
    if models_dir.exists():
        candidates = [p for p in models_dir.iterdir() if p.is_dir() and p.name.lower().startswith("deepfilternet")]
        if candidates:
            return _normalize_model_dir(sorted(candidates, key=lambda p: p.name, reverse=True)[0])

    return None


def denoise_audio_with_source(path: Path) -> tuple[Path, str]:
    """Denoise uploaded audio through the official DeepFilterNet source API."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = _resolve_deepfilternet_source_dir()
    model_dir = _resolve_deepfilternet_model_dir()

    if source_dir and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    try:
        enhance_module = importlib.import_module("df.enhance")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "DeepFilterNet source dependencies are not available. Install the official project "
            "requirements, or install DeepFilterNet in editable mode from its source tree."
        ) from exc

    init_df = enhance_module.init_df
    load_audio = enhance_module.load_audio
    enhance = enhance_module.enhance
    save_audio = enhance_module.save_audio

    out_dir = UPLOAD_DIR / f"deepfilternet_source_{uuid.uuid4().hex[:10]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_DIR / f"{path.stem}_deepfilternet_source.wav"
    model_base_dir = str(model_dir) if model_dir else None

    try:
        init_result = init_df(model_base_dir=model_base_dir, log_file=None)
        model, df_state = init_result[0], init_result[1]
        audio, _ = load_audio(str(path), df_state.sr(), "cpu")
        with torch.no_grad():
            enhanced = enhance(model, df_state, audio)
        save_audio(
            str(path),
            enhanced.to("cpu"),
            sr=df_state.sr(),
            output_dir=str(out_dir),
            suffix="deepfilternet_source",
            log=False,
        )
    except Exception as exc:
        raise RuntimeError(f"DeepFilterNet source inference failed: {exc}") from exc

    candidates = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        shutil.copyfile(candidates[0], output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("DeepFilterNet source inference finished without producing an output WAV file")

    model_name = model_dir.name if model_dir else "default pretrained model"
    return output_path, f"DeepFilterNet source inference ({model_name})"


def denoise_audio_with_cli(path: Path) -> tuple[Path, str]:
    """Denoise uploaded audio through the official DeepFilterNet CLI."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:10]

    deepfilter_cmd = shutil.which("deepFilter") or shutil.which("deep-filter")
    if not deepfilter_cmd:
        raise RuntimeError("DeepFilterNet CLI not found. Install it with: python -m pip install deepfilternet")

    out_dir = UPLOAD_DIR / f"deepfilter_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    command_variants = [
        [deepfilter_cmd, str(path), "-o", str(out_dir)],
        [deepfilter_cmd, str(path), "--output-dir", str(out_dir)],
    ]
    errors: list[str] = []
    for cmd in command_variants:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            candidates = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                final_path = UPLOAD_DIR / f"{path.stem}_deepfilter.wav"
                shutil.copyfile(candidates[0], final_path)
                return final_path, "DeepFilterNet denoise"
            errors.append(f"{cmd[0]} finished without producing a WAV file")
        except (subprocess.CalledProcessError, OSError) as exc:
            errors.append(str(exc))

    raise RuntimeError(f"DeepFilterNet failed to enhance audio: {'; '.join(errors)}")


def denoise_audio(path: Path) -> tuple[Path, str]:
    """Denoise uploaded audio with DeepFilterNet.

    Backends:
    - cli: official deepFilter/deep-filter command. This is the default for classroom demos.
    - source: official source code + pretrained model directory.
    """
    backend = os.getenv("DEEPFILTERNET_BACKEND", "cli").strip().lower()
    if backend == "source":
        return denoise_audio_with_source(path)
    if backend == "cli":
        return denoise_audio_with_cli(path)
    raise RuntimeError("Unsupported DEEPFILTERNET_BACKEND. Use 'source' or 'cli'.")


def enhance_uploaded_audio(path: Path) -> dict[str, str]:
    denoised_path, denoise_method = denoise_audio(path)
    return {
        "original_audio_url": audio_url(path),
        "enhanced_audio_url": audio_url(denoised_path),
        "method": denoise_method,
    }
