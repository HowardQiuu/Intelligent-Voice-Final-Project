from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or warm up optional local model weights.")
    parser.add_argument("--asr", action="store_true", help="Download faster-whisper model weights.")
    parser.add_argument("--separation", action="store_true", help="Download SpeechBrain SepFormer weights.")
    parser.add_argument("--deepfilter", action="store_true", help="Warm up DeepFilterNet CLI if installed.")
    parser.add_argument("--all", action="store_true", help="Run all model download/warmup steps.")
    parser.add_argument("--asr-model", default=os.getenv("ASR_MODEL", "small"), help="faster-whisper model name.")
    parser.add_argument(
        "--separation-model",
        default=os.getenv("SEPARATION_MODEL", "speechbrain/sepformer-wsj02mix"),
        help="SpeechBrain model name.",
    )
    args = parser.parse_args()

    if args.all:
        args.asr = True
        args.separation = True
        args.deepfilter = True
    if not (args.asr or args.separation or args.deepfilter):
        parser.error("Choose at least one of --asr, --separation, --deepfilter, or --all.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ok = True
    if args.asr:
        ok = download_asr_model(args.asr_model) and ok
    if args.separation:
        ok = download_separation_model(args.separation_model) and ok
    if args.deepfilter:
        ok = warm_deepfilter() and ok
    return 0 if ok else 1


def download_asr_model(model_name: str) -> bool:
    print(f"[ASR] Preparing faster-whisper model: {model_name}")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ASR][ERROR] faster-whisper is not installed. Run install_project.cmd --with-asr first.")
        return False

    download_root = MODELS_DIR / "faster-whisper" / safe_model_dir_name(model_name)
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(download_root),
        )
    except TypeError:
        WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        print(f"[ASR][ERROR] Failed to prepare {model_name}: {exc}")
        return False

    print(f"[ASR] Ready. Project cache: {download_root}")
    print("[ASR] The app will prefer this project-local cache when ASR_MODEL uses the same model name.")
    return True


def download_separation_model(model_name: str) -> bool:
    print(f"[Separation] Preparing SpeechBrain model: {model_name}")
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError:
        print("[Separation][ERROR] speechbrain/torch/torchaudio are not installed.")
        print("[Separation][ERROR] Run install_project.cmd --with-separation first.")
        return False

    savedir = MODELS_DIR / "speechbrain" / safe_model_dir_name(model_name)
    savedir.mkdir(parents=True, exist_ok=True)
    try:
        SepformerSeparation.from_hparams(
            source=model_name,
            savedir=str(savedir),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )
    except Exception as exc:
        print(f"[Separation][ERROR] Failed to prepare {model_name}: {exc}")
        return False

    print(f"[Separation] Ready. Project cache: {savedir}")
    return True


def warm_deepfilter() -> bool:
    print("[DeepFilterNet] Warming up CLI backend.")
    deepfilter = shutil.which("deepFilter") or shutil.which("deep-filter")
    if not deepfilter:
        print("[DeepFilterNet][ERROR] deepFilter/deep-filter command was not found.")
        print("[DeepFilterNet][ERROR] Run install_project.cmd --with-deepfilter first.")
        return False

    work_dir = MODELS_DIR / "deepfilternet-warmup"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_wav = work_dir / "silence.wav"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_silence_wav(input_wav)

    command = [deepfilter, str(input_wav), "-o", str(output_dir)]
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"[DeepFilterNet][ERROR] Warmup failed with exit code {exc.returncode}.")
        return False
    except OSError as exc:
        print(f"[DeepFilterNet][ERROR] Warmup failed: {exc}")
        return False

    print("[DeepFilterNet] CLI warmup finished.")
    print("[DeepFilterNet] Source backend still requires DEEPFILTERNET_SOURCE_DIR and DEEPFILTERNET_MODEL_DIR.")
    return True


def write_silence_wav(path: Path) -> None:
    sample_rate = 16000
    samples = sample_rate
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * samples)


def safe_model_dir_name(model_name: str) -> str:
    return model_name.strip().replace("\\", "/").split("/")[-1] or "model"


if __name__ == "__main__":
    raise SystemExit(main())
