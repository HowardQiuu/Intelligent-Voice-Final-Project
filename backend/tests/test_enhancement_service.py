from __future__ import annotations

import os
import sys
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import enhancement_service  # noqa: E402
from app.services.audio_service import UPLOAD_DIR  # noqa: E402


class EnhancementServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        enhancement_service._DEEPFILTER_SOURCE_CACHE.clear()
        enhancement_service._CLEARVOICE_ENHANCER_CACHE.clear()
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()

    def test_long_upload_uses_chunked_deepfilternet(self) -> None:
        path = UPLOAD_DIR / "long_upload.wav"
        enhanced = UPLOAD_DIR / "long_upload_deepfilter_chunked.wav"
        loudness = UPLOAD_DIR / "long_upload_deepfilter_chunked_loudness.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 8000 * 120)

        try:
            with patch.dict(os.environ, {"ENHANCEMENT_MAX_SECONDS": "60", "ENHANCEMENT_CANDIDATES": "deepfilternet"}, clear=True):
                with patch(
                    "app.services.enhancement_service.denoise_audio_in_chunks",
                    return_value=(enhanced, "DeepFilterNet chunked denoise (2 chunks x 60s)"),
                ) as chunked_mock:
                    with patch(
                        "app.services.enhancement_service.postprocess_enhanced_audio",
                        return_value=(loudness, "ok"),
                    ) as postprocess_mock:
                        result = enhancement_service.enhance_uploaded_audio(path)
        finally:
            path.unlink(missing_ok=True)

        chunked_mock.assert_called_once()
        postprocess_mock.assert_called_once_with(enhanced)
        self.assertEqual(result["original_audio_url"], "/static/uploads/long_upload.wav")
        self.assertEqual(result["enhanced_audio_url"], "/static/uploads/long_upload_deepfilter_chunked_loudness.wav")
        self.assertIn("DeepFilterNet chunked denoise", result["method"])
        self.assertEqual(result["metrics"]["响度处理状态"], "ok")

    def test_short_upload_postprocesses_deepfilternet_output(self) -> None:
        path = UPLOAD_DIR / "short_upload.wav"
        enhanced = UPLOAD_DIR / "short_upload_deepfilter.wav"
        loudness = UPLOAD_DIR / "short_upload_deepfilter_loudness.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 8000)

        try:
            with patch.dict(os.environ, {"ENHANCEMENT_CANDIDATES": "deepfilternet"}, clear=True):
                with patch(
                    "app.services.enhancement_service.apply_audibility_pregain",
                    return_value=(path, "skipped", {"quality_pregain_status": "skipped"}),
                ):
                    with patch(
                        "app.services.enhancement_service.denoise_audio",
                        return_value=(enhanced, "DeepFilterNet denoise"),
                    ) as denoise_mock:
                        with patch(
                            "app.services.enhancement_service.postprocess_enhanced_audio",
                            return_value=(loudness, "ok"),
                        ) as postprocess_mock:
                            result = enhancement_service.enhance_uploaded_audio(path)
        finally:
            path.unlink(missing_ok=True)

        denoise_mock.assert_called_once_with(path)
        postprocess_mock.assert_called_once_with(enhanced)
        self.assertEqual(result["enhanced_audio_url"], "/static/uploads/short_upload_deepfilter_loudness.wav")
        self.assertIn("loudness normalization", result["method"])
        self.assertEqual(result["metrics"]["增强后响度处理"], "loudnorm(-18 LUFS) + limiter")

    def test_postprocess_enhanced_audio_falls_back_when_filter_fails(self) -> None:
        path = UPLOAD_DIR / "enhanced_for_loudness.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-real-wav")

        try:
            with patch("app.services.enhancement_service.apply_audio_filter", return_value=False) as filter_mock:
                output_path, status = enhancement_service.postprocess_enhanced_audio(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(output_path, path)
        self.assertEqual(status, "fallback")
        filter_mock.assert_called_once()

    def test_should_skip_enhancement_is_false_for_long_audio(self) -> None:
        path = UPLOAD_DIR / "long_upload_skip_guard.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 8000 * 120)

        try:
            with patch.dict(os.environ, {"ENHANCEMENT_MAX_SECONDS": "60"}, clear=True):
                self.assertFalse(enhancement_service.should_skip_enhancement(path))
        finally:
            path.unlink(missing_ok=True)

    def test_skip_backend_reuses_input_audio(self) -> None:
        path = UPLOAD_DIR / "skip_backend_upload.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")

        try:
            with patch.dict(os.environ, {"DEEPFILTERNET_BACKEND": "off"}, clear=True):
                with patch("app.services.enhancement_service.denoise_audio") as denoise_mock:
                    result = enhancement_service.enhance_uploaded_audio(path)
        finally:
            path.unlink(missing_ok=True)

        denoise_mock.assert_not_called()
        self.assertEqual(result["original_audio_url"], "/static/uploads/skip_backend_upload.wav")
        self.assertEqual(result["enhanced_audio_url"], "/static/uploads/skip_backend_upload.wav")
        self.assertIn("Enhancement skipped", result["method"])

    def test_cli_chunk_denoise_runs_parallel_and_preserves_concat_order(self) -> None:
        source = UPLOAD_DIR / "parallel_source.wav"
        chunks = [UPLOAD_DIR / f"parallel_chunk_{index}.wav" for index in range(3)]
        outputs = [UPLOAD_DIR / f"parallel_chunk_{index}_deepfilter.wav" for index in range(3)]
        concat_inputs: list[Path] = []
        for path in [source, *chunks]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")

        def fake_denoise(chunk_path: Path):
            index = chunks.index(chunk_path)
            outputs[index].write_bytes(f"out-{index}".encode("utf-8"))
            return outputs[index], "DeepFilterNet denoise"

        def fake_concat(chunk_paths: list[Path], output_path: Path) -> None:
            concat_inputs.extend(chunk_paths)
            output_path.write_bytes(b"joined")

        try:
            with patch.dict(os.environ, {"ENHANCEMENT_WORKERS": "2", "DEEPFILTERNET_BACKEND": "cli"}, clear=True):
                with patch("app.services.enhancement_service._split_audio_to_chunks", return_value=chunks):
                    with patch("app.services.enhancement_service.denoise_audio", side_effect=fake_denoise):
                        with patch("app.services.enhancement_service._concat_audio_chunks", side_effect=fake_concat):
                            output_path, method = enhancement_service.denoise_audio_in_chunks(source, 180.0)
        finally:
            for path in [source, *chunks, *outputs, UPLOAD_DIR / "parallel_source_deepfilter_chunked.wav"]:
                path.unlink(missing_ok=True)

        self.assertEqual(concat_inputs, outputs)
        self.assertEqual(output_path.name, "parallel_source_deepfilter_chunked.wav")
        self.assertIn("2 workers", method)

    def test_enhancement_workers_invalid_value_uses_default(self) -> None:
        with patch.dict(os.environ, {"ENHANCEMENT_WORKERS": "not-a-number"}, clear=True):
            self.assertEqual(enhancement_service._get_enhancement_workers(), 2)

    def test_source_backend_uses_sequential_chunk_denoise_even_with_workers(self) -> None:
        chunks = [UPLOAD_DIR / "source_seq_chunk_1.wav", UPLOAD_DIR / "source_seq_chunk_2.wav"]
        outputs = [UPLOAD_DIR / "source_seq_chunk_1_out.wav", UPLOAD_DIR / "source_seq_chunk_2_out.wav"]
        calls: list[Path] = []

        def fake_denoise(chunk_path: Path):
            calls.append(chunk_path)
            return outputs[len(calls) - 1], "DeepFilterNet source inference"

        with patch.dict(os.environ, {"DEEPFILTERNET_BACKEND": "source", "ENHANCEMENT_WORKERS": "2"}, clear=True):
            with patch("app.services.enhancement_service.denoise_audio", side_effect=fake_denoise):
                result = enhancement_service._denoise_chunk_paths(chunks)

        self.assertEqual(calls, chunks)
        self.assertEqual(result, outputs)

    def test_source_backend_caches_initialized_model(self) -> None:
        fake_enhance_module = _FakeEnhanceModule()
        fake_torch = _FakeTorchModule()
        source = UPLOAD_DIR / "source_cache_input.wav"
        second = UPLOAD_DIR / "source_cache_second.wav"
        source.write_bytes(b"wav")
        second.write_bytes(b"wav")

        def fake_import_module(name: str):
            if name == "df.enhance":
                return fake_enhance_module
            if name == "torch":
                return fake_torch
            raise ImportError(name)

        try:
            with patch("app.services.enhancement_service._resolve_deepfilternet_source_dir", return_value=Path()):
                with patch("app.services.enhancement_service._resolve_deepfilternet_model_dir", return_value=None):
                    with patch.object(enhancement_service.importlib, "import_module", side_effect=fake_import_module):
                        first_output, _ = enhancement_service.denoise_audio_with_source(source)
                        second_output, _ = enhancement_service.denoise_audio_with_source(second)
        finally:
            for path in [
                source,
                second,
                UPLOAD_DIR / "source_cache_input_deepfilternet_source.wav",
                UPLOAD_DIR / "source_cache_second_deepfilternet_source.wav",
            ]:
                path.unlink(missing_ok=True)

        self.assertEqual(fake_enhance_module.init_calls, 1)
        self.assertTrue(first_output.name.endswith("_deepfilternet_source.wav"))
        self.assertTrue(second_output.name.endswith("_deepfilternet_source.wav"))

    def test_source_backend_uses_different_cache_for_different_model_dirs(self) -> None:
        fake_enhance_module = _FakeEnhanceModule()
        fake_torch = _FakeTorchModule()
        model_a = UPLOAD_DIR / "df_model_a"
        model_b = UPLOAD_DIR / "df_model_b"
        source = UPLOAD_DIR / "source_cache_model_input.wav"
        for model_dir in [model_a, model_b]:
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.ini").write_text("fake", encoding="utf-8")
        source.write_bytes(b"wav")

        def fake_import_module(name: str):
            if name == "df.enhance":
                return fake_enhance_module
            if name == "torch":
                return fake_torch
            raise ImportError(name)

        try:
            with patch("app.services.enhancement_service._resolve_deepfilternet_source_dir", return_value=Path()):
                with patch.object(enhancement_service.importlib, "import_module", side_effect=fake_import_module):
                    with patch.dict(os.environ, {"DEEPFILTERNET_MODEL_DIR": str(model_a)}, clear=True):
                        enhancement_service.denoise_audio_with_source(source)
                    with patch.dict(os.environ, {"DEEPFILTERNET_MODEL_DIR": str(model_b)}, clear=True):
                        enhancement_service.denoise_audio_with_source(source)
        finally:
            source.unlink(missing_ok=True)
            (UPLOAD_DIR / "source_cache_model_input_deepfilternet_source.wav").unlink(missing_ok=True)
            for model_dir in [model_a, model_b]:
                (model_dir / "config.ini").unlink(missing_ok=True)
                model_dir.rmdir()

        self.assertEqual(fake_enhance_module.init_calls, 2)

    def test_quality_router_skips_unconfigured_candidate_and_selects_deepfilternet(self) -> None:
        source = UPLOAD_DIR / "router_source.wav"
        enhanced = UPLOAD_DIR / "router_source_deepfilter.wav"
        loudness = UPLOAD_DIR / "router_source_deepfilter_loudness.wav"
        source.write_bytes(b"wav")
        enhanced.write_bytes(b"enhanced")
        loudness.write_bytes(b"loud")

        try:
            with patch.dict(
                os.environ,
                {"ENHANCEMENT_CANDIDATES": "clearvoice,deepfilternet", "QUALITY_ROUTER_ENABLED": "true"},
                clear=True,
            ):
                with patch(
                    "app.services.enhancement_service.apply_audibility_pregain",
                    return_value=(source, "skipped", {"quality_pregain_status": "skipped"}),
                ):
                    with patch(
                        "app.services.enhancement_service._run_clearvoice_enhancement_candidate",
                        side_effect=RuntimeError("not installed"),
                    ):
                        with patch(
                            "app.services.enhancement_service.denoise_audio",
                            return_value=(enhanced, "DeepFilterNet denoise"),
                        ):
                            with patch(
                                "app.services.enhancement_service.postprocess_enhanced_audio",
                                return_value=(loudness, "ok"),
                            ):
                                result = enhancement_service.enhance_uploaded_audio(source)
        finally:
            for path in [source, enhanced, loudness]:
                path.unlink(missing_ok=True)

        self.assertEqual(result["enhanced_audio_url"], "/static/uploads/router_source_deepfilter_loudness.wav")
        self.assertEqual(result["metrics"]["quality_router_selected_enhancement"], "deepfilternet")
        self.assertIn("clearvoice=skipped", result["metrics"]["quality_router_enhancement_candidates"])

    def test_clearvoice_candidate_uses_native_api_output(self) -> None:
        source = UPLOAD_DIR / "clearvoice_native_input.wav"
        produced_dir = UPLOAD_DIR / "clearvoice_fake_model"
        output_path = UPLOAD_DIR / "clearvoice_native_input_mossformer2_se_48k_clearvoice.wav"
        source.write_bytes(b"wav")

        class FakeClearVoice:
            def __init__(self, task, model_names):
                self.task = task
                self.model_names = model_names

            def __call__(self, input_path, online_write=False, output_path=None):
                produced = Path(output_path) / "MossFormer2_SE_48K" / "clearvoice_native_input.wav"
                produced.parent.mkdir(parents=True, exist_ok=True)
                produced.write_bytes(b"enhanced")

        fake_module = SimpleNamespace(ClearVoice=FakeClearVoice)
        try:
            with patch.object(enhancement_service.importlib, "import_module", return_value=fake_module):
                output_path, method = enhancement_service._run_clearvoice_enhancement_candidate(source)
        finally:
            source.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            if produced_dir.exists():
                for child in produced_dir.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(produced_dir.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                produced_dir.rmdir()

        self.assertTrue(output_path.name.endswith("_mossformer2_se_48k_clearvoice.wav"))
        self.assertIn("ClearVoice MossFormer2_SE_48K", method)

class _FakeAudio:
    def to(self, _device: str):
        return self


class _FakeDfState:
    def sr(self) -> int:
        return 48000


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTorchModule:
    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeEnhanceModule:
    def __init__(self) -> None:
        self.init_calls = 0

    def init_df(self, model_base_dir=None, log_file=None):
        self.init_calls += 1
        return SimpleNamespace(name="model"), _FakeDfState()

    def load_audio(self, path: str, sr: int, device: str):
        return _FakeAudio(), sr

    def enhance(self, model, df_state, audio):
        return _FakeAudio()

    def save_audio(self, path: str, audio, sr: int, output_dir: str, suffix: str, log: bool):
        output = Path(output_dir) / f"{Path(path).stem}_{suffix}.wav"
        output.write_bytes(b"enhanced")


if __name__ == "__main__":
    unittest.main()
