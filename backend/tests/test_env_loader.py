from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.env_loader import load_backend_env  # noqa: E402


class EnvLoaderTest(unittest.TestCase):
    def test_load_backend_env_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_dir = Path(temp_dir)
            (backend_dir / ".env").write_text(
                "LLM_API_KEY=test-key\nLLM_MODEL=deepseek-chat\n",
                encoding="utf-8-sig",
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_backend_env(backend_dir))
                self.assertEqual(os.getenv("LLM_API_KEY"), "test-key")
                self.assertIsNone(os.getenv("\ufeffLLM_API_KEY"))
                self.assertEqual(os.getenv("LLM_MODEL"), "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
