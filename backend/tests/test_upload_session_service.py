from __future__ import annotations

import asyncio
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.upload_session_service import (  # noqa: E402
    complete_upload_session,
    create_upload_session,
    save_upload_chunk,
)


class UploadSessionServiceTest(unittest.TestCase):
    def test_chunked_upload_merges_parts_on_disk(self) -> None:
        first = b"a" * (1024 * 1024)
        second = b"b" * 7
        output_path = None
        with patch.dict(os.environ, {"UPLOAD_CHUNK_MB": "1"}, clear=True):
            session = create_upload_session("meeting.wav", len(first) + len(second), ".wav")
            try:
                self.assertEqual(session["total_chunks"], 2)
                asyncio.run(save_upload_chunk(session["upload_id"], 0, _upload_file(first)))
                asyncio.run(save_upload_chunk(session["upload_id"], 1, _upload_file(second)))

                output_path, display_name = complete_upload_session(session["upload_id"], 2)

                self.assertEqual(display_name, "meeting.wav")
                self.assertEqual(output_path.read_bytes(), first + second)
            finally:
                if output_path:
                    output_path.unlink(missing_ok=True)


def _upload_file(content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename="chunk.part")


if __name__ == "__main__":
    unittest.main()
