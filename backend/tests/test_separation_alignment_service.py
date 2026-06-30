from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.audio_service import UPLOAD_DIR, audio_url  # noqa: E402
from app.services.separation_alignment_service import (  # noqa: E402
    align_transcript_to_separation_tracks,
    build_textgrid_separation_evaluation,
    evaluation_metrics,
    parse_textgrid,
    should_transcribe_evaluation_tracks,
)


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "N_SPK1"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 3.0
            text = "项目需要语音分离"
    item [2]:
        class = "IntervalTier"
        name = "N_SPK2"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 2.0
            xmax = 5.0
            text = "我补充文本验证"
'''


class SeparationAlignmentServiceTest(unittest.TestCase):
    def test_textgrid_parser_preserves_overlapping_speaker_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "meeting.TextGrid"
            path.write_text(TEXTGRID, encoding="utf-8")

            segments = parse_textgrid(path)

        self.assertEqual([item["speaker"] for item in segments], ["N_SPK1", "N_SPK2"])
        self.assertEqual(segments[0]["start"], "00:00")
        self.assertEqual(segments[1]["start"], "00:02")
        self.assertEqual(segments[0]["text"], "项目需要语音分离")

    def test_asr_segments_align_to_separated_track_energy_without_textgrid(self) -> None:
        import numpy
        import soundfile

        sample_rate = 8000
        first = numpy.zeros(sample_rate * 4, dtype="float32")
        second = numpy.zeros(sample_rate * 4, dtype="float32")
        first[: sample_rate * 2] = 0.4
        second[sample_rate : sample_rate * 3] = 0.5
        first_track = UPLOAD_DIR / "energy_track_first.wav"
        second_track = UPLOAD_DIR / "energy_track_second.wav"
        soundfile.write(first_track, first, sample_rate)
        soundfile.write(second_track, second, sample_rate)

        try:
            aligned, metrics = align_transcript_to_separation_tracks(
                [
                    {"speaker": "说话人 A", "start": "00:00", "end": "00:01", "text": "第一段"},
                    {"speaker": "说话人 B", "start": "00:01", "end": "00:02", "text": "重叠段"},
                    {"speaker": "说话人 B", "start": "00:02", "end": "00:03", "text": "第二段"},
                ],
                [
                    {"track_id": "track_a", "label": "track A", "audio_url": audio_url(first_track)},
                    {"track_id": "track_b", "label": "track B", "audio_url": audio_url(second_track)},
                ],
            )
        finally:
            first_track.unlink(missing_ok=True)
            second_track.unlink(missing_ok=True)

        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(aligned[0]["primary_track_id"], "track_a")
        self.assertEqual(aligned[2]["primary_track_id"], "track_b")
        self.assertIn("track_a", aligned[1]["separation_tracks"])
        self.assertIn("track_b", aligned[1]["separation_tracks"])
        self.assertEqual(metrics["multi_track_segments"], 1)

    def test_textgrid_evaluation_uses_reference_only_after_separation(self) -> None:
        first_track = UPLOAD_DIR / "eval_track_first.wav"
        second_track = UPLOAD_DIR / "eval_track_second.wav"
        first_track.write_bytes(b"first")
        second_track.write_bytes(b"second")

        def fake_transcribe(path: Path, _display_name: str) -> dict:
            if path == first_track:
                return {"enhanced_asr_text": "我补充文本验证", "signal_metrics": {"ASR 状态": "ok"}}
            return {"enhanced_asr_text": "项目需要语音分离", "signal_metrics": {"ASR 状态": "ok"}}

        with tempfile.TemporaryDirectory() as temp_dir:
            textgrid = Path(temp_dir) / "meeting.TextGrid"
            textgrid.write_text(TEXTGRID, encoding="utf-8")
            try:
                with patch("app.services.separation_alignment_service.find_textgrid_for_audio", return_value=textgrid):
                    evaluation = build_textgrid_separation_evaluation(
                        separated_tracks=[
                            {"track_id": "track_a", "label": "speaker 1", "audio_url": audio_url(first_track)},
                            {"track_id": "track_b", "label": "speaker 2", "audio_url": audio_url(second_track)},
                        ],
                        display_name="unit.wav",
                        transcribe_track=fake_transcribe,
                    )
            finally:
                first_track.unlink(missing_ok=True)
                second_track.unlink(missing_ok=True)

        self.assertEqual(evaluation["source"], "textgrid")
        self.assertGreater(evaluation["reference_overlap_ratio"], 0)
        self.assertEqual(evaluation["track_matches"][0]["matched_reference_speaker"], "N_SPK2")
        self.assertEqual(evaluation["track_matches"][1]["matched_reference_speaker"], "N_SPK1")
        metrics = evaluation_metrics(evaluation)
        self.assertEqual(metrics["textgrid_eval_matched_tracks"], "2")

    def test_auto_track_evaluation_respects_duration_limit(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SEPARATION_EVAL_TRANSCRIBE_TRACKS": "auto",
                "SEPARATION_EVAL_MAX_TRANSCRIBE_SECONDS": "120",
            },
            clear=True,
        ), patch(
            "app.services.separation_alignment_service.find_textgrid_for_audio",
            return_value=Path("meeting.TextGrid"),
        ), patch(
            "app.services.separation_alignment_service.get_audio_duration_seconds",
            return_value=180.0,
        ):
            enabled = should_transcribe_evaluation_tracks(Path("meeting.wav"), "meeting.wav")

        self.assertFalse(enabled)


if __name__ == "__main__":
    unittest.main()
