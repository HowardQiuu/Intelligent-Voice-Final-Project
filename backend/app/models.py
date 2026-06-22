from pydantic import BaseModel


class PipelineStep(BaseModel):
    key: str
    name: str
    status: str
    detail: str


class TranscriptSegment(BaseModel):
    start: str
    end: str
    speaker: str
    text: str


class MeetingSummary(BaseModel):
    title: str
    keywords: list[str]
    abstract: str
    decisions: list[str]
    action_items: list[str]


class ProcessResult(BaseModel):
    case_id: str
    case_name: str
    original_audio_url: str
    enhanced_audio_url: str
    direct_asr_text: str
    enhanced_asr_text: str
    signal_metrics: dict[str, str]
    steps: list[PipelineStep]
    transcript: list[TranscriptSegment]
    summary: MeetingSummary
