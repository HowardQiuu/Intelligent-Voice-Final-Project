from pydantic import BaseModel, Field


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


class SeparatedTrack(BaseModel):
    track_id: str
    label: str
    audio_url: str
    description: str


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
    separated_tracks: list[SeparatedTrack] = Field(default_factory=list)
    direct_asr_text: str
    enhanced_asr_text: str
    signal_metrics: dict[str, str]
    steps: list[PipelineStep]
    transcript: list[TranscriptSegment]
    summary: MeetingSummary
