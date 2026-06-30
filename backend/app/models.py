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
    primary_track_id: str | None = None
    primary_track_label: str | None = None
    separation_tracks: list[str] = Field(default_factory=list)


class TranscriptTopicBlock(BaseModel):
    block_id: str
    start: str
    end: str
    summary: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class TranscriptTopic(BaseModel):
    topic_id: str
    title: str
    summary: str
    blocks: list[TranscriptTopicBlock] = Field(default_factory=list)


class SeparatedTrack(BaseModel):
    track_id: str
    label: str
    audio_url: str
    description: str


class ProcessingChunk(BaseModel):
    chunk_id: str
    start: str
    end: str
    duration_seconds: float
    status: str
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
    enhancement_visual_url: str | None = None
    processing_chunks: list[ProcessingChunk] = Field(default_factory=list)
    separated_tracks: list[SeparatedTrack] = Field(default_factory=list)
    direct_asr_text: str
    enhanced_asr_text: str
    signal_metrics: dict[str, str]
    speaker_count_estimation: dict = Field(default_factory=dict)
    steps: list[PipelineStep]
    transcript: list[TranscriptSegment]
    transcript_topics: list[TranscriptTopic] = Field(default_factory=list)
    separation_alignment: dict = Field(default_factory=dict)
    separation_evaluation: dict = Field(default_factory=dict)
    summary: MeetingSummary


class UploadSessionCreateRequest(BaseModel):
    filename: str
    size_bytes: int = Field(ge=1)


class UploadSessionResponse(BaseModel):
    upload_id: str
    chunk_size_bytes: int
    total_chunks: int


class UploadSessionCompleteRequest(BaseModel):
    filename: str
    total_chunks: int = Field(ge=1)
    processing_mode: str = "fast"
