from __future__ import annotations


def separate_demo_audio(case_id: str, enhanced_audio_url: str) -> dict:
    """Return demo speech-separation outputs.

    The current classroom demo keeps separation deterministic by reusing the
    enhanced meeting audio as a placeholder track. A real model can replace
    this function with SepFormer, Conv-TasNet, Demucs, or a speaker-conditioned
    separation backend without changing the API shape.
    """
    label = {
        "clear_meeting": "主说话人轨道",
        "noisy_meeting": "降噪后会议语音轨道",
        "overlap_meeting": "多人讨论分离轨道",
    }.get(case_id, "会议语音轨道")
    return {
        "method": "Demo speech separation placeholder",
        "track_count": "1",
        "tracks": [
            {
                "track_id": f"{case_id}_speaker_mix",
                "label": label,
                "audio_url": enhanced_audio_url,
                "description": "演示模式复用增强后音频，后续可替换为真实说话人分离模型输出。",
            }
        ],
    }


def separate_uploaded_audio(enhanced_audio_url: str) -> dict:
    return {
        "method": "Upload separation placeholder",
        "track_count": "1",
        "tracks": [
            {
                "track_id": "upload_speaker_mix",
                "label": "上传音频语音轨道",
                "audio_url": enhanced_audio_url,
                "description": "上传演示模式保留接口形状，后续可输出多个说话人独立音轨。",
            }
        ],
    }
