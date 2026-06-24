from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx

from .summary_service import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _env_enabled,
    _extract_json_object,
    _get_timeout_seconds,
)


DEFAULT_TOPIC_WINDOW_SECONDS = 120
DEFAULT_TOPIC_MAX_BLOCKS = 80


@dataclass(frozen=True)
class TopicClassificationResult:
    topics: list[dict]
    metrics: dict[str, str]
    used_llm: bool


def classify_transcript_topics(transcript: list[dict], case_name: str) -> TopicClassificationResult:
    segments = [_segment_to_dict(item) for item in transcript if _segment_to_dict(item).get("text")]
    blocks = _build_blocks(segments)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not blocks:
        return TopicClassificationResult(
            topics=[],
            metrics={
                "转写主题分类": "无转写内容",
                "转写主题模型": model,
                "转写主题数量": "0",
            },
            used_llm=False,
        )

    fallback_topics = _fallback_topics(blocks)
    api_key = os.getenv("LLM_API_KEY", "").strip()

    if not _env_enabled("LLM_ENABLED", default=True):
        return _fallback_result(fallback_topics, model, "LLM_ENABLED=false")
    if not api_key:
        return _fallback_result(fallback_topics, model, "未配置 LLM_API_KEY")

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
    timeout = _get_timeout_seconds()

    try:
        raw = _request_topic_classification(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            case_name=case_name,
            blocks=blocks,
        )
        topics = _validate_topics(_extract_json_object(raw), blocks)
    except Exception as exc:
        return _fallback_result(fallback_topics, model, f"LLM 调用失败：{type(exc).__name__}")

    return TopicClassificationResult(
        topics=topics,
        metrics={
            "转写主题分类": "LLM API",
            "转写主题模型": model,
            "转写主题数量": str(len(topics)),
            "转写主题状态": "success",
        },
        used_llm=True,
    )


def _fallback_result(topics: list[dict], model: str, status: str) -> TopicClassificationResult:
    return TopicClassificationResult(
        topics=topics,
        metrics={
            "转写主题分类": "本地兜底",
            "转写主题模型": model,
            "转写主题数量": str(len(topics)),
            "转写主题状态": status,
        },
        used_llm=False,
    )


def _request_topic_classification(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    case_name: str,
    blocks: list[dict],
) -> str:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是会议转写主题整理助手。请把带时间范围的转写块按讨论主题归类。"
                    "只能输出 JSON，不要输出 Markdown。不要改写 block_id。"
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(case_name, blocks),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    return content


def _build_prompt(case_name: str, blocks: list[dict]) -> str:
    compact_blocks = [
        {
            "block_id": block["block_id"],
            "time": f"{block['start']} - {block['end']}",
            "text": _truncate(block["summary"], 900),
        }
        for block in blocks
    ]
    return (
        f"会议名称：{case_name}\n"
        "请把下面这些转写时间块按语义主题归类。每个 block_id 必须只出现一次。\n"
        "输出 JSON 格式：\n"
        '{"topics":[{"title":"主题名称","summary":"主题摘要","block_ids":["block_001"]}]}\n\n'
        f"转写块：\n{json.dumps(compact_blocks, ensure_ascii=False)}"
    )


def _validate_topics(data: dict[str, Any], blocks: list[dict]) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        raise ValueError("topic response must include topics list")

    block_map = {block["block_id"]: block for block in blocks}
    used: set[str] = set()
    topics: list[dict] = []

    for index, topic in enumerate(data["topics"], start=1):
        if not isinstance(topic, dict):
            continue
        block_ids = topic.get("block_ids")
        if not isinstance(block_ids, list):
            continue

        topic_blocks = []
        for raw_id in block_ids:
            block_id = str(raw_id).strip()
            if block_id in block_map and block_id not in used:
                topic_blocks.append(block_map[block_id])
                used.add(block_id)

        if topic_blocks:
            topics.append(
                {
                    "topic_id": f"topic_{len(topics) + 1:02d}",
                    "title": str(topic.get("title") or f"主题 {index}").strip()[:40],
                    "summary": str(topic.get("summary") or "").strip()[:160],
                    "blocks": topic_blocks,
                }
            )

    missing = [block for block in blocks if block["block_id"] not in used]
    if missing:
        topics.append(
            {
                "topic_id": f"topic_{len(topics) + 1:02d}",
                "title": "其他讨论",
                "summary": "模型未明确归类的剩余时间块。",
                "blocks": missing,
            }
        )

    if not topics:
        raise ValueError("topic response has no valid blocks")
    return topics[:12]


def _fallback_topics(blocks: list[dict]) -> list[dict]:
    group_size = 4
    topics = []
    for offset in range(0, len(blocks), group_size):
        grouped = blocks[offset : offset + group_size]
        topics.append(
            {
                "topic_id": f"topic_{len(topics) + 1:02d}",
                "title": f"讨论主题 {len(topics) + 1}",
                "summary": _truncate(" ".join(block["summary"] for block in grouped), 120),
                "blocks": grouped,
            }
        )
    return topics


def _build_blocks(segments: list[dict]) -> list[dict]:
    window_seconds = _int_env("LLM_TOPIC_WINDOW_SECONDS", DEFAULT_TOPIC_WINDOW_SECONDS, minimum=30)
    blocks = []
    current = None

    for segment in segments:
        start_seconds = _parse_timestamp(segment.get("start"))
        should_start = (
            current is None
            or math.isnan(start_seconds)
            or start_seconds >= current["start_seconds"] + window_seconds
        )
        if should_start:
            current = {
                "block_id": f"block_{len(blocks) + 1:03d}",
                "start": segment.get("start") or "00:00",
                "end": segment.get("end") or segment.get("start") or "00:00",
                "start_seconds": start_seconds if not math.isnan(start_seconds) else len(blocks) * window_seconds,
                "segments": [],
            }
            blocks.append(current)
        current["segments"].append(segment)
        current["end"] = segment.get("end") or current["end"]

    blocks = [_finalize_block(block) for block in blocks]
    return _merge_to_max_blocks(blocks)


def _merge_to_max_blocks(blocks: list[dict]) -> list[dict]:
    max_blocks = _int_env("LLM_TOPIC_MAX_BLOCKS", DEFAULT_TOPIC_MAX_BLOCKS, minimum=8)
    if len(blocks) <= max_blocks:
        return blocks

    stride = math.ceil(len(blocks) / max_blocks)
    merged = []
    for offset in range(0, len(blocks), stride):
        group = blocks[offset : offset + stride]
        segments = [segment for block in group for segment in block["segments"]]
        merged.append(
            _finalize_block(
                {
                    "block_id": f"block_{len(merged) + 1:03d}",
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "segments": segments,
                }
            )
        )
    return merged


def _finalize_block(block: dict) -> dict:
    summary = " ".join(
        f"{segment.get('speaker') or '说话人'}: {segment.get('text', '').strip()}"
        for segment in block["segments"]
        if segment.get("text")
    )
    return {
        "block_id": block["block_id"],
        "start": block["start"],
        "end": block["end"],
        "summary": _truncate(summary, 1400),
        "segments": block["segments"],
    }


def _segment_to_dict(item: Any) -> dict:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {}
    return {
        "start": str(item.get("start") or ""),
        "end": str(item.get("end") or ""),
        "speaker": str(item.get("speaker") or "说话人"),
        "text": str(item.get("text") or "").strip(),
    }


def _parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return math.nan
    try:
        parts = [float(part) for part in str(value).split(":")]
    except ValueError:
        return math.nan
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return math.nan


def _int_env(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return max(minimum, value)


def _truncate(value: str, limit: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."
