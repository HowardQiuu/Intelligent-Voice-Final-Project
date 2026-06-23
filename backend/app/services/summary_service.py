from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


SUMMARY_FIELDS = ("title", "keywords", "abstract", "decisions", "action_items")
BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class SummaryGenerationResult:
    summary: dict
    metrics: dict[str, str]
    used_llm: bool


def fallback_summary() -> dict:
    return {
        "title": "上传会议音频演示结果",
        "keywords": ["音频上传", "语音增强", "ASR 接入", "会议纪要"],
        "abstract": "系统已完成上传音频的接收和预处理，并通过兜底数据展示从音频到会议纪要的完整链路。",
        "decisions": ["当前版本优先保证课堂演示稳定", "后续可接入真实 ASR 与摘要 API"],
        "action_items": ["准备真实会议样例", "接入 faster-whisper", "根据课程展示需要优化前端效果"],
    }


def generate_summary(
    transcript: list[dict],
    case_name: str,
    enhanced_asr_text: str,
    fallback: dict | None = None,
) -> SummaryGenerationResult:
    fallback_data = _validate_summary(fallback or fallback_summary(), case_name)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    api_key = os.getenv("LLM_API_KEY", "").strip()

    if not _env_enabled("LLM_ENABLED", default=True):
        return SummaryGenerationResult(
            summary=fallback_data,
            metrics={
                "摘要生成": "缓存兜底",
                "摘要模型": model,
                "摘要状态": "LLM_ENABLED=false",
            },
            used_llm=False,
        )

    if not api_key:
        return SummaryGenerationResult(
            summary=fallback_data,
            metrics={
                "摘要生成": "缓存兜底",
                "摘要模型": model,
                "摘要状态": "未配置 LLM_API_KEY",
            },
            used_llm=False,
        )

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
    timeout = _get_timeout_seconds()

    try:
        content = _request_summary(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            case_name=case_name,
            enhanced_asr_text=enhanced_asr_text,
            transcript=transcript,
        )
        summary = _validate_summary(_extract_json_object(content), case_name)
    except Exception as exc:
        return SummaryGenerationResult(
            summary=fallback_data,
            metrics={
                "摘要生成": "缓存兜底",
                "摘要模型": model,
                "摘要状态": f"LLM 调用失败：{type(exc).__name__}",
            },
            used_llm=False,
        )

    return SummaryGenerationResult(
        summary=summary,
        metrics={
            "摘要生成": "LLM API",
            "摘要模型": model,
            "摘要状态": "结构化 JSON 生成成功",
        },
        used_llm=True,
    )


def _request_summary(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    case_name: str,
    enhanced_asr_text: str,
    transcript: list[dict],
) -> str:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是智能会议纪要助手。请根据会议转写生成结构化中文纪要。"
                    "只能输出一个 JSON 对象，不要输出 Markdown、解释或代码块。"
                    "JSON 字段必须包含 title、keywords、abstract、decisions、action_items。"
                ),
            },
            {
                "role": "user",
                "content": _build_user_prompt(case_name, enhanced_asr_text, transcript),
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
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    return content


def _build_user_prompt(case_name: str, enhanced_asr_text: str, transcript: list[dict]) -> str:
    transcript_text = "\n".join(
        f"{item.get('start', '')}-{item.get('end', '')} {item.get('speaker', '未知说话人')}：{item.get('text', '')}"
        for item in transcript
    )
    return (
        f"会议名称：{case_name}\n\n"
        f"增强后 ASR 文本：\n{enhanced_asr_text}\n\n"
        f"带时间戳和说话人标签的转写：\n{transcript_text}\n\n"
        "请输出如下 JSON：\n"
        "{"
        '"title":"会议标题",'
        '"keywords":["关键词1","关键词2"],'
        '"abstract":"100-200字会议摘要",'
        '"decisions":["关键决策1","关键决策2"],'
        '"action_items":["待办事项1","待办事项2"]'
        "}"
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM summary is not a JSON object")
    return data


def _validate_summary(data: dict, case_name: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError("summary fallback is not a dict")

    title = _clean_text(data.get("title")) or f"{case_name}会议纪要"
    abstract = _clean_text(data.get("abstract")) or "系统已根据会议转写生成结构化纪要，包含主题、关键词、关键决策和待办事项。"
    keywords = _clean_list(data.get("keywords")) or ["会议纪要", "语音转写", "摘要生成"]
    decisions = _clean_list(data.get("decisions")) or ["保留结构化会议纪要输出"]
    action_items = _clean_list(data.get("action_items")) or ["继续完善会议摘要生成效果"]

    return {
        "title": title,
        "keywords": keywords[:8],
        "abstract": abstract,
        "decisions": decisions[:6],
        "action_items": action_items[:6],
    }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[；;\n]", value)
    elif isinstance(value, list):
        items = value
    else:
        return []
    cleaned = [_clean_text(item) for item in items]
    return [item for item in cleaned if item]


def _get_timeout_seconds() -> float:
    raw = os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(1.0, timeout)


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}
