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
DEFAULT_SUMMARY_MAX_CHARS = 12000
STATIC_FALLBACK_TITLE = "上传会议音频演示结果"


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
    base_fallback_data = _validate_summary(fallback or fallback_summary(), case_name)
    fallback_data = _content_aware_fallback(base_fallback_data, transcript, case_name, enhanced_asr_text)
    fallback_generation = "本地转写兜底" if fallback_data != base_fallback_data else "缓存兜底"
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    api_key = os.getenv("LLM_API_KEY", "").strip()

    if not _env_enabled("LLM_ENABLED", default=True):
        return SummaryGenerationResult(
            summary=fallback_data,
            metrics={
                "摘要生成": fallback_generation,
                "摘要模型": model,
                "摘要状态": "LLM_ENABLED=false",
            },
            used_llm=False,
        )

    if not api_key:
        return SummaryGenerationResult(
            summary=fallback_data,
            metrics={
                "摘要生成": fallback_generation,
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
        if _summary_is_low_information(summary, transcript, enhanced_asr_text):
            raise ValueError("LLM summary is low information")
    except Exception as exc:
        return SummaryGenerationResult(
            summary=fallback_data,
            metrics={
                "摘要生成": fallback_generation,
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
    max_chars = _get_int_env("LLM_SUMMARY_MAX_CHARS", DEFAULT_SUMMARY_MAX_CHARS, minimum=2000)
    transcript_text = "\n".join(
        f"{item.get('start', '')}-{item.get('end', '')} {item.get('speaker', '未知说话人')}：{item.get('text', '')}"
        for item in transcript
    )
    transcript_text = _clip_text(transcript_text, max_chars)
    enhanced_text = _clip_text(enhanced_asr_text, max_chars // 2)
    return (
        f"会议名称：{case_name}\n\n"
        f"增强后 ASR 文本：\n{enhanced_text}\n\n"
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


def _content_aware_fallback(
    fallback_data: dict,
    transcript: list[dict],
    case_name: str,
    enhanced_asr_text: str,
) -> dict:
    if not _has_transcript_content(transcript, enhanced_asr_text):
        return fallback_data
    title = _clean_text(fallback_data.get("title"))
    abstract = _clean_text(fallback_data.get("abstract"))
    if title == STATIC_FALLBACK_TITLE or "兜底数据" in abstract or "演示" in title:
        return _build_extractive_summary(transcript, case_name, enhanced_asr_text)
    return fallback_data


def _build_extractive_summary(transcript: list[dict], case_name: str, enhanced_asr_text: str) -> dict:
    segments = [_clean_text(item.get("text")) for item in transcript if _clean_text(item.get("text"))]
    full_text = _clean_text(enhanced_asr_text) or "。".join(segments)
    sentences = _split_sentences(full_text)
    abstract_source = "。".join(sentences[:4]) or full_text
    abstract = _clip_text(abstract_source, 220).strip("，。；、 ")
    if abstract:
        abstract = f"{abstract}。"
    else:
        abstract = "系统已根据会议转写生成本地纪要，建议补充更清晰的会议音频以提升摘要质量。"

    decisions = _pick_sentences(
        sentences,
        ("确定", "决定", "确认", "同意", "采用", "优先", "可以", "适合", "需要"),
        limit=4,
    )
    action_items = _pick_sentences(
        sentences,
        ("后续", "需要", "准备", "继续", "确认", "检查", "优化", "接入", "补充", "调整"),
        limit=4,
    )
    if not decisions:
        decisions = ["围绕会议转写内容保留结构化纪要输出"]
    if not action_items:
        action_items = ["继续根据会议转写补充决策与待办归属"]

    return _validate_summary(
        {
            "title": f"{case_name}会议纪要",
            "keywords": _extract_keywords(full_text),
            "abstract": abstract,
            "decisions": decisions,
            "action_items": action_items,
        },
        case_name,
    )


def _has_transcript_content(transcript: list[dict], enhanced_asr_text: str) -> bool:
    if _clean_text(enhanced_asr_text):
        return True
    return any(_clean_text(item.get("text")) for item in transcript)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?；;\n]+", _clean_text(text))
    return [part.strip(" ，、") for part in parts if len(part.strip()) >= 4]


def _pick_sentences(sentences: list[str], keywords: tuple[str, ...], limit: int) -> list[str]:
    picked: list[str] = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in keywords):
            item = _clip_text(sentence.strip("，。；、 "), 90)
            if item and item not in picked:
                picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def _extract_keywords(text: str) -> list[str]:
    candidates = [
        "目标人群",
        "产品设计",
        "手机产品",
        "促销方式",
        "年龄段",
        "功能配置",
        "会议纪要",
        "说话人分段",
        "语音转写",
        "语音增强",
    ]
    keywords = [item for item in candidates if item in text]
    if len(keywords) >= 3:
        return keywords[:8]
    words = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    stopwords = {"这个", "就是", "然后", "我们", "他们", "大家", "觉得", "可以", "还是", "因为", "所以", "一下"}
    counts: dict[str, int] = {}
    for word in words:
        if word in stopwords:
            continue
        counts[word] = counts.get(word, 0) + 1
    for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= 8:
            break
    return keywords or ["会议纪要", "语音转写", "摘要生成"]


def _summary_is_low_information(summary: dict, transcript: list[dict], enhanced_asr_text: str) -> bool:
    if not _has_transcript_content(transcript, enhanced_asr_text):
        return False
    text = " ".join(
        [
            _clean_text(summary.get("title")),
            _clean_text(summary.get("abstract")),
            " ".join(_clean_list(summary.get("decisions"))),
            " ".join(_clean_list(summary.get("action_items"))),
        ]
    )
    bad_markers = ("未知会议", "无法生成摘要", "无决策", "无待办", "内容不完整")
    return any(marker in text for marker in bad_markers)


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


def _clip_text(text: str, limit: int) -> str:
    value = _clean_text(text)
    if len(value) <= limit:
        return value
    head = value[: int(limit * 0.7)].rstrip()
    tail = value[-int(limit * 0.25) :].lstrip()
    return f"{head}\n……（中间内容已截断）……\n{tail}"


def _get_timeout_seconds() -> float:
    raw = os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(1.0, timeout)


def _get_int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}
