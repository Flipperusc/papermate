"""Qwen VLM client for image descriptions used by multimodal chunks."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import PROJECT_ROOT, settings
from src.logger import get_logger


QWEN_VLM_SYSTEM_PROMPT = (
    "You are PaperMate's image understanding component for academic papers. "
    "Describe only visible image content and supplied metadata. "
    "Return concise structured Chinese text for retrieval. Do not invent paper conclusions."
)

logger = get_logger(__name__)


class VLMConfigError(RuntimeError):
    """Raised when VLM configuration is missing or invalid."""


class VLMCallError(RuntimeError):
    """Raised when the VLM API call fails."""


class QwenVLMClient:
    """Describe local paper images with Alibaba Cloud Bailian Qwen VLM."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or settings.vlm_model
        self.api_key = api_key if api_key is not None else settings.vlm_api_key
        self.base_url = base_url or settings.vlm_base_url
        self.timeout = timeout if timeout is not None else settings.vlm_timeout
        self.temperature = (
            temperature if temperature is not None else settings.vlm_temperature
        )
        self.max_tokens = max_tokens if max_tokens is not None else settings.vlm_max_tokens

    def describe(self, image: dict[str, Any]) -> str:
        """Return a structured VLM description for one image metadata record."""
        client = self._create_client()
        image_url = image_url_for_qwen(image)
        prompt = build_image_prompt(image)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QWEN_VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.exception("Qwen VLM call failed. model=%s base_url=%s", self.model, self.base_url)
            raise VLMCallError(str(exc)) from exc

        content = response.choices[0].message.content
        description = normalize_response_content(content)
        if not description:
            raise VLMCallError("Qwen VLM returned an empty image description")
        return description

    def _create_client(self):
        if not self.api_key:
            raise VLMConfigError(
                "Missing VLM API key. Set VLM_API_KEY or DASHSCOPE_API_KEY in .env."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VLMConfigError("Missing openai dependency. Install requirements.txt.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        return OpenAI(**client_kwargs)


def build_image_prompt(image: dict[str, Any]) -> str:
    """Build the prompt sent with each image."""
    return "\n".join(
        [
            "请为论文 RAG 检索生成这张图片的结构化描述。",
            "要求：不超过 260 个中文字符；包含图像类型、主要内容、关键文字/符号/坐标轴/流程关系、与 caption 的关系、可检索关键词。",
            "如果图片是流程图、架构图、曲线图、表格截图或公式，请明确指出结构和关键变量。",
            "元数据：",
            f"- caption: {image.get('caption') or ''}",
            f"- alt_text: {image.get('alt_text') or ''}",
            f"- label: {image.get('label') or ''}",
            f"- kind: {image.get('kind') or 'image'}",
            f"- page: {image.get('page_num') or ''}",
            f"- bbox: {image.get('bbox') or ''}",
        ]
    )


def image_url_for_qwen(image: dict[str, Any]) -> str:
    """Return a remote URL or local Base64 data URL accepted by OpenAI-compatible VLM APIs."""
    path = str(image.get("path") or "").strip()
    if is_remote_or_data_url(path):
        return path

    local_path = resolve_image_path(image)
    if not local_path:
        raise VLMConfigError(
            f"Image file not found for VLM description: {path or image.get('source_paths') or ''}"
        )

    mime_type = (
        str(image.get("mime_type") or "").strip()
        or mimetypes.guess_type(local_path.name)[0]
        or "image/png"
    )
    if mime_type == "application/octet-stream":
        mime_type = "image/png"
    encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_image_path(image: dict[str, Any]) -> Path | None:
    """Resolve image paths from MinerU metadata to an existing local file."""
    candidates = [image.get("path"), *(image.get("source_paths") or [])]
    for candidate in candidates:
        if not candidate:
            continue
        path_text = str(candidate).strip()
        if not path_text or is_remote_or_data_url(path_text):
            continue

        path = Path(path_text)
        if path.is_file():
            return path
        if not path.is_absolute():
            project_path = PROJECT_ROOT / path
            if project_path.is_file():
                return project_path
    return None


def is_remote_or_data_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https", "data"}


def normalize_response_content(content: Any) -> str:
    """Coerce SDK response content to one compact string."""
    if isinstance(content, str):
        return " ".join(content.split())
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return " ".join(" ".join(parts).split())
    return " ".join(str(content or "").split())
