"""DeepSeek LLM client using the OpenAI-compatible Chat API."""

from __future__ import annotations

from config import settings
from src.errors import ErrorCode, LLMError
from src.logger import get_logger


DEFAULT_SYSTEM_PROMPT = (
    "你是 PaperMate 论文阅读助手。你必须只基于用户提供的论文片段回答，"
    "不允许编造论文中没有的信息。如果论文片段不足以回答，"
    "请明确说明‘论文原文中没有找到足够依据回答这个问题’。"
)
DEEPSEEK_CALL_FAILED_MESSAGE = "DeepSeek 模型调用失败，请检查 API Key、模型名称、网络连接或账户额度。"

logger = get_logger(__name__)


class LLMClient:
    """Client responsible only for DeepSeek text generation."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float | None = None,
    ) -> None:
        self.model = model or settings.deepseek_model
        self.api_key = api_key if api_key is not None else settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.system_prompt = system_prompt
        self.timeout = timeout

    def _create_client(self):
        """Create an OpenAI SDK client pointed at DeepSeek, not OpenAI default."""
        if not self.api_key:
            raise LLMError(ErrorCode.DEEPSEEK_API_KEY_MISSING)

        try:
            from openai import OpenAI
        except ImportError as exc:
            logger.exception("OpenAI SDK is not installed.")
            raise LLMError(ErrorCode.DEEPSEEK_CALL_FAILED) from exc

        try:
            client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
            if self.timeout is not None:
                client_kwargs["timeout"] = self.timeout
            return OpenAI(**client_kwargs)
        except Exception as exc:
            logger.exception("Failed to create DeepSeek OpenAI-compatible client.")
            raise LLMError(ErrorCode.DEEPSEEK_CALL_FAILED) from exc

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        """Generate text from a user prompt via DeepSeek Chat Completions."""
        try:
            client = self._create_client()
        except LLMError as exc:
            if exc.code == ErrorCode.DEEPSEEK_API_KEY_MISSING:
                raise
            return DEEPSEEK_CALL_FAILED_MESSAGE

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                logger.error("DeepSeek returned an empty response.")
                return DEEPSEEK_CALL_FAILED_MESSAGE
            return content
        except Exception as exc:
            logger.exception("DeepSeek chat completion failed.")
            return DEEPSEEK_CALL_FAILED_MESSAGE

    def complete(self, prompt: str, **kwargs) -> str:
        """Backward-compatible alias for generate."""
        return self.generate(prompt, **kwargs)
