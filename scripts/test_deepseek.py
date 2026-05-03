"""Smoke test for the DeepSeek LLM client."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.errors import LLMError
from src.llm_client import LLMClient


def test_deepseek_generation() -> str:
    """Call DeepSeek once and return generated content."""
    prompt = "请用一句话回复：PaperMate 连接 DeepSeek 成功。"
    return LLMClient().generate(prompt, temperature=0.2, max_tokens=100)


def main() -> int:
    """Run a simple manual DeepSeek connectivity test."""
    try:
        content = test_deepseek_generation()
    except LLMError as exc:
        print(exc.message)
        return 1

    if not content.strip():
        print("DeepSeek 返回内容为空。")
        return 1

    print("DeepSeek 返回内容：")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
