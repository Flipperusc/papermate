"""Smoke tests for shared external API retry policy."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.embedding_client import EmbeddingClient
from src.external_call import RetryPolicy, call_with_retries, retry_delay_seconds
from src.llm_client import LLMClient
from src.vlm_client import QwenVLMClient


def main() -> None:
    test_call_with_retries_uses_backoff_without_retrying_permanent_status()
    test_llm_client_retries_transient_completion_failure()
    test_openai_embedding_retries_transient_batch_failure()
    test_zhipu_embedding_retries_retryable_http_status()
    test_vlm_client_retries_transient_description_failure()
    print("external call policy tests passed")


def test_call_with_retries_uses_backoff_without_retrying_permanent_status() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def transient_then_ok() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise StatusError(503)
        return "ok"

    result = call_with_retries(
        transient_then_ok,
        operation_name="transient-test",
        logger=FakeLogger(),
        policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.5, max_delay_seconds=2.0),
        sleep=sleeps.append,
    )
    assert result == "ok"
    assert calls["count"] == 2
    assert sleeps == [0.5]
    assert retry_delay_seconds(RetryPolicy(base_delay_seconds=0.5, max_delay_seconds=2.0), 3) == 2.0

    try:
        call_with_retries(
            lambda: (_ for _ in ()).throw(StatusError(400)),
            operation_name="permanent-test",
            logger=FakeLogger(),
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0),
            sleep=sleeps.append,
        )
    except StatusError:
        pass
    else:
        raise AssertionError("permanent 400 error should not be retried")


def test_llm_client_retries_transient_completion_failure() -> None:
    fake_client = FakeOpenAIClient(failures_before_success=1, content="paper answer")
    client = LLMClient(
        api_key="test-key",
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0),
    )
    client._create_client = lambda: fake_client  # type: ignore[method-assign]

    answer = client.generate("question")
    assert answer == "paper answer"
    assert fake_client.chat.completions.calls == 2


def test_openai_embedding_retries_transient_batch_failure() -> None:
    fake_client = FakeEmbeddingOpenAIClient(failures_before_success=1)
    client = EmbeddingClient(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0),
    )
    client._create_openai_client = lambda: fake_client  # type: ignore[method-assign]

    embeddings = client.embed(["alpha", "beta"])
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert fake_client.embeddings.calls == 2


def test_zhipu_embedding_retries_retryable_http_status() -> None:
    session = FakeZhipuSession([FakeResponse(503), FakeResponse(200, data_count=2)])
    client = EmbeddingClient(
        provider="zhipu",
        api_key="test-key",
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0),
        session=session,
    )

    embeddings = client.embed(["alpha", "beta"])
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert session.calls == 2


def test_vlm_client_retries_transient_description_failure() -> None:
    fake_client = FakeOpenAIClient(failures_before_success=1, content="红色方块图片")
    client = QwenVLMClient(
        api_key="test-key",
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0),
    )
    client._create_client = lambda: fake_client  # type: ignore[method-assign]

    description = client.describe(
        {
            "kind": "image",
            "caption": "red square",
            "path": "data:image/png;base64,AA==",
            "mime_type": "image/png",
            "page_num": 1,
        }
    )
    assert description == "红色方块图片"
    assert fake_client.chat.completions.calls == 2


class FakeLogger:
    def warning(self, *args, **kwargs) -> None:
        return None


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class FakeOpenAIClient:
    def __init__(self, failures_before_success: int, content: str) -> None:
        self.chat = FakeChat(failures_before_success, content)


class FakeChat:
    def __init__(self, failures_before_success: int, content: str) -> None:
        self.completions = FakeChatCompletions(failures_before_success, content)


class FakeChatCompletions:
    def __init__(self, failures_before_success: int, content: str) -> None:
        self.failures_before_success = failures_before_success
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise StatusError(503)
        return FakeChatResponse(self.content)


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeEmbeddingOpenAIClient:
    def __init__(self, failures_before_success: int) -> None:
        self.embeddings = FakeEmbeddingEndpoint(failures_before_success)


class FakeEmbeddingEndpoint:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise StatusError(503)
        return FakeEmbeddingResponse(
            [
                FakeEmbeddingItem(1, [0.0, 1.0]),
                FakeEmbeddingItem(0, [1.0, 0.0]),
            ]
        )


class FakeEmbeddingResponse:
    def __init__(self, data: list) -> None:
        self.data = data


class FakeEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class FakeZhipuSession:
    def __init__(self, responses: list["FakeResponse"]) -> None:
        self.responses = responses
        self.calls = 0

    def post(self, *args, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class FakeResponse:
    def __init__(self, status_code: int, data_count: int = 0) -> None:
        self.status_code = status_code
        self.text = f"HTTP {status_code}"
        self._data_count = data_count

    def raise_for_status(self) -> None:
        error = requests.HTTPError(f"HTTP {self.status_code}")
        error.response = self
        raise error

    def json(self) -> dict:
        data = []
        for index in range(self._data_count):
            embedding = [0.0, 0.0]
            embedding[index] = 1.0
            data.append({"index": index, "embedding": embedding})
        return {"data": data}


if __name__ == "__main__":
    main()
