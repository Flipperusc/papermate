"""Shared retry helpers for external API calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar


T = TypeVar("T")
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    """Small retry policy shared by model and API clients."""

    max_attempts: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0


def call_with_retries(
    operation: Callable[[], T],
    *,
    operation_name: str,
    logger: Any,
    policy: RetryPolicy,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an external operation with bounded exponential backoff."""
    max_attempts = max(1, int(policy.max_attempts or 1))
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except retry_exceptions as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise
            retry_allowed = should_retry(exc) if should_retry else default_should_retry(exc)
            if not retry_allowed:
                raise
            delay = retry_delay_seconds(policy, attempt)
            logger.warning(
                "%s failed; retrying. attempt=%s max_attempts=%s delay=%.2fs error=%s",
                operation_name,
                attempt,
                max_attempts,
                delay,
                safe_error_message(exc),
            )
            if delay > 0:
                sleep(delay)

    # Unreachable in normal control flow, but keeps type-checkers honest.
    if last_error:
        raise last_error
    raise RuntimeError(f"{operation_name} did not run")


def retry_delay_seconds(policy: RetryPolicy, failed_attempt: int) -> float:
    """Return the backoff delay before the next attempt."""
    base = max(0.0, float(policy.base_delay_seconds or 0.0))
    cap = max(base, float(policy.max_delay_seconds or base))
    delay = base * (2 ** max(0, int(failed_attempt) - 1))
    return min(cap, delay)


def default_should_retry(error: BaseException) -> bool:
    """Return whether an exception looks transient."""
    status_code = exception_status_code(error)
    if status_code is None:
        return True
    return int(status_code) in RETRYABLE_STATUS_CODES


def exception_status_code(error: BaseException) -> int | None:
    """Extract HTTP-like status code from common SDK exceptions."""
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def safe_error_message(error: BaseException, limit: int = 500) -> str:
    """Return a compact error string safe for structured logs."""
    message = " ".join(str(error or type(error).__name__).split())
    return message[: max(1, int(limit))]

