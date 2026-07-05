"""Async OpenAI-compatible client for the local vLLM server.

Thin wrapper around httpx.AsyncClient with tenacity-driven retries for
transient 5xx / timeouts. NOT responsible for validating the response
body — that is the caller's job via the Pydantic sub-task models.

Every call returns the raw text plus token usage so the orchestrator can
persist the raw response (for debugging) and monitor the 16384-token
context budget.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import VLMSettings, get_settings

_log = structlog.stdlib.get_logger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One chat message. Content may be a str or a list of OpenAI content blocks."""

    role: Role
    content: str | list[dict[str, Any]]


@dataclass
class VLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class VLMError(RuntimeError):
    """Raised for non-retryable failures or after retries are exhausted."""


class VLMClientProtocol(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = True,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse: ...

    async def aclose(self) -> None: ...


_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


def _is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)


class VLMClient:
    """Real client. One instance per event loop; call ``aclose`` on shutdown."""

    def __init__(self, settings: VLMSettings | None = None) -> None:
        s = settings or get_settings().vlm
        self._settings = s
        self._client = httpx.AsyncClient(
            base_url=s.base_url.rstrip("/"),
            timeout=httpx.Timeout(s.timeout_sec, connect=10.0),
            headers={
                "Authorization": f"Bearer {s.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> VLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = True,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": (
                temperature if temperature is not None else self._settings.temperature
            ),
            "top_p": self._settings.top_p,
            "max_tokens": (
                max_output_tokens
                if max_output_tokens is not None
                else self._settings.max_output_tokens
            ),
        }
        if response_format_json:
            # vLLM honors OpenAI's response_format for guided JSON decoding
            # when the backend supports it; on servers that don't, this key
            # is ignored, and validation catches the resulting drift.
            payload["response_format"] = {"type": "json_object"}

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._settings.max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            retry=retry_if_exception_type(
                (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError)
            ),
        ):
            with attempt:
                try:
                    resp = await self._client.post("/chat/completions", json=payload)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if not _is_retryable_http(e):
                        _log.warning(
                            "vlm_http_non_retryable",
                            status=e.response.status_code,
                            body=e.response.text[:400],
                        )
                        raise VLMError(
                            f"vLLM returned {e.response.status_code}: {e.response.text[:400]}"
                        ) from e
                    _log.warning(
                        "vlm_http_retry",
                        status=e.response.status_code,
                        body=e.response.text[:200],
                    )
                    raise
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    _log.warning("vlm_transport_retry", err=str(e))
                    raise
                break
        else:
            raise VLMError("vLLM call retries exhausted without exception (unreachable)")

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise VLMError(f"malformed vLLM response body: {e}") from e

        usage = data.get("usage") or {}
        return VLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            raw=data,
        )


__all__ = [
    "ChatMessage",
    "VLMClient",
    "VLMClientProtocol",
    "VLMError",
    "VLMResponse",
]


# Guard for tenacity's stop condition when max_retries=0 (single attempt).
def _ensure_retry_sanity() -> None:
    s = get_settings().vlm
    if s.max_retries < 0:
        raise VLMError(f"vlm.max_retries must be >= 0, got {s.max_retries}")


with contextlib.suppress(Exception):
    # Module import must not crash the process; settings may not be loaded
    # yet in every context (e.g. --help before config.yaml is present).
    _ensure_retry_sanity()


# The RetryError type is re-exported for callers that want to catch it
# specifically (e.g. distinguish "retries exhausted" from validation failures).
__all__.append("RetryError")
