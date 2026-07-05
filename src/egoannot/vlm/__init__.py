"""VLM client: async OpenAI-compatible calls, mock, prompt templates."""

from __future__ import annotations

from .client import ChatMessage, VLMClient, VLMError, VLMResponse
from .mock import MockVLMClient
from .prompts import build_messages

__all__ = [
    "ChatMessage",
    "MockVLMClient",
    "VLMClient",
    "VLMError",
    "VLMResponse",
    "build_messages",
]
