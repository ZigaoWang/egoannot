"""Shared sub-task runner.

Each sub-task module wraps :func:`run_image_subtask` or :func:`run_text_subtask`.
Behavior:

1. Build chat messages via ``vlm.prompts.build_messages``.
2. Call the client.
3. Strip markdown fences if present.
4. ``json.loads`` -> ``response_model.model_validate``.
5. On JSON / validation failure, retry ONCE with a corrective user message
   ("Your previous JSON was invalid …") preserving the original images.
6. Persist a ``TaskResult`` row keyed by (video_id, segment_idx, task_name).
   ``ok=True`` on success; ``ok=False`` with an empty ``parsed_json`` on
   final failure. Raw text is always persisted for debugging.

A per-video failure (all core tasks fail, or frame extraction failed) is
handled one level up in ``orchestrator.py``.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar

import structlog
from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from ..db import TaskResult, session_scope
from ..vlm.client import ChatMessage, VLMError
from ..vlm.prompts import build_messages

R = TypeVar("R", bound=BaseModel)
_log = structlog.stdlib.get_logger(__name__)

_CORRECTIVE_TEXT = (
    "Your previous response was NOT valid JSON matching the schema. "
    "Re-emit ONE valid JSON object matching the schema exactly. "
    "Return the JSON alone: no prose, no markdown fences."
)


class _Client(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = True,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any: ...


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _persist_result(
    video_id: str,
    segment_idx: int,
    task_name: str,
    *,
    raw_response: str,
    parsed_json: str,
    ok: bool,
    attempts: int,
) -> None:
    """Upsert a TaskResult row.

    We use SELECT + UPDATE-or-INSERT (rather than SQLite's ON CONFLICT DO
    UPDATE) so the code stays portable across drivers and reads well.
    """
    with session_scope() as session:
        stmt = (
            select(TaskResult)
            .where(
                TaskResult.video_id == video_id,
                TaskResult.segment_idx == segment_idx,
                TaskResult.task_name == task_name,
            )
        )
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            row = TaskResult(
                video_id=video_id,
                segment_idx=segment_idx,
                task_name=task_name,
            )
            session.add(row)
        row.raw_response = raw_response
        row.parsed_json = parsed_json
        row.ok = ok
        row.attempts = attempts


async def _one_shot(
    client: _Client,
    messages: list[ChatMessage],
    response_model: type[R],
    *,
    max_output_tokens: int | None,
) -> tuple[R | None, str, str | None]:
    """Single call: returns (validated_or_None, raw_text, error_summary_or_None)."""
    try:
        resp = await client.chat(messages, max_output_tokens=max_output_tokens)
    except VLMError as e:
        return None, "", f"vlm_error: {e}"

    raw_text = resp.text or ""
    try:
        parsed = json.loads(_strip_fences(raw_text))
    except json.JSONDecodeError as e:
        return None, raw_text, f"json_decode: {e}"

    try:
        validated = response_model.model_validate(parsed)
    except ValidationError as e:
        return None, raw_text, f"validation: {e.errors(include_url=False)[:2]}"

    return validated, raw_text, None


async def run_image_subtask(
    *,
    client: _Client,
    task_name: str,
    response_model: type[R],
    image_content: list[dict[str, Any]],
    video_id: str,
    segment_idx: int,
    template_params: dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    persist: bool = True,
) -> tuple[R | None, bool]:
    """Run one image-bearing sub-task, validate, persist, return the result.

    Returns ``(validated_model | None, ok)``. When ``ok`` is False, the
    caller falls back to the assembly-stage default for this sub-task.
    """
    messages = build_messages(
        task_name,
        image_content=image_content,
        template_params=template_params,
    )
    return await _run(
        client,
        task_name=task_name,
        response_model=response_model,
        messages=messages,
        video_id=video_id,
        segment_idx=segment_idx,
        max_output_tokens=max_output_tokens,
        persist=persist,
    )


async def run_text_subtask(
    *,
    client: _Client,
    task_name: str,
    response_model: type[R],
    context_json: str,
    video_id: str,
    segment_idx: int,
    template_params: dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    persist: bool = True,
) -> tuple[R | None, bool]:
    """Run one text-only synthesis sub-task on prior sub-task outputs."""
    messages = build_messages(
        task_name,
        context_json=context_json,
        template_params=template_params,
    )
    return await _run(
        client,
        task_name=task_name,
        response_model=response_model,
        messages=messages,
        video_id=video_id,
        segment_idx=segment_idx,
        max_output_tokens=max_output_tokens,
        persist=persist,
    )


async def _run(
    client: _Client,
    *,
    task_name: str,
    response_model: type[R],
    messages: list[ChatMessage],
    video_id: str,
    segment_idx: int,
    max_output_tokens: int | None,
    persist: bool,
) -> tuple[R | None, bool]:
    structlog.contextvars.bind_contextvars(
        video_id=video_id, task_name=task_name, segment_idx=segment_idx
    )
    try:
        validated, raw_text, err = await _one_shot(
            client, messages, response_model, max_output_tokens=max_output_tokens
        )
        attempts = 1
        if validated is None:
            _log.warning("subtask_first_attempt_failed", err=err)
            corrective_messages = [*messages, ChatMessage(role="user", content=_CORRECTIVE_TEXT)]
            validated, raw_text, err = await _one_shot(
                client,
                corrective_messages,
                response_model,
                max_output_tokens=max_output_tokens,
            )
            attempts = 2
            if validated is None:
                _log.warning("subtask_failed", err=err)

        ok = validated is not None
        parsed_json = ""
        if validated is not None:
            parsed_json = validated.model_dump_json()

        if persist:
            _persist_result(
                video_id,
                segment_idx,
                task_name,
                raw_response=raw_text,
                parsed_json=parsed_json,
                ok=ok,
                attempts=attempts,
            )
        else:
            _log.debug("subtask_result_not_persisted", ok=ok, attempts=attempts)
        _log.info("subtask_done", ok=ok, attempts=attempts)
        return validated, ok
    finally:
        structlog.contextvars.unbind_contextvars("task_name", "segment_idx")


__all__ = ["run_image_subtask", "run_text_subtask"]
