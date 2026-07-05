"""caption sub-task: text-only synthesis of prior segment results into a
2-3 sentence first-person Chinese caption.
"""

from __future__ import annotations

from typing import Any

from ..schemas.subtasks import CaptionResponse
from .base import run_text_subtask

TASK_NAME = "caption"


async def run(
    *,
    client: Any,
    context_json: str,
    video_id: str,
    segment_idx: int,
) -> tuple[CaptionResponse | None, bool]:
    return await run_text_subtask(
        client=client,
        task_name=TASK_NAME,
        response_model=CaptionResponse,
        context_json=context_json,
        video_id=video_id,
        segment_idx=segment_idx,
    )


__all__ = ["TASK_NAME", "run"]
