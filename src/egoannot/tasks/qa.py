"""qa sub-task: text-only synthesis of prior segment results into exactly
``tasks.num_qa`` question-answer pairs spanning the six conceptual layers.
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..schemas.subtasks import QAResponse
from .base import run_text_subtask

TASK_NAME = "qa"


async def run(
    *,
    client: Any,
    context_json: str,
    video_id: str,
    segment_idx: int,
) -> tuple[QAResponse | None, bool]:
    settings = get_settings()
    return await run_text_subtask(
        client=client,
        task_name=TASK_NAME,
        response_model=QAResponse,
        context_json=context_json,
        video_id=video_id,
        segment_idx=segment_idx,
        template_params={"num_qa": settings.tasks.num_qa},
    )


__all__ = ["TASK_NAME", "run"]
