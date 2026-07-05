"""events sub-task: layer 3 (temporally grounded happenings)."""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..schemas.subtasks import EventsResponse
from .base import run_image_subtask

TASK_NAME = "events"


async def run(
    *,
    client: Any,
    image_content: list[dict[str, Any]],
    video_id: str,
    segment_idx: int,
) -> tuple[EventsResponse | None, bool]:
    settings = get_settings()
    return await run_image_subtask(
        client=client,
        task_name=TASK_NAME,
        response_model=EventsResponse,
        image_content=image_content,
        video_id=video_id,
        segment_idx=segment_idx,
        template_params={"max_events": settings.tasks.max_events},
    )


__all__ = ["TASK_NAME", "run"]
