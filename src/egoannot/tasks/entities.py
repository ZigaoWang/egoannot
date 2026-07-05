"""entities sub-task: layers 2 (key entities) + 4 (motion & trends)."""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..schemas.subtasks import EntitiesResponse
from .base import run_image_subtask

TASK_NAME = "entities"


async def run(
    *,
    client: Any,
    image_content: list[dict[str, Any]],
    video_id: str,
    segment_idx: int,
) -> tuple[EntitiesResponse | None, bool]:
    settings = get_settings()
    return await run_image_subtask(
        client=client,
        task_name=TASK_NAME,
        response_model=EntitiesResponse,
        image_content=image_content,
        video_id=video_id,
        segment_idx=segment_idx,
        template_params={"max_entities": settings.tasks.max_entities},
    )


__all__ = ["TASK_NAME", "run"]
