"""scene sub-task: layers 1 (scene/localization) + 5 (ego-motion)."""

from __future__ import annotations

from typing import Any

from ..schemas.subtasks import SceneResponse
from .base import run_image_subtask

TASK_NAME = "scene"


async def run(
    *,
    client: Any,
    image_content: list[dict[str, Any]],
    video_id: str,
    segment_idx: int,
) -> tuple[SceneResponse | None, bool]:
    return await run_image_subtask(
        client=client,
        task_name=TASK_NAME,
        response_model=SceneResponse,
        image_content=image_content,
        video_id=video_id,
        segment_idx=segment_idx,
    )


__all__ = ["TASK_NAME", "run"]
