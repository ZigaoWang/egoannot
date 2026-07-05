"""judgment sub-task: layer 6 (walkability & risks). Optional; skipped
under ``--no-risk``, in which case assembly uses walkability=unknown,
risks=[], actions=[observe].
"""

from __future__ import annotations

from typing import Any

from ..schemas.subtasks import JudgmentResponse
from .base import run_image_subtask

TASK_NAME = "judgment"


async def run(
    *,
    client: Any,
    image_content: list[dict[str, Any]],
    video_id: str,
    segment_idx: int,
) -> tuple[JudgmentResponse | None, bool]:
    return await run_image_subtask(
        client=client,
        task_name=TASK_NAME,
        response_model=JudgmentResponse,
        image_content=image_content,
        video_id=video_id,
        segment_idx=segment_idx,
    )


__all__ = ["TASK_NAME", "run"]
