"""End-to-end orchestrator run under MockVLMClient.

Generates a tiny synthetic clip on disk using ffmpeg (via subprocess) and
drives the full annotate + assemble pipeline. Skipped when ffmpeg is not
on PATH so the suite still runs on a minimal dev box.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from egoannot.assemble import assemble_video
from egoannot.db import Video, session_scope
from egoannot.orchestrator import annotate_video
from egoannot.schemas.annotation import FinalAnnotation
from egoannot.vlm.mock import MockVLMClient


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")
def test_mock_end_to_end(tmp_pipeline: Path) -> None:
    video_path = tmp_pipeline / "data" / "videos" / "VID_777777.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=c=0x808080:s=320x240:d=6:r=15",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    with session_scope() as s:
        s.add(
            Video(
                id="VID_777777",
                source_dataset="jaad",
                source_path=str(video_path),
                selected=True,
                status="curated",
            )
        )

    async def _drive() -> str:
        client = MockVLMClient()
        try:
            return await annotate_video("VID_777777", client=client)
        finally:
            await client.aclose()

    status = asyncio.run(_drive())
    assert status == "tasks_done"

    payload = assemble_video("VID_777777")
    FinalAnnotation.model_validate(payload)
    assert payload["walkability"] in {"passable_with_caution"}
    assert len(payload["qa_pairs"]) >= 1
    # Mock caption is English.
    assert "walking" in payload["caption"].lower()
