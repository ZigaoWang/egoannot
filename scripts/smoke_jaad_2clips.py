"""Run 2 real JAAD clips end-to-end. Bypasses curate by inserting Video
rows directly with selected=True + status=curated so short clips are not
filtered by ``curate.min_sec``.

Usage:
    python scripts/smoke_jaad_2clips.py \
        --clip VID_900001=data/raw/jaad/JAAD_clips/video_0001.mp4 \
        --clip VID_900002=data/raw/jaad/JAAD_clips/video_0002.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from egoannot.assemble import assemble_video
from egoannot.config import get_settings
from egoannot.db import Video, init_engine, session_scope
from egoannot.export import export_all
from egoannot.logging import configure_logging
from egoannot.orchestrator import annotate_video
from egoannot.vlm.client import VLMClient


def _parse_clip(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "clip spec must be VIDEO_ID=/absolute/path.mp4"
        )
    vid, path = spec.split("=", 1)
    return vid.strip(), str(Path(path).expanduser().resolve())


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        type=_parse_clip,
        help="VIDEO_ID=/path/to/clip.mp4 (repeatable)",
    )
    args = parser.parse_args()
    clips: list[tuple[str, str]] = list(args.clip)

    configure_logging(level="INFO")
    settings = get_settings()
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.paths.db_path.exists():
        settings.paths.db_path.unlink()
    init_engine(settings.paths.db_path)

    for vid, _ in clips:
        fd = settings.paths.frames_dir / vid
        if fd.exists():
            for p in fd.glob("*"):
                p.unlink()

    with session_scope() as s:
        for vid, src in clips:
            s.add(
                Video(
                    id=vid,
                    source_dataset="jaad",
                    source_path=src,
                    selected=True,
                    status="curated",
                )
            )

    client = VLMClient()
    try:
        for vid, _ in clips:
            print(f"\n=== annotate {vid} ===")
            status = await annotate_video(vid, client=client)
            print(f"[jaad] {vid}: annotate status={status}")
    finally:
        await client.aclose()

    for vid, _ in clips:
        payload = assemble_video(vid)
        print(f"\n=== ASSEMBLED {vid} ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    per_video, jsonl_path = export_all(jsonl=True)
    print(f"\n[jaad] exported {len(per_video)} json files + {jsonl_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
