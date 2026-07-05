"""Smoke test: extract + sample frames, call scene endpoint, validate, print
usage.

Generates a synthetic short clip via ffmpeg (testsrc + noise), so no external
asset is required. Point --video at a real clip to exercise a real scene.

Usage:
    python scripts/smoke_scene.py
    python scripts/smoke_scene.py --video /path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from egoannot.config import get_settings
from egoannot.logging import configure_logging
from egoannot.media.frames import (
    extract_frames,
    probe_video,
    sample_segment_frames,
    sampled_frames_to_content,
    segment_video,
)
from egoannot.schemas.subtasks import SceneResponse
from egoannot.vlm.client import VLMClient
from egoannot.vlm.prompts import build_messages


def _make_synthetic_clip(dst: Path, duration_sec: float = 20.0) -> None:
    """Generate a synthetic testsrc clip with visible motion."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"testsrc=size=640x360:rate=30:duration={duration_sec}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"synthetic clip generation failed: {proc.stderr}")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove leading ``` or ```json
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None, help="Path to input clip; synthesised if omitted.")
    parser.add_argument("--frames-per-segment", type=int, default=None)
    args = parser.parse_args()

    configure_logging(level="INFO")
    settings = get_settings()

    tmpdir = Path(tempfile.mkdtemp(prefix="egoannot-smoke-"))
    print(f"[smoke] tmpdir = {tmpdir}")

    video_path = args.video
    if video_path is None:
        video_path = tmpdir / "synthetic.mp4"
        print(f"[smoke] generating synthetic clip -> {video_path}")
        _make_synthetic_clip(video_path)

    meta = probe_video(video_path)
    print(f"[smoke] meta: duration={meta.duration_sec:.2f}s fps={meta.fps:.2f} {meta.width}x{meta.height}")

    frame_dir = tmpdir / "frames"
    frames = extract_frames(video_path, frame_dir, meta)
    print(f"[smoke] extracted {len(frames)} frames at {settings.frames.fps} fps into {frame_dir}")

    segments = segment_video(meta.duration_sec)
    print(f"[smoke] segments: {[(s.idx, round(s.start_sec, 2), round(s.end_sec, 2)) for s in segments]}")

    per_seg = args.frames_per_segment or settings.frames.per_segment
    seg = segments[0]
    sampled = sample_segment_frames(frames, settings.frames.fps, seg, per_segment=per_seg)
    print(f"[smoke] sampled {len(sampled)} frames from segment 0")
    for f in sampled:
        print(f"        t={f.timestamp_sec:.2f}s  {f.path.name}")

    content = sampled_frames_to_content(sampled)
    total_bytes = sum(len(b["image_url"]["url"]) for b in content if b.get("type") == "image_url")
    print(f"[smoke] base64 payload total = {total_bytes / 1024:.1f} KiB across {len(sampled)} images")

    messages = build_messages("scene", image_content=content)
    print(f"[smoke] system prompt length = {len(messages[0].content)} chars")

    async with VLMClient() as client:
        resp = await client.chat(messages)

    print("--- RAW MODEL OUTPUT ---")
    print(resp.text)
    print("------------------------")
    print(
        f"[smoke] usage: prompt={resp.prompt_tokens}  completion={resp.completion_tokens}  "
        f"total={resp.total_tokens}  (ceiling=16384)"
    )
    headroom = 16384 - resp.total_tokens
    print(f"[smoke] headroom vs 16384: {headroom} tokens ({headroom / 16384:.1%})")

    try:
        parsed = json.loads(_strip_fences(resp.text))
    except json.JSONDecodeError as e:
        print(f"[smoke] FAILED to parse JSON: {e}")
        return 2

    scene = SceneResponse.model_validate(parsed)
    print("[smoke] validated SceneResponse:")
    print(json.dumps(scene.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
