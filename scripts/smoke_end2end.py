"""End-to-end smoke: synthesize a clip with a known visible event, run the
full per-video pipeline (probe → extract → segment → 6 sub-tasks → assemble),
print the assembled JSON, and verify time_span alignment against ground truth.

Ground truth event (deliberately hard-coded so the model's output can be
compared against it):

    A solid red rectangle is composited onto a gray background, visible
    ONLY between t=8.0s and t=14.0s. The rest of the clip (0-8s, 14-30s)
    is a plain gray backdrop.

    Any entity / event that describes this red block should have a
    time_span close to [8.0, 14.0]. Wider is acceptable (model conservatism);
    inverted or wildly off spans are a red flag.

Also runs a partial-failure demo (--force-fail entities): the entities call
is forced to return malformed JSON, exercising the "graceful degradation"
path where caption + qa still run on whatever validated.

Usage:
    python scripts/smoke_end2end.py                     # normal run
    python scripts/smoke_end2end.py --force-fail entities
    python scripts/smoke_end2end.py --no-risk           # skip judgment
    python scripts/smoke_end2end.py --mock              # offline canned data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from egoannot.assemble import assemble_video
from egoannot.config import get_settings
from egoannot.db import Video, init_engine, session_scope
from egoannot.logging import configure_logging
from egoannot.orchestrator import annotate_video
from egoannot.vlm.client import ChatMessage, VLMClient, VLMResponse
from egoannot.vlm.mock import MockVLMClient

EVENT_START = 8.0
EVENT_END = 14.0
CLIP_DURATION = 30.0


def _make_event_clip(dst: Path) -> None:
    """Generate the reference clip: gray backdrop with a red block visible
    only during [EVENT_START, EVENT_END].
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    drawbox_expr = (
        f"drawbox=x=280:y=140:w=80:h=200:color=red@1:t=fill:"
        f"enable='between(t,{EVENT_START},{EVENT_END})'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c=0x808080:s=640x480:d={CLIP_DURATION}:r=30",
        "-vf", drawbox_expr,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"synthetic clip generation failed: {proc.stderr.strip()}")


class ForceFailClient:
    """Wrap another client; force one sub-task to return malformed JSON.

    The forced task fails validation on both attempts (attempt 1 gets
    ``NOT JSON``, the corrective retry gets a JSON object that lacks
    required fields), exercising the ``ok=False`` code path end to end.
    """

    def __init__(self, inner: VLMClient | MockVLMClient, task_to_fail: str) -> None:
        self._inner = inner
        self._task = task_to_fail
        self._call_count: dict[str, int] = {}

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> "ForceFailClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def chat(self, messages: list[ChatMessage], **kwargs: object) -> VLMResponse:
        task_name = _extract_task_name(messages)
        if task_name == self._task:
            self._call_count[task_name] = self._call_count.get(task_name, 0) + 1
            if self._call_count[task_name] == 1:
                text = "NOT JSON — provoking a validation failure."
            else:
                text = '{"nonsense": true}'
            return VLMResponse(
                text=text,
                prompt_tokens=0,
                completion_tokens=len(text),
                total_tokens=len(text),
                raw={"forced_fail": True, "task": task_name},
            )
        return await self._inner.chat(messages, **kwargs)


def _extract_task_name(messages: list[ChatMessage]) -> str:
    for m in messages:
        if m.role == "system" and isinstance(m.content, str):
            for line in m.content.splitlines():
                if line.startswith("SUBTASK="):
                    return line.split("=", 1)[1].strip()
    return "unknown"


async def _run(args: argparse.Namespace) -> int:
    configure_logging(level="INFO")
    settings = get_settings()

    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.videos_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.frames_dir.mkdir(parents=True, exist_ok=True)

    # Fresh DB per smoke run for clarity.
    if settings.paths.db_path.exists():
        settings.paths.db_path.unlink()
    init_engine(settings.paths.db_path)

    video_id = args.video_id
    video_path = settings.paths.videos_dir / f"{video_id}.mp4"
    frame_dir = settings.paths.frames_dir / video_id
    if frame_dir.exists():
        for p in frame_dir.glob("*"):
            p.unlink()

    if args.video is None:
        print(f"[e2e] generating synthetic event clip -> {video_path}")
        print(
            f"[e2e] ground truth: red block visible ONLY t={EVENT_START:.1f}s .. "
            f"t={EVENT_END:.1f}s (clip is {CLIP_DURATION:.0f}s total)"
        )
        _make_event_clip(video_path)
    else:
        video_path = Path(args.video)
        print(f"[e2e] using clip: {video_path} (ground truth unknown)")

    with session_scope() as s:
        s.add(
            Video(
                id=video_id,
                source_dataset="jaad",
                source_path=str(video_path),
                selected=True,
                status="curated",
            )
        )

    # Wire up client (real or mock, optionally wrapped for forced failure).
    inner_client: VLMClient | MockVLMClient
    if args.mock:
        inner_client = MockVLMClient()
    else:
        inner_client = VLMClient()

    if args.force_fail:
        client = ForceFailClient(inner_client, args.force_fail)
    else:
        client = inner_client

    try:
        status = await annotate_video(
            video_id, client=client, skip_risk=args.no_risk
        )
        print(f"[e2e] annotate status = {status}")
    finally:
        await client.aclose()

    if status == "failed":
        print("[e2e] video failed; skipping assemble.")
        return 2

    payload = assemble_video(video_id)
    print("[e2e] --- assembled annotation ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("[e2e] --- time_span alignment vs ground truth ---")
    print(f"[e2e] ground truth event: [{EVENT_START:.2f}, {EVENT_END:.2f}]")
    _report_alignment(payload)
    return 0


def _report_alignment(payload: dict[str, object]) -> None:
    key_elements = payload.get("key_elements") or []
    print(f"[e2e]   key_elements ({len(key_elements)}):")
    for k in key_elements:  # type: ignore[union-attr]
        assert isinstance(k, dict)
        ts = k.get("time_span") or [None, None]
        print(
            f"[e2e]     - {k.get('label')!r} category={k.get('category')} "
            f"time_span={ts} position={k.get('position')} motion={k.get('motion')}"
        )
    risks = payload.get("risk_labels") or []
    print(f"[e2e]   risks ({len(risks)}):")
    for r in risks:  # type: ignore[union-attr]
        assert isinstance(r, dict)
        ts = r.get("time_span") or [None, None]
        print(f"[e2e]     - {r.get('type')} sev={r.get('severity')} time_span={ts}")
    qa_pairs = payload.get("qa_pairs") or []
    print(f"[e2e]   qa evidence spans:")
    for q in qa_pairs:  # type: ignore[union-attr]
        assert isinstance(q, dict)
        print(
            f"[e2e]     - {q.get('qid')} type={q.get('type')} "
            f"evidence_time_span={q.get('evidence_time_span')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="VID_990001")
    parser.add_argument("--video", default=None, help="Use this clip instead of the synthetic one")
    parser.add_argument("--mock", action="store_true", help="Use MockVLMClient (offline canned data)")
    parser.add_argument(
        "--force-fail",
        choices=["scene", "entities", "events", "judgment", "caption", "qa"],
        default=None,
        help="Force one sub-task to return malformed JSON (partial-failure demo)",
    )
    parser.add_argument("--no-risk", action="store_true", help="Skip the judgment sub-task")
    args = parser.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
