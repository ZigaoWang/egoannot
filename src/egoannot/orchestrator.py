"""Per-video orchestration.

Contract:

    For each segment of a video:
        1. Build image_content (base64 data URIs with [t=Xs] markers).
        2. Run image tasks (scene, entities, events, [judgment]) CONCURRENTLY.
        3. Wait for ALL image tasks to settle (validated or ok=False).
        4. Assemble a compact context JSON from the sub-tasks that
           validated successfully. Failed sub-tasks are OMITTED from the
           context so caption/qa see only trustworthy inputs.
        5. Run text-only tasks (caption, qa) CONCURRENTLY against that
           context.

Failure model:
    - A single sub-task failing (ok=False) never fails the segment.
    - A segment with all four image tasks failing marks the video ``failed``.
    - A segment where scene failed but at least one of entities/events
      succeeded still runs caption+qa on whatever validated.
    - Frame extraction failure (raised by media/frames) marks the video
      ``failed`` before any task runs.

Idempotence:
    - ``TaskResult`` rows are upserted by (video_id, segment_idx, task_name),
      so re-running annotate on a video overwrites results in place.
    - Video ``status`` monotonically advances; a re-run of an already
      ``tasks_done`` video re-executes and lands on ``tasks_done`` again.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol

import structlog
from sqlalchemy import select

from .config import get_settings
from .db import Segment, Video, session_scope
from .media.frames import (
    FrameExtractionError,
    SampledFrame,
    VideoMeta,
    extract_frames,
    probe_video,
    sample_segment_frames,
    sampled_frames_to_content,
    segment_video,
)
from .media.frames import (
    Segment as MediaSegment,
)
from .schemas.enums import CORE_TASKS, TaskName, VideoStatus
from .tasks import caption, entities, events, judgment, qa, scene
from .vlm.client import ChatMessage, VLMClient, VLMResponse

_log = structlog.stdlib.get_logger(__name__)


class _ClientLike(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = ...,
        max_output_tokens: int | None = ...,
        temperature: float | None = ...,
    ) -> VLMResponse: ...

    async def aclose(self) -> None: ...


IMAGE_TASKS_ALL: tuple[str, ...] = ("scene", "entities", "events", "judgment")
TEXT_TASKS: tuple[str, ...] = ("caption", "qa")


async def annotate_video(
    video_id: str,
    *,
    client: _ClientLike | None = None,
    skip_risk: bool = False,
    force: bool = False,
) -> str:
    """Annotate one video end-to-end. Returns the final status string.

    Args:
        video_id: existing Video row id.
        client: an open VLM client (or MockVLMClient). If None, a real
            :class:`VLMClient` is opened for this call and closed at exit.
        skip_risk: drop the judgment sub-task; assembly will use defaults
            (walkability=unknown, risks=[], actions=[observe]).
        force: re-run every sub-task even if a previous run already
            reached ``tasks_done``. Otherwise a video already at or past
            ``tasks_done`` is a no-op and its status is returned.
    """
    settings = get_settings()

    structlog.contextvars.bind_contextvars(video_id=video_id)
    try:
        row_data = _load_video_row(video_id)
        if row_data is None:
            _log.error("annotate_video_not_found")
            return VideoStatus.failed.value
        source_path, current_status, frame_dir_str, chunk_start, chunk_end = row_data

        already_done = current_status in {
            VideoStatus.tasks_done.value,
            VideoStatus.assembled.value,
        }
        if already_done and not force:
            _log.info("annotate_skip_already_done", status=current_status)
            return current_status

        # Frame extraction path — mark failed and bail if this raises.
        try:
            meta = probe_video(Path(source_path))
            # When the Video row is a chunk of a longer recording, extract
            # only that window and treat its duration as the annotation's
            # duration. ``chunk_end == 0`` means "no chunk, whole file".
            is_chunk = chunk_end > 0.0 and chunk_end > chunk_start
            effective_duration = (
                (chunk_end - chunk_start) if is_chunk else meta.duration_sec
            )
            frame_dir = Path(frame_dir_str) if frame_dir_str else _default_frame_dir(video_id)
            frames = extract_frames(
                Path(source_path),
                frame_dir,
                meta,
                start_sec=chunk_start if is_chunk else None,
                end_sec=chunk_end if is_chunk else None,
            )
        except FrameExtractionError as e:
            _log.error("frame_extraction_failed", err=str(e))
            _set_status(video_id, VideoStatus.failed.value, error=f"frames: {e}")
            return VideoStatus.failed.value

        media_segs = segment_video(effective_duration)
        if not media_segs:
            _set_status(video_id, VideoStatus.failed.value, error="no_segments")
            return VideoStatus.failed.value

        _persist_video_metadata(
            video_id=video_id,
            meta=meta,
            effective_duration=effective_duration,
            frame_dir=str(frame_dir),
            media_segs=media_segs,
            per_segment=settings.frames.per_segment,
        )
        _set_status(video_id, VideoStatus.frames_done.value)

        # Ensure a client is available; keep our own if the caller didn't supply.
        own_client = client is None
        if client is None:
            client = VLMClient()

        try:
            per_segment_results: list[dict[str, Any]] = []
            for seg in media_segs:
                sampled = sample_segment_frames(
                    frames, settings.frames.fps, seg, per_segment=settings.frames.per_segment
                )
                if not sampled:
                    _log.warning("segment_no_frames_sampled", seg_idx=seg.idx)
                    continue

                seg_result = await _run_one_segment(
                    client=client,
                    video_id=video_id,
                    segment=seg,
                    sampled=sampled,
                    skip_risk=skip_risk,
                )
                per_segment_results.append(seg_result)

            # Video-level failure check.
            if not per_segment_results:
                _set_status(video_id, VideoStatus.failed.value, error="no_segment_results")
                return VideoStatus.failed.value

            all_image_failed = all(
                not any(r["image_ok"].get(t, False) for t in CORE_TASKS if t.value in IMAGE_TASKS_ALL)
                for r in per_segment_results
            )
            if all_image_failed:
                _set_status(
                    video_id,
                    VideoStatus.failed.value,
                    error="all_core_image_tasks_failed",
                )
                return VideoStatus.failed.value

            _set_status(video_id, VideoStatus.tasks_done.value)
            return VideoStatus.tasks_done.value
        finally:
            if own_client and client is not None:
                await client.aclose()
    finally:
        structlog.contextvars.unbind_contextvars("video_id")


async def _run_one_segment(
    *,
    client: _ClientLike,
    video_id: str,
    segment: MediaSegment,
    sampled: list[SampledFrame],
    skip_risk: bool,
) -> dict[str, Any]:
    """Run all sub-tasks for one segment. Returns a per-segment status dict."""
    structlog.contextvars.bind_contextvars(segment_idx=segment.idx)
    try:
        image_content = sampled_frames_to_content(sampled)

        image_ok: dict[TaskName, bool] = {}
        image_results: dict[str, Any] = {}

        image_task_names: list[str] = ["scene", "entities", "events"]
        if not skip_risk:
            image_task_names.append("judgment")

        image_coros = [
            _dispatch_image(name, client, image_content, video_id, segment.idx)
            for name in image_task_names
        ]
        image_settled = await asyncio.gather(*image_coros, return_exceptions=True)

        for name, outcome in zip(image_task_names, image_settled, strict=True):
            if isinstance(outcome, BaseException):
                _log.error("image_subtask_raised", task=name, err=repr(outcome))
                image_ok[TaskName(name)] = False
                continue
            validated, ok = outcome
            image_ok[TaskName(name)] = ok
            if ok and validated is not None:
                image_results[name] = validated.model_dump(mode="json")

        # Build context JSON from ONLY the sub-tasks that validated OK.
        # If everything failed, still run text tasks against an empty context
        # so the model produces defensible defaults; assembly may override.
        context_payload = {
            "segment_idx": segment.idx,
            "segment_start_sec": segment.start_sec,
            "segment_end_sec": segment.end_sec,
            **image_results,
        }
        context_json = json.dumps(context_payload, ensure_ascii=False)

        text_task_names: list[str] = ["caption", "qa"]
        text_coros = [
            _dispatch_text(name, client, context_json, video_id, segment.idx)
            for name in text_task_names
        ]
        text_settled = await asyncio.gather(*text_coros, return_exceptions=True)

        text_ok: dict[TaskName, bool] = {}
        for name, outcome in zip(text_task_names, text_settled, strict=True):
            if isinstance(outcome, BaseException):
                _log.error("text_subtask_raised", task=name, err=repr(outcome))
                text_ok[TaskName(name)] = False
                continue
            _validated, ok = outcome
            text_ok[TaskName(name)] = ok

        return {
            "segment_idx": segment.idx,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "image_ok": image_ok,
            "text_ok": text_ok,
        }
    finally:
        structlog.contextvars.unbind_contextvars("segment_idx")


async def _dispatch_image(
    name: str,
    client: _ClientLike,
    image_content: list[dict[str, Any]],
    video_id: str,
    segment_idx: int,
) -> Any:
    if name == "scene":
        return await scene.run(
            client=client, image_content=image_content, video_id=video_id, segment_idx=segment_idx
        )
    if name == "entities":
        return await entities.run(
            client=client, image_content=image_content, video_id=video_id, segment_idx=segment_idx
        )
    if name == "events":
        return await events.run(
            client=client, image_content=image_content, video_id=video_id, segment_idx=segment_idx
        )
    if name == "judgment":
        return await judgment.run(
            client=client, image_content=image_content, video_id=video_id, segment_idx=segment_idx
        )
    raise ValueError(f"unknown image task: {name}")


async def _dispatch_text(
    name: str,
    client: _ClientLike,
    context_json: str,
    video_id: str,
    segment_idx: int,
) -> Any:
    if name == "caption":
        return await caption.run(
            client=client,
            context_json=context_json,
            video_id=video_id,
            segment_idx=segment_idx,
        )
    if name == "qa":
        return await qa.run(
            client=client, context_json=context_json, video_id=video_id, segment_idx=segment_idx
        )
    raise ValueError(f"unknown text task: {name}")


def _load_video_row(video_id: str) -> tuple[str, str, str, float, float] | None:
    with session_scope() as session:
        v = session.get(Video, video_id)
        if v is None:
            return None
        return (
            v.source_path,
            v.status,
            v.frame_dir,
            float(v.chunk_start_sec or 0.0),
            float(v.chunk_end_sec or 0.0),
        )


def _default_frame_dir(video_id: str) -> Path:
    return get_settings().paths.frames_dir / video_id


def _persist_video_metadata(
    *,
    video_id: str,
    meta: VideoMeta,
    effective_duration: float,
    frame_dir: str,
    media_segs: list[MediaSegment],
    per_segment: int,
) -> None:
    total_candidate = per_segment * len(media_segs)
    with session_scope() as session:
        v = session.get(Video, video_id)
        if v is None:
            return
        # ``duration_sec`` on the Video row is the EFFECTIVE annotation
        # duration (chunk window for chunked rows, full file otherwise).
        # Assembly emits every time_span relative to this, so keeping it
        # in one place makes the semantics consistent end-to-end.
        v.duration_sec = effective_duration
        v.fps = meta.fps
        v.resolution_w = meta.width
        v.resolution_h = meta.height
        v.frame_dir = frame_dir
        v.num_candidate_frames = total_candidate
        v.candidate_fps = (
            total_candidate / effective_duration if effective_duration > 0 else 0.0
        )

        # Replace segments in place (idempotent).
        existing = session.execute(
            select(Segment).where(Segment.video_id == video_id)
        ).scalars().all()
        for s in existing:
            session.delete(s)
        session.flush()
        for m in media_segs:
            session.add(
                Segment(
                    video_id=video_id,
                    idx=m.idx,
                    start_sec=m.start_sec,
                    end_sec=m.end_sec,
                )
            )


def _set_status(video_id: str, status: str, *, error: str | None = None) -> None:
    with session_scope() as session:
        v = session.get(Video, video_id)
        if v is None:
            return
        v.status = status
        if error is not None:
            v.error = error


__all__ = ["annotate_video"]
