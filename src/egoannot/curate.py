"""Curate stage: two-pass filter of ingested videos down to ~500.

Pass 1 (metadata, no model): drop videos whose duration is outside
``[curate.min_sec, curate.max_sec]`` or whose ffprobe metadata is
unusable. Set ``selected=False`` on rejects and record ``select_reason``.

Pass 2 (cheap model call, survivors only): sample a few frames and
classify whether the viewpoint is a forward-facing eye-level walking
scene. Videos the classifier says to drop are marked ``selected=False``.

Selection: take up to ``curate.target_count`` (default 500), balanced
across source datasets. Balancing is done by round-robin picking from
per-dataset queues so no dataset dominates.

Idempotent: videos already in a terminal curated state (either kept or
explicitly dropped) are skipped on re-run.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy import select, update

from .config import get_settings
from .db import Video, session_scope
from .media.frames import FrameExtractionError, encode_data_uri, probe_video
from .schemas.subtasks import CurateClassification
from .tasks.base import run_image_subtask
from .vlm.client import ChatMessage, VLMClient, VLMResponse

_log = structlog.stdlib.get_logger(__name__)


class _Client(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = ...,
        max_output_tokens: int | None = ...,
        temperature: float | None = ...,
    ) -> VLMResponse: ...

    async def aclose(self) -> None: ...


async def curate_all(
    *,
    client: _Client | None = None,
    target_count: int | None = None,
    concurrency: int | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Run the full curate stage.

    Returns ``(kept, dropped)``. Idempotent unless ``force=True`` which
    re-evaluates already-curated videos.
    """
    settings = get_settings()
    target = target_count or settings.curate.target_count
    conc = concurrency or settings.curate.model_pass_concurrency

    metadata_survivors = _metadata_pass(force=force)
    _log.info("curate_metadata_pass_done", survivors=len(metadata_survivors))
    if not metadata_survivors:
        return 0, 0

    own_client = client is None
    if client is None:
        active_client: _Client = VLMClient()
    else:
        active_client = client

    try:
        classifications = await _model_pass(
            metadata_survivors, client=active_client, concurrency=conc
        )
    finally:
        if own_client:
            await active_client.aclose()

    kept, dropped = _apply_selection(classifications, target=target)
    _log.info("curate_done", kept=kept, dropped=dropped, target=target)
    return kept, dropped


# --------------------------------------------------------------------- pass 1

def _metadata_pass(*, force: bool) -> list[str]:
    """Probe every candidate row; reject those that fail basic gates.

    Returns the list of video_ids that survive to pass 2.
    """
    settings = get_settings()
    min_sec, max_sec = settings.curate.min_sec, settings.curate.max_sec

    with session_scope() as session:
        stmt = select(Video)
        if not force:
            stmt = stmt.where(Video.status == "pending")
        rows = session.execute(stmt).scalars().all()
        candidate_ids = [(r.id, r.source_path) for r in rows]

    survivors: list[str] = []
    for vid, src in candidate_ids:
        source = Path(src)
        try:
            meta = probe_video(source)
        except FrameExtractionError as e:
            _reject(vid, f"metadata_ffprobe: {e}")
            continue
        if meta.duration_sec < min_sec:
            _reject(vid, f"duration<{min_sec}s ({meta.duration_sec:.2f})")
            continue
        if meta.duration_sec > max_sec:
            _reject(vid, f"duration>{max_sec}s ({meta.duration_sec:.2f})")
            continue
        if min(meta.width, meta.height) < 240:
            _reject(vid, f"resolution_too_small ({meta.width}x{meta.height})")
            continue
        with session_scope() as session:
            v = session.get(Video, vid)
            if v is None:
                continue
            v.duration_sec = meta.duration_sec
            v.fps = meta.fps
            v.resolution_w = meta.width
            v.resolution_h = meta.height
        survivors.append(vid)
    return survivors


def _reject(video_id: str, reason: str) -> None:
    with session_scope() as session:
        v = session.get(Video, video_id)
        if v is None:
            return
        v.selected = False
        v.select_reason = reason
        # Keep status=pending so retry-failed does not sweep these; they
        # are explicit drops, not failures.
    _log.info("curate_reject", video_id=video_id, reason=reason)


# --------------------------------------------------------------------- pass 2

async def _model_pass(
    video_ids: list[str],
    *,
    client: _Client,
    concurrency: int,
) -> list[tuple[str, str, bool, str]]:
    """Return one (video_id, source_dataset, keep_bool, reason) per row.

    keep_bool=True when the model classifier says the viewpoint is a
    forward-facing eye-level walking scene.
    """
    sem = asyncio.Semaphore(concurrency)
    load: dict[str, tuple[str, str]] = {}
    with session_scope() as session:
        rows = session.execute(
            select(Video.id, Video.source_dataset, Video.source_path).where(
                Video.id.in_(video_ids)
            )
        ).all()
        for row in rows:
            load[row.id] = (row.source_dataset, row.source_path)

    async def _one(vid: str) -> tuple[str, str, bool, str]:
        async with sem:
            dataset, source = load[vid]
            try:
                image_content = _sample_classifier_frames(Path(source))
            except FrameExtractionError as e:
                return vid, dataset, False, f"pass2_ffmpeg: {e}"
            validated, ok = await run_image_subtask(
                client=client,
                task_name="curate",
                response_model=CurateClassification,
                image_content=image_content,
                video_id=vid,
                segment_idx=-1,
                persist=False,
            )
            if not ok or validated is None:
                # Classifier itself failed — conservative default: drop.
                return vid, dataset, False, "pass2_classifier_failed"
            return vid, dataset, bool(validated.keep), validated.reason

    results = await asyncio.gather(*[_one(v) for v in video_ids])
    return list(results)


def _sample_classifier_frames(source: Path) -> list[dict[str, object]]:
    """Grab a handful of frames for the cheap curate classifier.

    Uses ffmpeg to write a few thumbnails to a scratch dir; encodes them
    as base64 data URIs and cleans up. Does NOT populate the pipeline's
    permanent frames directory — that only happens once curate has kept a
    clip and the ``frames`` stage runs.
    """
    settings = get_settings()
    n = max(2, settings.curate.classifier_frames)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FrameExtractionError("ffmpeg not on PATH")

    tmp = Path(tempfile.mkdtemp(prefix="egoannot-curate-"))
    try:
        pattern = tmp / "f_%03d.jpg"
        # thumbnail filter picks n visually representative frames.
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            "thumbnail,scale='min(720,iw)':-2",
            "-frames:v",
            str(n),
            "-q:v",
            "6",
            str(pattern),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise FrameExtractionError(f"ffmpeg failed: {proc.stderr.strip()}")
        frames = sorted(tmp.glob("f_*.jpg"))
        if not frames:
            raise FrameExtractionError("no classifier frames produced")
        content: list[dict[str, object]] = []
        for f in frames:
            content.append({"type": "text", "text": "[classifier_frame]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_data_uri(f, max_long_side=720)},
                }
            )
        return content
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------- selection

def _apply_selection(
    classifications: list[tuple[str, str, bool, str]],
    *,
    target: int,
) -> tuple[int, int]:
    """Balance across datasets, mark selected/reason, transition status."""
    per_ds: dict[str, deque[tuple[str, str]]] = defaultdict(deque)
    dropped_here = 0
    for vid, dataset, keep, reason in classifications:
        if not keep:
            _reject(vid, f"pass2: {reason}"[:200])
            dropped_here += 1
            continue
        per_ds[dataset].append((vid, reason))

    picked: list[tuple[str, str]] = []
    non_empty = [k for k in per_ds if per_ds[k]]
    while len(picked) < target and non_empty:
        for k in list(non_empty):
            if not per_ds[k]:
                non_empty.remove(k)
                continue
            picked.append(per_ds[k].popleft())
            if len(picked) >= target:
                break

    # Videos left in per_ds queues past the cap are DROPPED (excess).
    for ds, queue in per_ds.items():
        for vid, _reason in queue:
            _reject(vid, f"over_target({ds})")
            dropped_here += 1

    with session_scope() as session:
        for vid, reason in picked:
            v = session.get(Video, vid)
            if v is None:
                continue
            v.selected = True
            v.select_reason = f"pass2: {reason}"[:200] if reason else "pass2: kept"
            v.status = "curated"
        session.execute(
            update(Video)
            .where(Video.selected.is_(False), Video.status == "pending")
            .values(status="pending")  # no-op: keep pending for clarity
        )
    return len(picked), dropped_here


__all__ = ["curate_all"]
