"""Deterministic final-JSON assembly.

Python owns the final schema. This module consumes validated ``TaskResult``
rows for one video, merges results across segments, applies safe defaults
for anything missing, generates ids/paths/qids/splits, clamps all
time_spans to ``[0, duration_sec]``, and validates the whole thing against
:class:`FinalAnnotation` before returning it.

Aggregation rules across segments:
    - ``environment``: taken from the FIRST segment's scene response.
    - ``scene_category``: taken from the FIRST segment's scene response.
    - ``caption``: SINGLE-SEGMENT videos use the segment caption directly.
      MULTI-SEGMENT videos issue ONE additional text-only synthesis call
      (task ``caption_merge``) that takes the per-segment captions plus a
      compact scene summary and produces one coherent first-person
      caption. If the merge call is unavailable (no client passed or the
      call fails), we fall back to a length-safe join of segment captions.
      Regardless of source, the final caption is clamped to the schema
      max length at a sentence boundary.
    - ``key_elements``: union across segments. Entries with the same label
      are merged into a single row with the widest time_span; the maximum
      importance is retained.
    - ``risk_labels``: concatenation of every risk from every segment;
      each ``description`` is clamped to the RiskLabel max length.
    - ``walkability``: worst across segments (not_passable > with_caution
      > passable > unknown).
    - ``acceptable_actions``: union across segments, ordered stably.
    - ``qa_pairs``: concatenation across segments, capped at
      ``tasks.num_qa`` when a single segment; multi-segment videos keep
      all pairs up to a schema limit of 16.
    - ``sampling_interval_sec``: ``duration_sec / num_candidate_frames``,
      informing consumers of the effective time resolution.

Time-span clamping:
    Every [start, end] pair in key_elements, risk_labels, and qa
    evidence_time_span is clamped to [0, duration_sec]. This tolerates
    model drift past the clip end without corrupting the schema.

String-length clamping:
    Every free-text field that is aggregated across segments (caption,
    risk description) is clamped to its schema max at a sentence
    boundary, then at word boundary, and only as a last resort mid-word.
    A ``string_clamped`` structlog event fires on any truncation so the
    overflow is discoverable.

Defaults for missing sub-tasks:
    - No scene:      environment=all-unknown, scene_category=[], caption placeholder.
    - No entities:   key_elements=[].
    - No events:     dropped (events don't appear in the final schema).
    - No judgment:   walkability=unknown, risks=[], actions=[observe].
    - No caption:    ``caption`` synthesised from scene_summary if available,
      else placeholder ``First-person navigation view.``.
    - No qa:         qa_pairs=[].

Split:
    Deterministic 80/10/10 bucketing from ``sha1(video_id)`` unless a
    dataset-provided split is already set on the Video row (in which case
    it takes precedence).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

import structlog
from sqlalchemy import select

from .config import get_settings
from .db import Annotation, TaskResult, Video, session_scope
from .db import Segment as DBSegment
from .schemas.annotation import (
    Environment,
    FinalAnnotation,
    KeyElement,
    QAPair,
    RiskLabel,
)
from .schemas.enums import (
    Action,
    AnswerType,
    CameraMotion,
    Category,
    CrowdLevel,
    Distance,
    Importance,
    Lighting,
    LocationType,
    Motion,
    Position,
    QAType,
    RiskType,
    Severity,
    Split,
    Walkability,
    Weather,
)
from .schemas.subtasks import CaptionResponse
from .tasks.base import run_text_subtask
from .vlm.client import ChatMessage, VLMResponse

_log = structlog.stdlib.get_logger(__name__)


# Schema field maxima; keep in sync with schemas/annotation.py.
_CAPTION_MAX_LEN = 700
_RISK_DESCRIPTION_MAX_LEN = 300

# Target the model at a slightly-shorter length so it aims below the cap
# and leaves headroom for the safety clamp.
_CAPTION_MERGE_TARGET_LEN = 600

_MERGE_TASK = "caption_merge"


class _MergeClient(Protocol):
    """Minimum surface the merge call needs from a VLM client."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = ...,
        max_output_tokens: int | None = ...,
        temperature: float | None = ...,
    ) -> VLMResponse: ...

    async def aclose(self) -> None: ...


_WALKABILITY_ORDER: dict[Walkability, int] = {
    Walkability.not_passable: 3,
    Walkability.passable_with_caution: 2,
    Walkability.passable: 1,
    Walkability.unknown: 0,
}

_IMPORTANCE_ORDER: dict[Importance, int] = {
    Importance.high: 3,
    Importance.medium: 2,
    Importance.low: 1,
}


_MergeClientFactory = Callable[[], "_MergeClient"]


class AssembleError(RuntimeError):
    """Raised when a video row is missing or the assembled dict fails
    final-schema validation."""


def assemble_video(
    video_id: str,
    *,
    persist: bool = True,
    merge_client_factory: _MergeClientFactory | None = None,
) -> dict[str, Any]:
    """Assemble the final annotation dict for one video.

    Returns the JSON-serialisable dict (already schema-validated). When
    ``persist`` is True, an ``Annotation`` row is upserted and the Video
    status advances to ``assembled``.

    ``merge_client_factory`` is called once, ONLY when the video has
    multiple segments AND no cached merge result exists. The factory
    must return a fresh client bound to no other event loop; the merge
    call runs inside its own :func:`asyncio.run` scope and closes the
    client at the end. Pass ``None`` (default) to skip the model call
    entirely — assembly falls back to a length-safe join. Never crashes
    on caption length.
    """
    structlog.contextvars.bind_contextvars(video_id=video_id)
    try:
        with session_scope() as session:
            v = session.get(Video, video_id)
            if v is None:
                raise AssembleError(f"unknown video_id: {video_id}")

            segments = (
                session.execute(
                    select(DBSegment)
                    .where(DBSegment.video_id == video_id)
                    .order_by(DBSegment.idx)
                )
                .scalars()
                .all()
            )
            task_rows = (
                session.execute(
                    select(TaskResult)
                    .where(TaskResult.video_id == video_id, TaskResult.ok.is_(True))
                    .order_by(TaskResult.segment_idx, TaskResult.task_name)
                )
                .scalars()
                .all()
            )

            per_seg: dict[int, dict[str, Any]] = {}
            merged_caption_cached: str | None = None
            for row in task_rows:
                if not row.parsed_json:
                    continue
                try:
                    parsed = json.loads(row.parsed_json)
                except json.JSONDecodeError:
                    _log.warning(
                        "assemble_bad_parsed_json",
                        segment_idx=row.segment_idx,
                        task=row.task_name,
                    )
                    continue
                if row.task_name == _MERGE_TASK and row.segment_idx == -1:
                    merged_caption_cached = str(parsed.get("caption") or "")
                    continue
                per_seg.setdefault(row.segment_idx, {})[row.task_name] = parsed

            annotation_dict = _build_annotation(
                video=v,
                segment_indices=[s.idx for s in segments],
                per_seg=per_seg,
                merge_client_factory=merge_client_factory,
                cached_merged_caption=merged_caption_cached,
            )

            validated = FinalAnnotation.model_validate(annotation_dict)
            payload = validated.model_dump(mode="json")

            if persist:
                existing = session.get(Annotation, video_id)
                serialized = json.dumps(payload, ensure_ascii=False)
                if existing is None:
                    session.add(Annotation(video_id=video_id, payload_json=serialized))
                else:
                    existing.payload_json = serialized
                v.status = "assembled"
                v.error = None
                v.split = payload["split"]

            _log.info(
                "assemble_ok",
                key_elements=len(payload["key_elements"]),
                risks=len(payload["risk_labels"]),
                qa=len(payload["qa_pairs"]),
                walkability=payload["walkability"],
                sampling_interval_sec=payload["sampling_interval_sec"],
            )
            return payload
    finally:
        structlog.contextvars.unbind_contextvars("video_id")


# --------------------------------------------------------------------- builders


def _clamp(t: float, duration: float) -> float:
    if t < 0.0:
        return 0.0
    if t > duration:
        return duration
    return t


def _clamp_span(start: float, end: float, duration: float) -> tuple[float, float]:
    s = _clamp(start, duration)
    e = _clamp(end, duration)
    if e < s:
        e = s
    return s, e


_SENTENCE_END_RE = re.compile(r"[\.!\?]\s")


def _clean_truncate(text: str, max_len: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``<=max_len`` chars at the cleanest boundary.

    Prefer a sentence boundary (``.``/``!``/``?`` + whitespace); fall back
    to a word boundary; last resort is a hard slice. Trailing whitespace
    is stripped. Returns ``(text, truncated)``.
    """
    if len(text) <= max_len:
        return text, False

    window = text[:max_len]
    best_end = -1
    for m in _SENTENCE_END_RE.finditer(window):
        best_end = m.end()
    if best_end > max_len // 2:
        return window[:best_end].rstrip(), True

    space = window.rfind(" ")
    if space > max_len // 2:
        return window[:space].rstrip(), True

    return window.rstrip(), True


def _clamp_text(text: str, max_len: int, *, field: str) -> str:
    clamped, truncated = _clean_truncate(text, max_len)
    if truncated:
        _log.warning(
            "string_clamped",
            field=field,
            max_len=max_len,
            original_len=len(text),
            clamped_len=len(clamped),
        )
    return clamped


def _build_annotation(
    *,
    video: Video,
    segment_indices: list[int],
    per_seg: dict[int, dict[str, Any]],
    merge_client_factory: _MergeClientFactory | None = None,
    cached_merged_caption: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    duration = float(video.duration_sec)

    # ------------------------------------------------- scene -> env + tags
    scene_env, scene_tags = _aggregate_scene(per_seg, segment_indices)

    # ------------------------------------------------- caption
    per_seg_captions = _collect_segment_captions(per_seg, segment_indices)
    caption_text = _resolve_caption(
        video_id=video.id,
        per_seg_captions=per_seg_captions,
        per_seg=per_seg,
        segment_indices=segment_indices,
        merge_client_factory=merge_client_factory,
        cached=cached_merged_caption,
    )
    # Belt-and-suspenders: no matter where the caption came from, respect
    # the schema max. Assembly must never crash on caption length.
    caption_text = _clamp_text(caption_text, _CAPTION_MAX_LEN, field="caption")
    if not caption_text:
        caption_text = "First-person navigation view."

    # ------------------------------------------------- key_elements
    key_elements = _aggregate_entities(per_seg, segment_indices, duration)

    # ------------------------------------------------- risks / walkability / actions
    risks = _aggregate_risks(per_seg, segment_indices, duration)
    walkability = _aggregate_walkability(per_seg, segment_indices)
    actions = _aggregate_actions(per_seg, segment_indices)

    # ------------------------------------------------- qa
    qa_pairs = _aggregate_qa(
        per_seg,
        segment_indices,
        video_id=video.id,
        duration=duration,
        cap=max(settings.tasks.num_qa if len(segment_indices) <= 1 else 16, 1),
    )

    # ------------------------------------------------- privacy + split + paths
    privacy = settings.privacy.for_dataset(video.source_dataset)
    split = video.split if video.split else _deterministic_split(video.id).value
    split = _coerce_split(split).value

    video_path_rel = f"{settings.paths.videos_subdir}/{video.id}.mp4"
    frame_dir_rel = f"{settings.paths.frames_subdir}/{video.id}"

    num_candidate = int(video.num_candidate_frames)
    sampling_interval = (
        duration / num_candidate if duration > 0 and num_candidate > 0 else 0.0
    )

    return {
        "video_id": video.id,
        "split": split,
        "video_path": video_path_rel,
        "frame_dir": frame_dir_rel,
        "duration_sec": duration,
        "fps": float(video.fps),
        "candidate_fps": float(video.candidate_fps),
        "num_candidate_frames": num_candidate,
        "sampling_interval_sec": sampling_interval,
        "resolution": [int(video.resolution_w), int(video.resolution_h)],
        "scene_category": scene_tags,
        "environment": scene_env,
        "caption": caption_text,
        "key_elements": key_elements,
        "risk_labels": risks,
        "walkability": walkability,
        "acceptable_actions": actions,
        "qa_pairs": qa_pairs,
        "privacy": {
            "face_blurred": privacy.face_blurred,
            "plate_blurred": privacy.plate_blurred,
            "contains_sensitive_info": privacy.contains_sensitive_info,
        },
    }


# --------------------------------------------------------------------- helpers


def _aggregate_scene(
    per_seg: dict[int, dict[str, Any]], indices: list[int]
) -> tuple[dict[str, str], list[str]]:
    """Return (environment_dict, scene_category_tags)."""
    for idx in indices:
        scene = per_seg.get(idx, {}).get("scene")
        if scene:
            env = {
                "location_type": _coerce_enum(scene.get("location_type"), LocationType).value,
                "lighting": _coerce_enum(scene.get("lighting"), Lighting).value,
                "weather": _coerce_enum(scene.get("weather"), Weather).value,
                "crowd_level": _coerce_enum(scene.get("crowd_level"), CrowdLevel).value,
                "camera_motion": _coerce_enum(scene.get("camera_motion"), CameraMotion).value,
            }
            raw_tags = scene.get("scene_category") or []
            tags = [str(t).strip() for t in raw_tags if str(t).strip()][:6]
            return env, tags
    default_env = Environment(
        location_type=LocationType.unknown,
        lighting=Lighting.unknown,
        weather=Weather.unknown,
        crowd_level=CrowdLevel.low,
        camera_motion=CameraMotion.unknown,
    ).model_dump(mode="json")
    return default_env, []


def _collect_segment_captions(
    per_seg: dict[int, dict[str, Any]], indices: list[int]
) -> list[tuple[int, str]]:
    """Return the non-empty caption for each segment, in order."""
    out: list[tuple[int, str]] = []
    for idx in indices:
        cap = per_seg.get(idx, {}).get("caption")
        if cap and cap.get("caption"):
            out.append((idx, str(cap["caption"]).strip()))
    return out


def _collect_scene_summaries(
    per_seg: dict[int, dict[str, Any]], indices: list[int]
) -> list[str]:
    out: list[str] = []
    for idx in indices:
        scene = per_seg.get(idx, {}).get("scene")
        if scene and scene.get("scene_summary"):
            out.append(str(scene["scene_summary"]).strip())
    return out


def _resolve_caption(
    *,
    video_id: str,
    per_seg_captions: list[tuple[int, str]],
    per_seg: dict[int, dict[str, Any]],
    segment_indices: list[int],
    merge_client_factory: _MergeClientFactory | None,
    cached: str | None,
) -> str:
    """Pick or synthesise the final caption for the video."""
    if not per_seg_captions:
        for idx in segment_indices:
            scene_raw = per_seg.get(idx, {}).get("scene")
            if scene_raw and scene_raw.get("scene_summary"):
                return str(scene_raw["scene_summary"]).strip()
        return ""

    if len(per_seg_captions) == 1:
        return per_seg_captions[0][1]

    if cached:
        return cached

    if merge_client_factory is not None:
        merged = _merge_captions_via_model(
            video_id=video_id,
            per_seg_captions=per_seg_captions,
            scene_summaries=_collect_scene_summaries(per_seg, segment_indices),
            factory=merge_client_factory,
        )
        if merged:
            return merged

    return _length_safe_join([c for _, c in per_seg_captions])


def _length_safe_join(parts: list[str]) -> str:
    """Fallback merge when no model is available.

    Joins by ``"; "`` and lets the caller clamp; documented for its
    predictability under crash conditions.
    """
    return "; ".join(parts).strip()


def _merge_captions_via_model(
    *,
    video_id: str,
    per_seg_captions: list[tuple[int, str]],
    scene_summaries: list[str],
    factory: _MergeClientFactory,
) -> str:
    """Issue one text-only synthesis call to summarise per-segment captions.

    A fresh client is instantiated INSIDE the coroutine that runs on a
    dedicated event loop, so no httpx state is shared across
    ``asyncio.run`` boundaries. The result is persisted as a TaskResult
    row keyed by ``(video_id, segment_idx=-1, task_name="caption_merge")``
    so repeated ``assemble`` invocations do not re-hit the model.
    """
    context = {
        "target_max_chars": _CAPTION_MERGE_TARGET_LEN,
        "scene_summaries": scene_summaries[:8],
        "segment_captions": [
            {"segment_idx": idx, "caption": text} for idx, text in per_seg_captions
        ],
    }
    context_json = json.dumps(context, ensure_ascii=False)

    async def _run() -> tuple[Any, bool]:
        client = factory()
        try:
            return await run_text_subtask(
                client=client,
                task_name=_MERGE_TASK,
                response_model=CaptionResponse,
                context_json=context_json,
                video_id=video_id,
                segment_idx=-1,
                max_output_tokens=None,
            )
        finally:
            await client.aclose()

    try:
        validated, ok = _run_sync(_run)
    except Exception as exc:
        _log.warning("caption_merge_call_failed", err=repr(exc))
        return ""

    if not ok or validated is None:
        return ""
    return str(validated.caption).strip()


def _run_sync(coro_factory: Any) -> Any:
    """Run an async coroutine from sync code.

    ``assemble_video`` is synchronous (called from the CLI + tests). If
    we are not inside a running loop, ``asyncio.run`` is enough. If we
    ARE inside a loop (e.g. ``run --all``), we hand the coroutine to a
    worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    import concurrent.futures

    def _worker() -> Any:
        return asyncio.new_event_loop().run_until_complete(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_worker).result()


def _aggregate_entities(
    per_seg: dict[int, dict[str, Any]], indices: list[int], duration: float
) -> list[dict[str, Any]]:
    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for idx in indices:
        ent_wrapper = per_seg.get(idx, {}).get("entities") or {}
        for e in ent_wrapper.get("entities", []) or []:
            label = str(e.get("label", "")).strip()
            if not label:
                continue
            first = float(e.get("first_seen_sec") or 0.0)
            last = float(e.get("last_seen_sec") or first)
            first, last = _clamp_span(first, last, duration)
            category = _coerce_enum(e.get("category"), Category)
            importance = _coerce_enum(e.get("importance"), Importance)
            position = _coerce_enum(e.get("position"), Position)
            distance = _coerce_enum(e.get("distance"), Distance)
            motion = _coerce_enum(e.get("motion"), Motion)
            overhead = bool(e.get("overhead", False))

            if label in merged:
                m = merged[label]
                m["time_span"] = [min(m["time_span"][0], first), max(m["time_span"][1], last)]
                if _IMPORTANCE_ORDER[importance] > _IMPORTANCE_ORDER[Importance(m["importance"])]:
                    m["importance"] = importance.value
                m["overhead"] = m["overhead"] or overhead
            else:
                merged[label] = {
                    "label": label,
                    "category": category.value,
                    "importance": importance.value,
                    "time_span": [first, last],
                    "position": position.value,
                    "distance": distance.value,
                    "motion": motion.value,
                    "overhead": overhead,
                }
    out = list(merged.values())[:16]
    return [KeyElement.model_validate(x).model_dump(mode="json") for x in out]


def _aggregate_risks(
    per_seg: dict[int, dict[str, Any]], indices: list[int], duration: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in indices:
        judg = per_seg.get(idx, {}).get("judgment") or {}
        for r in judg.get("risks", []) or []:
            start = float(r.get("start_sec") or 0.0)
            end = float(r.get("end_sec") or start)
            start, end = _clamp_span(start, end, duration)
            raw_desc = str(r.get("description") or "").strip() or "No description."
            desc = _clamp_text(raw_desc, _RISK_DESCRIPTION_MAX_LEN, field="risk.description")
            item = {
                "type": _coerce_enum(r.get("risk_type"), RiskType).value,
                "severity": _coerce_enum(r.get("severity"), Severity).value,
                "time_span": [start, end],
                "description": desc,
            }
            out.append(RiskLabel.model_validate(item).model_dump(mode="json"))
            if len(out) >= 8:
                return out
    return out


def _aggregate_walkability(per_seg: dict[int, dict[str, Any]], indices: list[int]) -> str:
    worst = Walkability.unknown
    seen_any = False
    for idx in indices:
        judg = per_seg.get(idx, {}).get("judgment")
        if not judg:
            continue
        w = _coerce_enum(judg.get("walkability"), Walkability)
        seen_any = True
        if _WALKABILITY_ORDER[w] > _WALKABILITY_ORDER[worst]:
            worst = w
    if not seen_any:
        return Walkability.unknown.value
    return worst.value


def _aggregate_actions(per_seg: dict[int, dict[str, Any]], indices: list[int]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for idx in indices:
        judg = per_seg.get(idx, {}).get("judgment")
        if not judg:
            continue
        for a in judg.get("actions", []) or []:
            act = _coerce_enum(a, Action).value
            seen.setdefault(act, None)
    if not seen:
        seen.setdefault(Action.observe.value, None)
    return list(seen.keys())[:8]


def _aggregate_qa(
    per_seg: dict[int, dict[str, Any]],
    indices: list[int],
    *,
    video_id: str,
    duration: float,
    cap: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counter = 1
    for idx in indices:
        qa = per_seg.get(idx, {}).get("qa") or {}
        for item in qa.get("qa", []) or []:
            evidence_start = item.get("evidence_start_sec")
            evidence_end = item.get("evidence_end_sec")
            time_span: list[float] | None = None
            if evidence_start is not None and evidence_end is not None:
                s = float(evidence_start)
                e = float(evidence_end)
                s, e = _clamp_span(s, e, duration)
                time_span = [s, e]
            pair = {
                "qid": f"{video_id}_Q{counter:03d}",
                "type": _coerce_enum(item.get("qa_type"), QAType).value,
                "question": str(item.get("question") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "evidence_elements": [
                    str(x).strip()
                    for x in (item.get("evidence_entities") or [])
                    if str(x).strip()
                ][:8],
                "evidence_time_span": time_span,
                "answer_type": _coerce_enum(item.get("answer_type"), AnswerType).value,
            }
            if not pair["question"] or not pair["answer"]:
                continue
            out.append(QAPair.model_validate(pair).model_dump(mode="json"))
            counter += 1
            if len(out) >= cap:
                return out
    return out


def _coerce_enum(value: Any, enum_cls: Any) -> Any:
    member, _ = enum_cls.coerce(value)
    return member


def _deterministic_split(video_id: str) -> Split:
    h = int(hashlib.sha1(video_id.encode("utf-8")).hexdigest(), 16)
    bucket = h % 100
    if bucket < 80:
        return Split.train
    if bucket < 90:
        return Split.val
    return Split.test


def _coerce_split(value: str) -> Split:
    member, _ = Split.coerce(value)
    return member


__all__ = ["AssembleError", "MergeClient", "MergeClientFactory", "assemble_video"]


# Public aliases so callers (CLI, tests) can annotate factory types without
# importing the private-underscored symbols.
MergeClient = _MergeClient
MergeClientFactory = _MergeClientFactory
