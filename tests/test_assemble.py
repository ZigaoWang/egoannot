"""Assembly tests: canned TaskResult rows -> schema-valid final JSON."""

from __future__ import annotations

import json

import pytest

from egoannot.assemble import assemble_video
from egoannot.db import Annotation, Segment, TaskResult, Video, session_scope
from egoannot.schemas.annotation import FinalAnnotation
from egoannot.vlm.mock import MockVLMClient


def _mk_video(
    vid: str,
    *,
    duration: float = 30.0,
    dataset: str = "jaad",
    num_segments: int = 1,
    num_candidate_frames: int | None = None,
) -> None:
    ncf = num_candidate_frames if num_candidate_frames is not None else 12 * num_segments
    with session_scope() as s:
        s.add(
            Video(
                id=vid,
                source_dataset=dataset,
                source_path=f"/tmp/{vid}.mp4",
                selected=True,
                duration_sec=duration,
                fps=30.0,
                resolution_w=640,
                resolution_h=480,
                num_candidate_frames=ncf,
                candidate_fps=ncf / duration,
                status="tasks_done",
            )
        )
        seg_len = duration / num_segments
        for i in range(num_segments):
            s.add(
                Segment(
                    video_id=vid,
                    idx=i,
                    start_sec=i * seg_len,
                    end_sec=(i + 1) * seg_len,
                )
            )


def _mk_task(vid: str, task: str, payload: dict, segment_idx: int = 0) -> None:
    with session_scope() as s:
        s.add(
            TaskResult(
                video_id=vid,
                segment_idx=segment_idx,
                task_name=task,
                raw_response=json.dumps(payload),
                parsed_json=json.dumps(payload),
                ok=True,
                attempts=1,
            )
        )


def _load_annotation(vid: str) -> dict:
    with session_scope() as s:
        ann = s.get(Annotation, vid)
        assert ann is not None
        return json.loads(ann.payload_json)


def test_assemble_full_happy_path(tmp_pipeline):
    vid = "VID_000042"
    _mk_video(vid)
    _mk_task(vid, "scene", {
        "location_type": "outdoor",
        "scene_category": ["outdoor_sidewalk"],
        "lighting": "normal",
        "weather": "clear",
        "crowd_level": "low",
        "camera_motion": "walking",
        "scene_summary": "Walking outdoors.",
    })
    _mk_task(vid, "entities", {
        "entities": [
            {
                "label": "pedestrian",
                "category": "person",
                "importance": "high",
                "position": "front",
                "distance": "near",
                "motion": "crossing",
                "first_seen_sec": 8.0,
                "last_seen_sec": 14.0,
                "overhead": False,
            }
        ]
    })
    _mk_task(vid, "events", {
        "events": [{
            "description": "Person crosses.",
            "start_sec": 8.0,
            "end_sec": 14.0,
            "involves": ["pedestrian"],
            "caused_by": None,
        }]
    })
    _mk_task(vid, "judgment", {
        "walkability": "passable_with_caution",
        "actions": ["slow_down", "observe"],
        "risks": [{
            "risk_type": "person_crossing",
            "severity": "medium",
            "start_sec": 8.0,
            "end_sec": 14.0,
            "description": "Pedestrian crossing ahead.",
            "related_entities": ["pedestrian"],
        }],
        "confidence": 0.8,
    })
    _mk_task(vid, "caption", {"caption": "Walking with a crossing pedestrian ahead."})
    _mk_task(vid, "qa", {"qa": [
        {
            "qa_type": "risk",
            "question": "Any dynamic obstacles?",
            "answer": "Yes.",
            "evidence_entities": ["pedestrian"],
            "evidence_start_sec": 8.0,
            "evidence_end_sec": 14.0,
            "answer_type": "boolean",
        }
    ]})

    payload = assemble_video(vid)
    FinalAnnotation.model_validate(payload)  # schema-valid
    assert payload["video_id"] == vid
    assert payload["walkability"] == "passable_with_caution"
    assert payload["key_elements"][0]["label"] == "pedestrian"
    assert payload["qa_pairs"][0]["qid"] == f"{vid}_Q001"
    assert 0.0 <= payload["sampling_interval_sec"] <= 30.0


def test_assemble_clamps_time_spans(tmp_pipeline):
    vid = "VID_000043"
    _mk_video(vid, duration=20.0)
    _mk_task(vid, "scene", {
        "location_type": "outdoor",
        "scene_category": [],
        "lighting": "normal",
        "weather": "clear",
        "crowd_level": "low",
        "camera_motion": "walking",
        "scene_summary": "OK.",
    })
    _mk_task(vid, "entities", {
        "entities": [
            {
                "label": "x",
                "category": "other",
                "importance": "low",
                "position": "front",
                "distance": "unknown",
                "motion": "unknown",
                "first_seen_sec": 15.0,
                "last_seen_sec": 999.0,  # past end
                "overhead": False,
            }
        ]
    })
    _mk_task(vid, "judgment", {
        "walkability": "passable",
        "actions": [],
        "risks": [],
        "confidence": 0.5,
    })
    _mk_task(vid, "caption", {"caption": "OK."})
    _mk_task(vid, "qa", {"qa": []})
    payload = assemble_video(vid)
    assert payload["key_elements"][0]["time_span"] == [15.0, 20.0]


def test_assemble_defaults_when_all_optional_missing(tmp_pipeline):
    vid = "VID_000044"
    _mk_video(vid)
    # scene only.
    _mk_task(vid, "scene", {
        "location_type": "indoor",
        "scene_category": [],
        "lighting": "normal",
        "weather": "unknown",
        "crowd_level": "low",
        "camera_motion": "walking",
        "scene_summary": "Indoors.",
    })
    payload = assemble_video(vid)
    FinalAnnotation.model_validate(payload)
    assert payload["walkability"] == "unknown"
    assert payload["acceptable_actions"] == ["observe"]
    assert payload["key_elements"] == []
    assert payload["risk_labels"] == []
    assert payload["qa_pairs"] == []
    assert payload["caption"] == "Indoors."


def test_assemble_deterministic_split(tmp_pipeline):
    _mk_video("VID_000001")
    _mk_task("VID_000001", "scene", {
        "location_type": "outdoor",
        "scene_category": [],
        "lighting": "normal",
        "weather": "clear",
        "crowd_level": "low",
        "camera_motion": "walking",
        "scene_summary": "Outdoor.",
    })
    p1 = assemble_video("VID_000001")
    p2 = assemble_video("VID_000001")
    assert p1["split"] == p2["split"]


def test_assemble_missing_video_raises(tmp_pipeline):
    from egoannot.assemble import AssembleError
    with pytest.raises(AssembleError):
        assemble_video("VID_999999")


# --------------------------------------------------- multi-segment captions


_LONG_SEG_CAPTION = (
    "The walker moves ahead in this segment, notices pedestrians on both "
    "sides, and adjusts speed accordingly to avoid contact while continuing "
    "to observe the surroundings for further changes in the environment."
)


def _seed_multi_segment_video(vid: str, *, num_segments: int) -> None:
    _mk_video(vid, duration=260.0, num_segments=num_segments)
    for i in range(num_segments):
        _mk_task(
            vid,
            "scene",
            {
                "location_type": "outdoor",
                "scene_category": ["outdoor_walkway"],
                "lighting": "normal",
                "weather": "clear",
                "crowd_level": "low",
                "camera_motion": "walking",
                "scene_summary": f"Segment {i}: walking a corridor.",
            },
            segment_idx=i,
        )
        _mk_task(vid, "caption", {"caption": _LONG_SEG_CAPTION}, segment_idx=i)


class _CountingMockClient(MockVLMClient):
    """MockVLMClient that counts calls per task, to prove merge invocation."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[str, int] = {}

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        # Extract SUBTASK= marker to record the task name.
        for m in messages:
            if m.role == "system" and isinstance(m.content, str):
                for line in m.content.splitlines():
                    if line.startswith("SUBTASK="):
                        name = line.split("=", 1)[1].strip()
                        self.calls[name] = self.calls.get(name, 0) + 1
                        break
        return await super().chat(messages, **kwargs)


def test_multi_segment_caption_stays_within_schema_limit(tmp_pipeline):
    """13 segments (like VID_895826). Naive join would exceed 700 chars.

    With a merge factory the merged caption fits AND the merge call is
    invoked exactly once.
    """
    vid = "VID_000700"
    _seed_multi_segment_video(vid, num_segments=13)

    # Naive join length would exceed the schema max — sanity check.
    naive = "; ".join([_LONG_SEG_CAPTION] * 13)
    assert len(naive) > 700

    counter = _CountingMockClient()
    payload = assemble_video(vid, merge_client_factory=lambda: counter)
    FinalAnnotation.model_validate(payload)  # schema-valid
    assert len(payload["caption"]) <= 700
    assert payload["caption"], "caption must not be empty"
    # Merge call happened exactly once for this multi-segment video.
    assert counter.calls.get("caption_merge", 0) == 1


def test_multi_segment_without_client_falls_back_and_clamps(tmp_pipeline):
    """No factory -> length-safe join + safety clamp; no crash."""
    vid = "VID_000701"
    _seed_multi_segment_video(vid, num_segments=13)

    payload = assemble_video(vid, merge_client_factory=None)
    FinalAnnotation.model_validate(payload)
    # The clamp keeps caption within the schema max.
    assert len(payload["caption"]) <= 700
    # And produced a non-trivial caption (not the fallback placeholder).
    assert payload["caption"] != "First-person navigation view."


def test_merged_caption_is_cached_and_not_re_called(tmp_pipeline):
    """Second assemble on the same video does not re-invoke the merge call."""
    vid = "VID_000702"
    _seed_multi_segment_video(vid, num_segments=5)

    counter = _CountingMockClient()
    _ = assemble_video(vid, merge_client_factory=lambda: counter)
    calls_first = counter.calls.get("caption_merge", 0)
    _ = assemble_video(vid, merge_client_factory=lambda: counter)
    calls_second = counter.calls.get("caption_merge", 0)
    assert calls_first == 1
    assert calls_second == 1  # cached; no new call


def test_clean_truncate_prefers_sentence_boundary():
    from egoannot.assemble import _clean_truncate

    text = "One sentence. Two sentence. Three sentence."
    truncated, changed = _clean_truncate(text, max_len=20)
    assert changed
    # Must end at a sentence boundary that fits.
    assert truncated.endswith(".")
    assert len(truncated) <= 20


def test_risk_description_is_clamped(tmp_pipeline):
    """Overlong risk descriptions must be truncated, not fail validation."""
    vid = "VID_000703"
    _mk_video(vid, duration=30.0)
    _mk_task(vid, "scene", {
        "location_type": "outdoor",
        "scene_category": [],
        "lighting": "normal",
        "weather": "clear",
        "crowd_level": "low",
        "camera_motion": "walking",
        "scene_summary": "OK.",
    })
    long_desc = "Alert! " + ("Watch the path here. " * 40)  # >>300 chars
    _mk_task(vid, "judgment", {
        "walkability": "passable_with_caution",
        "actions": ["observe"],
        "risks": [{
            "risk_type": "obstacle_ahead",
            "severity": "medium",
            "start_sec": 0.0,
            "end_sec": 30.0,
            "description": long_desc,
            "related_entities": [],
        }],
        "confidence": 0.5,
    })
    payload = assemble_video(vid)
    assert len(payload["risk_labels"]) == 1
    assert len(payload["risk_labels"][0]["description"]) <= 300
