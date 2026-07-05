"""Assembly tests: canned TaskResult rows -> schema-valid final JSON."""

from __future__ import annotations

import json

import pytest

from egoannot.assemble import assemble_video
from egoannot.db import Annotation, Segment, TaskResult, Video, session_scope
from egoannot.schemas.annotation import FinalAnnotation


def _mk_video(vid: str, *, duration: float = 30.0, dataset: str = "jaad") -> None:
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
                num_candidate_frames=12,
                candidate_fps=12 / duration,
                status="tasks_done",
            )
        )
        s.add(Segment(video_id=vid, idx=0, start_sec=0.0, end_sec=duration))


def _mk_task(vid: str, task: str, payload: dict) -> None:
    with session_scope() as s:
        s.add(
            TaskResult(
                video_id=vid,
                segment_idx=0,
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
