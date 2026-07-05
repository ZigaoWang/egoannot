"""Pydantic schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from egoannot.schemas import (
    CaptionResponse,
    EntitiesResponse,
    EventsResponse,
    FinalAnnotation,
    JudgmentResponse,
    QAResponse,
    SceneResponse,
)
from egoannot.schemas.enums import Category, LocationType, Walkability

# ---------- valid fixtures -----------------------------------------------


def test_scene_valid():
    m = SceneResponse.model_validate(
        {
            "location_type": "outdoor",
            "scene_category": ["outdoor_sidewalk"],
            "lighting": "normal",
            "weather": "clear",
            "crowd_level": "low",
            "camera_motion": "walking",
            "scene_summary": "Walking along a sidewalk.",
        }
    )
    assert m.location_type is LocationType.outdoor
    assert m.scene_category == ["outdoor_sidewalk"]


def test_entity_time_order_autoswaps():
    payload = {
        "entities": [
            {
                "label": "pedestrian",
                "category": "person",
                "importance": "high",
                "position": "front",
                "distance": "near",
                "motion": "crossing",
                "first_seen_sec": 12.0,
                "last_seen_sec": 6.0,   # inverted
                "overhead": False,
            }
        ]
    }
    m = EntitiesResponse.model_validate(payload)
    ent = m.entities[0]
    assert ent.first_seen_sec == 6.0
    assert ent.last_seen_sec == 12.0


def test_event_end_before_start_autoclamps():
    m = EventsResponse.model_validate(
        {
            "events": [
                {
                    "description": "Something happens.",
                    "start_sec": 10.0,
                    "end_sec": 5.0,
                    "involves": [],
                    "caused_by": None,
                }
            ]
        }
    )
    assert m.events[0].end_sec >= m.events[0].start_sec


def test_judgment_valid():
    m = JudgmentResponse.model_validate(
        {
            "walkability": "passable",
            "actions": ["proceed"],
            "risks": [],
            "confidence": 0.9,
        }
    )
    assert m.walkability is Walkability.passable
    assert m.actions[0].value == "proceed"


def test_caption_valid():
    m = CaptionResponse.model_validate({"caption": "Walking ahead."})
    assert m.caption == "Walking ahead."


def test_qa_valid():
    m = QAResponse.model_validate(
        {
            "qa": [
                {
                    "qa_type": "walkability",
                    "question": "Is it passable?",
                    "answer": "Yes.",
                    "evidence_entities": [],
                    "evidence_start_sec": None,
                    "evidence_end_sec": None,
                    "answer_type": "boolean",
                }
            ]
        }
    )
    assert m.qa[0].qa_type.value == "walkability"


# ---------- invalid fixtures ---------------------------------------------


def test_scene_missing_required():
    with pytest.raises(ValidationError):
        SceneResponse.model_validate({"location_type": "outdoor"})


def test_scene_forbids_extra_key():
    with pytest.raises(ValidationError):
        SceneResponse.model_validate(
            {
                "location_type": "outdoor",
                "scene_category": [],
                "lighting": "normal",
                "weather": "clear",
                "crowd_level": "low",
                "camera_motion": "walking",
                "scene_summary": "OK.",
                "unknown_key": 1,
            }
        )


def test_entity_negative_time_rejected():
    with pytest.raises(ValidationError):
        EntitiesResponse.model_validate(
            {
                "entities": [
                    {
                        "label": "x",
                        "category": "other",
                        "importance": "low",
                        "position": "front",
                        "distance": "near",
                        "motion": "static",
                        "first_seen_sec": -1.0,
                        "last_seen_sec": 5.0,
                        "overhead": False,
                    }
                ]
            }
        )


# ---------- coercion ------------------------------------------------------


def test_enum_coerce_case_insensitive():
    m, coerced = Category.coerce("Person")
    assert m is Category.person
    assert coerced is True


def test_enum_coerce_unknown_defaults():
    m, coerced = Category.coerce("garbage-value")
    assert m is Category.other
    assert coerced is True


def test_enum_coerce_ok_pass_through():
    m, coerced = Category.coerce("vehicle")
    assert m is Category.vehicle
    assert coerced is False


def test_subtask_enum_field_silently_coerces():
    m = SceneResponse.model_validate(
        {
            "location_type": "Outdoor",   # case-mismatch
            "scene_category": [],
            "lighting": "normal",
            "weather": "clear",
            "crowd_level": "low",
            "camera_motion": "walking",
            "scene_summary": "OK.",
        }
    )
    assert m.location_type is LocationType.outdoor


# ---------- final annotation ---------------------------------------------


def _ref_dict() -> dict:
    return {
        "video_id": "VID_000001",
        "split": "train",
        "video_path": "videos/VID_000001.mp4",
        "frame_dir": "frames/VID_000001",
        "duration_sec": 30.0,
        "fps": 30.0,
        "candidate_fps": 0.4,
        "num_candidate_frames": 12,
        "sampling_interval_sec": 2.5,
        "resolution": [640, 480],
        "scene_category": ["outdoor_sidewalk"],
        "environment": {
            "location_type": "outdoor",
            "lighting": "normal",
            "weather": "clear",
            "crowd_level": "low",
            "camera_motion": "walking",
        },
        "caption": "Walking along a sidewalk.",
        "key_elements": [
            {
                "label": "pedestrian",
                "category": "person",
                "importance": "high",
                "time_span": [8.0, 14.0],
                "position": "front",
            }
        ],
        "risk_labels": [],
        "walkability": "passable",
        "acceptable_actions": ["proceed"],
        "qa_pairs": [
            {
                "qid": "VID_000001_Q001",
                "type": "scene",
                "question": "Where is this?",
                "answer": "A sidewalk.",
                "evidence_elements": [],
                "evidence_time_span": None,
                "answer_type": "open",
            }
        ],
        "privacy": {
            "face_blurred": False,
            "plate_blurred": False,
            "contains_sensitive_info": False,
        },
    }


def test_final_annotation_valid():
    m = FinalAnnotation.model_validate(_ref_dict())
    assert m.video_id == "VID_000001"
    assert m.key_elements[0].label == "pedestrian"


def test_final_annotation_bad_video_id():
    bad = _ref_dict()
    bad["video_id"] = "not-valid"
    with pytest.raises(ValidationError):
        FinalAnnotation.model_validate(bad)


def test_final_annotation_bad_qid():
    bad = _ref_dict()
    bad["qa_pairs"][0]["qid"] = "Q001"
    with pytest.raises(ValidationError):
        FinalAnnotation.model_validate(bad)


def test_final_annotation_bad_resolution_shape():
    bad = _ref_dict()
    bad["resolution"] = [640]
    with pytest.raises(ValidationError):
        FinalAnnotation.model_validate(bad)
