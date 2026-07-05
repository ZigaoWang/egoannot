"""Smoke test for schemas. Run: python scripts/smoke_schemas.py"""

from __future__ import annotations

from egoannot.schemas import (
    CaptionResponse,
    EntitiesResponse,
    EventsResponse,
    FinalAnnotation,
    JudgmentResponse,
    QAResponse,
    SceneResponse,
)
from egoannot.schemas.enums import Category


def main() -> None:
    scene = SceneResponse.model_validate(
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
    print("scene ok:", scene.location_type.value)

    ents = EntitiesResponse.model_validate(
        {
            "entities": [
                {
                    "label": "pedestrian",
                    "category": "person",
                    "importance": "high",
                    "position": "front",
                    "distance": "near",
                    "motion": "crossing",
                    "first_seen_sec": 8.0,
                    "last_seen_sec": 14.5,
                    "overhead": False,
                }
            ]
        }
    )
    print("entities ok:", ents.entities[0].label)

    ev = EventsResponse.model_validate(
        {
            "events": [
                {
                    "description": "A pedestrian crosses from right to left.",
                    "start_sec": 9.0,
                    "end_sec": 12.0,
                    "involves": ["pedestrian"],
                    "caused_by": None,
                }
            ]
        }
    )
    print("events ok:", ev.events[0].start_sec)

    judg = JudgmentResponse.model_validate(
        {
            "walkability": "passable_with_caution",
            "actions": ["slow_down", "observe"],
            "risks": [
                {
                    "risk_type": "person_crossing",
                    "severity": "medium",
                    "start_sec": 8.0,
                    "end_sec": 14.5,
                    "description": "Pedestrian crossing ahead.",
                    "related_entities": ["pedestrian"],
                }
            ],
            "confidence": 0.7,
        }
    )
    print("judgment ok:", judg.walkability.value, len(judg.risks))

    cap = CaptionResponse.model_validate({"caption": "Walking along a sidewalk."})
    print("caption ok:", len(cap.caption))

    qa = QAResponse.model_validate(
        {
            "qa": [
                {
                    "qa_type": "risk",
                    "question": "Are there dynamic obstacles ahead?",
                    "answer": "Yes, a pedestrian is crossing.",
                    "evidence_entities": ["pedestrian"],
                    "evidence_start_sec": 8.0,
                    "evidence_end_sec": 14.5,
                    "answer_type": "open",
                }
            ]
        }
    )
    print("qa ok:", qa.qa[0].qa_type.value)

    m, coerced = Category.coerce("Person")
    print("coerce Person ->", m.value, "coerced=", coerced)
    m, coerced = Category.coerce("garbage-value")
    print("coerce garbage ->", m.value, "coerced=", coerced)

    ref = {
        "video_id": "VID_000001",
        "split": "train",
        "video_path": "videos/VID_000001.mp4",
        "frame_dir": "frames/VID_000001",
        "duration_sec": 32.4,
        "fps": 30,
        "candidate_fps": 2.0,
        "num_candidate_frames": 65,
        "sampling_interval_sec": 0.5,
        "resolution": [1920, 1080],
        "scene_category": ["outdoor_sidewalk", "dynamic_obstacle"],
        "environment": {
            "location_type": "outdoor",
            "lighting": "normal",
            "weather": "clear",
            "crowd_level": "low",
            "camera_motion": "walking",
        },
        "caption": (
            "I am walking along a sidewalk in first-person view. A pedestrian is "
            "crossing ahead; a parked car is on the right shoulder. The path is "
            "mostly passable but I should slow down."
        ),
        "key_elements": [
            {
                "label": "pedestrian",
                "category": "person",
                "importance": "high",
                "time_span": [8.0, 14.5],
                "position": "front",
            },
            {
                "label": "parked_car",
                "category": "vehicle",
                "importance": "medium",
                "time_span": [5.0, 20.0],
                "position": "right",
            },
        ],
        "risk_labels": [
            {
                "type": "person_crossing",
                "severity": "medium",
                "time_span": [8.0, 14.5],
                "description": "A pedestrian is crossing in front; slow down and observe.",
            }
        ],
        "walkability": "passable_with_caution",
        "acceptable_actions": ["slow_down", "keep_left", "observe"],
        "qa_pairs": [
            {
                "qid": "VID_000001_Q001",
                "type": "risk",
                "question": "Are there any dynamic obstacles ahead?",
                "answer": "Yes, a pedestrian is crossing ahead.",
                "evidence_elements": ["pedestrian"],
                "evidence_time_span": [8.0, 14.5],
                "answer_type": "open",
            },
            {
                "qid": "VID_000001_Q002",
                "type": "walkability",
                "question": "Can the walker proceed at normal speed?",
                "answer": "No; slow down and observe the crossing pedestrian.",
                "evidence_elements": ["pedestrian"],
                "evidence_time_span": [8.0, 14.5],
                "answer_type": "open",
            },
        ],
        "privacy": {
            "face_blurred": True,
            "plate_blurred": True,
            "contains_sensitive_info": False,
        },
    }
    final = FinalAnnotation.model_validate(ref)
    print(
        "final ok:",
        final.video_id,
        final.walkability.value,
        "n_qa=",
        len(final.qa_pairs),
    )


if __name__ == "__main__":
    main()
