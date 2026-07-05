"""Deterministic mock client for offline runs and tests.

Returns canned valid JSON per task name. Frame count / segment context is
ignored — the mock never inspects images. Used via ``--mock`` on the CLI
and by ``tests/test_orchestrator_mock.py`` for the end-to-end run.
"""

from __future__ import annotations

import json
from typing import Any

from .client import ChatMessage, VLMResponse

_CANNED: dict[str, dict[str, Any]] = {
    "scene": {
        "location_type": "outdoor",
        "scene_category": ["outdoor_sidewalk", "dynamic_obstacle"],
        "lighting": "normal",
        "weather": "clear",
        "crowd_level": "low",
        "camera_motion": "walking",
        "scene_summary": "First-person view walking along a sidewalk with a pedestrian crossing ahead.",
    },
    "entities": {
        "entities": [
            {
                "label": "pedestrian",
                "category": "person",
                "importance": "high",
                "position": "front",
                "distance": "near",
                "motion": "crossing",
                "first_seen_sec": 3.0,
                "last_seen_sec": 8.5,
                "overhead": False,
            },
            {
                "label": "parked_car",
                "category": "vehicle",
                "importance": "medium",
                "position": "right",
                "distance": "mid",
                "motion": "static",
                "first_seen_sec": 1.0,
                "last_seen_sec": 12.0,
                "overhead": False,
            },
        ]
    },
    "events": {
        "events": [
            {
                "description": "A pedestrian crosses from the right in front of the walker.",
                "start_sec": 3.0,
                "end_sec": 8.5,
                "involves": ["pedestrian"],
                "caused_by": None,
            }
        ]
    },
    "judgment": {
        "walkability": "passable_with_caution",
        "actions": ["slow_down", "observe"],
        "risks": [
            {
                "risk_type": "person_crossing",
                "severity": "medium",
                "start_sec": 3.0,
                "end_sec": 8.5,
                "description": "A pedestrian is crossing in front; slow down and observe.",
                "related_entities": ["pedestrian"],
            }
        ],
        "confidence": 0.75,
    },
    "caption": {
        "caption": (
            "I am walking along a sidewalk in first-person view. A pedestrian is "
            "crossing ahead from the right; a parked car sits on the right shoulder. "
            "The path is mostly passable but I should slow down."
        )
    },
    "qa": {
        "qa": [
            {
                "qa_type": "scene",
                "question": "Is the current scene indoor or outdoor?",
                "answer": "Outdoor: a sidewalk scene.",
                "evidence_entities": [],
                "evidence_start_sec": None,
                "evidence_end_sec": None,
                "answer_type": "choice",
            },
            {
                "qa_type": "entity",
                "question": "Is there a pedestrian in front of the walker?",
                "answer": "Yes, one pedestrian is crossing from the right.",
                "evidence_entities": ["pedestrian"],
                "evidence_start_sec": 3.0,
                "evidence_end_sec": 8.5,
                "answer_type": "boolean",
            },
            {
                "qa_type": "risk",
                "question": "Are there any dynamic obstacles ahead?",
                "answer": "Yes, a pedestrian is crossing ahead; medium risk.",
                "evidence_entities": ["pedestrian"],
                "evidence_start_sec": 3.0,
                "evidence_end_sec": 8.5,
                "answer_type": "open",
            },
            {
                "qa_type": "walkability",
                "question": "Can the walker proceed at normal speed?",
                "answer": "No; slow down and observe the crossing pedestrian.",
                "evidence_entities": ["pedestrian"],
                "evidence_start_sec": 3.0,
                "evidence_end_sec": 8.5,
                "answer_type": "open",
            },
            {
                "qa_type": "motion",
                "question": "In which direction is the pedestrian moving?",
                "answer": "From right to left across the walker's path.",
                "evidence_entities": ["pedestrian"],
                "evidence_start_sec": 3.0,
                "evidence_end_sec": 8.5,
                "answer_type": "choice",
            },
        ]
    },
    "curate": {
        "keep": True,
        "viewpoint": "eye_level_walking",
        "reason": "Forward-facing eye-level walking viewpoint with visible activity.",
    },
}


class MockVLMClient:
    """Drop-in replacement for :class:`VLMClient` for offline runs."""

    def __init__(self) -> None:
        self._closed = False

    async def aclose(self) -> None:
        self._closed = True

    async def __aenter__(self) -> MockVLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format_json: bool = True,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        # The orchestrator embeds the task name in the system prompt via
        # ``vlm.prompts.build_messages``; the mock looks it up there.
        task = _extract_task_name(messages)
        payload = _CANNED.get(task, {"note": "unknown task in mock", "task": task})
        text = json.dumps(payload, ensure_ascii=False)
        return VLMResponse(
            text=text,
            prompt_tokens=0,
            completion_tokens=len(text),
            total_tokens=len(text),
            raw={"mock": True, "task": task},
        )


def _extract_task_name(messages: list[ChatMessage]) -> str:
    for m in messages:
        if m.role == "system" and isinstance(m.content, str):
            for line in m.content.splitlines():
                if line.startswith("SUBTASK="):
                    return line.split("=", 1)[1].strip()
    return "unknown"


__all__ = ["MockVLMClient"]
