"""Prompt templates.

Every prompt instructs the model, in English, to return one JSON object with
no prose and no markdown fences. All natural-language values and enum tokens
are English.

The first line of every system prompt carries ``SUBTASK=<name>`` so the
mock client can dispatch canned responses without inspecting images.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..schemas.enums import (
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
    Walkability,
    Weather,
)
from .client import ChatMessage


def _tokens(enum_cls: type[Enum]) -> str:
    return ", ".join(m.value for m in enum_cls)


_JSON_RULES = (
    "Return exactly ONE JSON object. Do not wrap it in markdown fences. "
    "Do not add prose before or after the JSON. All natural-language values "
    "(labels, descriptions, questions, answers, summaries) MUST be written in "
    "English. All categorical values MUST be one of the exact English tokens "
    "listed for that field."
)


# --------------------------------------------------------------------------- scene
_SCENE_INSTRUCTIONS = f"""SUBTASK=scene
You are analyzing an egocentric (first-person, eye-level) navigation video segment.
The user will send several frames labeled with [t=Xs] markers.

Task: classify the scene and ego-motion. Output JSON with these keys ONLY:
  - location_type: one of [{_tokens(LocationType)}]
  - scene_category: 1-3 short English tags describing the scene type
    (snake_case, e.g. outdoor_sidewalk, indoor_hallway, dynamic_obstacle)
  - lighting: one of [{_tokens(Lighting)}]
  - weather: one of [{_tokens(Weather)}]
  - crowd_level: one of [{_tokens(CrowdLevel)}]
  - camera_motion: one of [{_tokens(CameraMotion)}]
  - scene_summary: one English sentence (<=200 chars) summarising the scene

{_JSON_RULES}
"""


# --------------------------------------------------------------------------- entities
_ENTITIES_INSTRUCTIONS = f"""SUBTASK=entities
You are analyzing an egocentric navigation video segment via sampled frames.

Task: list the most navigation-relevant entities visible. Output JSON:
{{
  "entities": [
    {{
      "label": "<short English label, e.g. pedestrian, parked_car, curb>",
      "category": "one of [{_tokens(Category)}]",
      "importance": "one of [{_tokens(Importance)}]",
      "position": "one of [{_tokens(Position)}]",
      "distance": "one of [{_tokens(Distance)}]",
      "motion": "one of [{_tokens(Motion)}]",
      "first_seen_sec": <float seconds from segment start>,
      "last_seen_sec": <float seconds from segment start>,
      "overhead": <true if the object is at chest/head height and might be missed by a cane>
    }}
  ]
}}

Use the [t=Xs] markers to fill first_seen_sec / last_seen_sec. Report at most
{{max_entities}} entities. Prioritise items that affect the walker's decision
(pedestrians, moving vehicles, obstacles). {_JSON_RULES}
"""


# --------------------------------------------------------------------------- events
_EVENTS_INSTRUCTIONS = f"""SUBTASK=events
You are analyzing an egocentric navigation video segment via sampled frames.

Task: list the meaningful events in the segment. An event is something that
HAPPENS over time (a pedestrian crosses, the walker turns, a car passes).
Static facts belong in entities, not events. Output JSON:
{{
  "events": [
    {{
      "description": "<English sentence>",
      "start_sec": <float>,
      "end_sec": <float>,
      "involves": ["<entity label>", ...],
      "caused_by": <null or a short English cause description>
    }}
  ]
}}

Report at most {{max_events}} events; if nothing meaningful happens, return
{{ "events": [] }}. {_JSON_RULES}
"""


# --------------------------------------------------------------------------- judgment
_JUDGMENT_INSTRUCTIONS = f"""SUBTASK=judgment
You are analyzing an egocentric navigation video segment via sampled frames.

Task: judge walkability and enumerate risks. Output JSON:
{{
  "walkability": "one of [{_tokens(Walkability)}]",
  "actions": ["<one or more of {_tokens(Action)}>"],
  "risks": [
    {{
      "risk_type": "one of [{_tokens(RiskType)}]",
      "severity": "one of [{_tokens(Severity)}]",
      "start_sec": <float>, "end_sec": <float>,
      "description": "<English sentence>",
      "related_entities": ["<entity label>", ...]
    }}
  ],
  "confidence": <float 0..1>
}}

If no risks are present, return risks=[] and pick a walkability accordingly.
{_JSON_RULES}
"""


# --------------------------------------------------------------------------- caption
_CAPTION_INSTRUCTIONS = f"""SUBTASK=caption
You will be given the outputs of earlier sub-tasks (scene, entities, events,
judgment) as a compact JSON block. Do NOT ask for images; synthesise from
that text alone.

Task: produce a 2-3 sentence first-person English caption describing what
the walker sees and does. Output JSON:
{{
  "caption": "<2-3 English sentences>"
}}

{_JSON_RULES}
"""


# --------------------------------------------------------------------------- qa
_QA_INSTRUCTIONS = f"""SUBTASK=qa
You will be given the outputs of earlier sub-tasks (scene, entities, events,
judgment) as a compact JSON block. Do NOT ask for images; synthesise from
that text alone.

Task: produce exactly {{num_qa}} question-answer pairs spanning the six
conceptual layers (scene, entity, event, motion, risk, walkability).
Cover as many distinct qa_type values as possible. Output JSON:
{{
  "qa": [
    {{
      "qa_type": "one of [{_tokens(QAType)}]",
      "question": "<English question>",
      "answer": "<English answer>",
      "evidence_entities": ["<entity label>", ...],
      "evidence_start_sec": <float or null>,
      "evidence_end_sec": <float or null>,
      "answer_type": "one of [{_tokens(AnswerType)}]"
    }}
  ]
}}

{_JSON_RULES}
"""


# --------------------------------------------------------------------------- caption_merge
_CAPTION_MERGE_INSTRUCTIONS = """SUBTASK=caption_merge
You will be given the per-segment captions of a multi-segment egocentric
walking video, plus (optionally) per-segment scene summaries. These are
already validated JSON produced by earlier sub-tasks. Do NOT ask for
images.

Task: fold the per-segment captions into ONE coherent first-person
English caption for the WHOLE video. Preserve the walker's chronology
where the captions imply it (segment_idx is monotonically increasing).
Do not enumerate segments; write natural prose. Keep the output
strictly shorter than the ``target_max_chars`` value provided in the
input context — aim for 2-4 sentences.

Output JSON:
{
  "caption": "<2-4 English sentences summarising the whole walk>"
}

Return exactly ONE JSON object. No prose, no markdown fences.
"""


_CURATE_INSTRUCTIONS = """SUBTASK=curate
You are inspecting a few sample frames from a candidate video for a dataset
of egocentric (first-person, eye-level) walking/navigation clips.

Task: decide whether this clip is a forward-facing, roughly eye-level
walking/navigation viewpoint with visible activity. Output JSON:
{
  "keep": <true|false>,
  "viewpoint": "<short English descriptor, e.g. eye_level_walking, dashcam, aerial>",
  "reason": "<one English sentence explaining the decision>"
}

Return exactly ONE JSON object. No prose, no markdown fences.
"""


_TASK_TEMPLATES: dict[str, str] = {
    "scene": _SCENE_INSTRUCTIONS,
    "entities": _ENTITIES_INSTRUCTIONS,
    "events": _EVENTS_INSTRUCTIONS,
    "judgment": _JUDGMENT_INSTRUCTIONS,
    "caption": _CAPTION_INSTRUCTIONS,
    "qa": _QA_INSTRUCTIONS,
    "curate": _CURATE_INSTRUCTIONS,
    "caption_merge": _CAPTION_MERGE_INSTRUCTIONS,
}


def build_messages(
    task: str,
    *,
    image_content: list[dict[str, Any]] | None = None,
    context_json: str | None = None,
    template_params: dict[str, Any] | None = None,
) -> list[ChatMessage]:
    """Build the chat messages for one sub-task call.

    Args:
        task: sub-task name (scene, entities, events, judgment, caption, qa,
            curate).
        image_content: OpenAI-style content blocks with [t=Xs] markers and
            base64 image_url blocks. Required for image-bearing tasks;
            forbidden for the text-only synthesis tasks (caption, qa).
        context_json: compact JSON string of prior sub-task outputs for the
            text-only synthesis tasks.
        template_params: fills placeholders like ``{max_entities}`` in the
            system template via explicit string replacement (safe against
            JSON braces in the template body).

    Returns:
        [ChatMessage(system), ChatMessage(user)] ready to hand to the client.
    """
    if task not in _TASK_TEMPLATES:
        raise KeyError(f"unknown task template: {task!r}")

    template = _TASK_TEMPLATES[task]
    if template_params:
        # Use explicit substitution instead of str.format() because every
        # template contains JSON braces that .format() would misparse.
        for key, val in template_params.items():
            template = template.replace("{" + key + "}", str(val))
    system_msg = ChatMessage(role="system", content=template)

    if task in {"caption", "qa", "caption_merge"}:
        if not context_json:
            raise ValueError(f"task {task!r} requires context_json (prior sub-task outputs)")
        user_content = (
            "PRIOR_RESULTS_JSON:\n" + context_json + "\n\nProduce the JSON as instructed."
        )
        return [system_msg, ChatMessage(role="user", content=user_content)]

    if image_content is None:
        raise ValueError(f"task {task!r} requires image_content")
    return [system_msg, ChatMessage(role="user", content=image_content)]


__all__ = ["build_messages"]
