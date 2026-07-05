"""Final annotation schema.

Owned entirely by Python (never emitted by the model). The assembly stage
constructs a dict of this shape from validated sub-task rows plus values
sourced from the Video row and static config; the dict is then validated
against :class:`FinalAnnotation` before writing to disk.

Field naming and shape mirror the reference JSON at the bottom of SPEC.md.
Extra justified fields on nested items are allowed (``extra="ignore"``);
missing required fields are not.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
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

TimeSpan = Annotated[
    list[float],
    Field(
        min_length=2,
        max_length=2,
        description="[start_sec, end_sec] with end >= start >= 0.",
    ),
]


class _Model(BaseModel):
    # Ignore rather than forbid on nested items: allows the pipeline to
    # add justified fields later without a schema break.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class Environment(_Model):
    location_type: LocationType
    lighting: Lighting
    weather: Weather
    crowd_level: CrowdLevel
    camera_motion: CameraMotion


class KeyElement(_Model):
    """One element in the final annotation's ``key_elements`` list."""

    label: str = Field(min_length=1, max_length=60)
    category: Category
    importance: Importance
    time_span: TimeSpan
    position: Position
    # Optional additions retained from the entities sub-task for downstream use.
    distance: Distance = Distance.unknown
    motion: Motion = Motion.unknown
    overhead: bool = False


class RiskLabel(_Model):
    type: RiskType
    severity: Severity
    time_span: TimeSpan
    description: str = Field(min_length=1, max_length=300)


class QAPair(_Model):
    qid: str = Field(pattern=r"^VID_\d{6}_Q\d{3}$")
    type: QAType
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=600)
    evidence_elements: list[str] = Field(default_factory=list, max_length=8)
    evidence_time_span: TimeSpan | None = None
    answer_type: AnswerType


class Privacy(_Model):
    face_blurred: bool
    plate_blurred: bool
    contains_sensitive_info: bool


class FinalAnnotation(_Model):
    """The full per-video annotation. This is what gets written to disk."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    video_id: str = Field(pattern=r"^VID_\d{6}$")
    split: Split
    video_path: str
    frame_dir: str
    duration_sec: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    candidate_fps: float = Field(ge=0.0)
    num_candidate_frames: int = Field(ge=0)
    # Effective time resolution of any time_span in this annotation: the
    # spacing between adjacent sampled frames sent to the model. Downstream
    # consumers should treat any [start,end] as reliable only within
    # +/- sampling_interval_sec.
    sampling_interval_sec: float = Field(ge=0.0, default=0.0)
    resolution: Annotated[list[int], Field(min_length=2, max_length=2)]

    scene_category: list[str] = Field(default_factory=list, max_length=6)
    environment: Environment
    caption: str = Field(min_length=1, max_length=700)

    key_elements: list[KeyElement] = Field(default_factory=list, max_length=16)
    risk_labels: list[RiskLabel] = Field(default_factory=list, max_length=8)

    walkability: Walkability
    acceptable_actions: list[Action] = Field(default_factory=list, max_length=8)

    qa_pairs: list[QAPair] = Field(default_factory=list, max_length=16)
    privacy: Privacy


__all__ = [
    "Environment",
    "FinalAnnotation",
    "KeyElement",
    "Privacy",
    "QAPair",
    "RiskLabel",
    "TimeSpan",
]
