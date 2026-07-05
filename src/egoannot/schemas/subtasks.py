"""Pydantic v2 response models for each sub-task.

The model is instructed to return one JSON object with these exact keys.
Every response is validated here; on validation failure the orchestrator
retries once with a corrective instruction, then falls back to a safe
default and persists ``ok=False`` on the TaskResult row.

Enum fields use ``_CoerceValidator`` so that a mis-cased or slightly-off
token becomes the nearest valid member. Every coercion event emits a
structlog line with the ambient ``video_id`` / ``task_name`` context so
prompt weaknesses show up in the logs. Numeric time fields are clamped
by downstream logic; here we only enforce non-negativity and rough sanity.

All natural-language content is in English; enum values also English.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from pydantic.functional_validators import BeforeValidator

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
    Walkability,
    Weather,
    _CoerceMixin,
)

_E = TypeVar("_E", bound=_CoerceMixin)
_log = structlog.stdlib.get_logger(__name__)


def _coerce(enum_cls: type[_E]) -> BeforeValidator:
    """BeforeValidator that coerces to ``enum_cls`` and logs any coercion.

    The ambient structlog context (``video_id``, ``task_name`` bound by the
    orchestrator) is picked up automatically so log lines are actionable.
    """

    def _v(value: object, info: ValidationInfo) -> _E:
        member, coerced = enum_cls.coerce(value)
        if coerced:
            _log.info(
                "enum_coerced",
                enum=enum_cls.__name__,
                field=info.field_name,
                raw=value,
                coerced_to=member.value,
            )
        return member

    return BeforeValidator(_v)


CategoryF = Annotated[Category, _coerce(Category)]
ImportanceF = Annotated[Importance, _coerce(Importance)]
PositionF = Annotated[Position, _coerce(Position)]
DistanceF = Annotated[Distance, _coerce(Distance)]
MotionF = Annotated[Motion, _coerce(Motion)]
SeverityF = Annotated[Severity, _coerce(Severity)]
RiskTypeF = Annotated[RiskType, _coerce(RiskType)]
QATypeF = Annotated[QAType, _coerce(QAType)]
WalkabilityF = Annotated[Walkability, _coerce(Walkability)]
ActionF = Annotated[Action, _coerce(Action)]
LocationTypeF = Annotated[LocationType, _coerce(LocationType)]
LightingF = Annotated[Lighting, _coerce(Lighting)]
WeatherF = Annotated[Weather, _coerce(Weather)]
CrowdLevelF = Annotated[CrowdLevel, _coerce(CrowdLevel)]
CameraMotionF = Annotated[CameraMotion, _coerce(CameraMotion)]
AnswerTypeF = Annotated[AnswerType, _coerce(AnswerType)]


class _StrictModel(BaseModel):
    """Reject unknown keys; keep JSON-schema clean; frozen for hashability."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)


# --------------------------------------------------------------------------- scene
class SceneResponse(_StrictModel):
    """Layers 1 (scene/localization) and 5 (ego-motion)."""

    location_type: LocationTypeF
    scene_category: list[str] = Field(
        default_factory=list,
        description="1-3 short English scene tags, e.g. outdoor_sidewalk, dynamic_obstacle.",
        max_length=6,
    )
    lighting: LightingF
    weather: WeatherF
    crowd_level: CrowdLevelF
    camera_motion: CameraMotionF
    scene_summary: str = Field(min_length=1, max_length=400)


# --------------------------------------------------------------------------- entities
class EntityItem(_StrictModel):
    """One key element in the scene (layers 2 & 4)."""

    label: str = Field(min_length=1, max_length=60)
    category: CategoryF
    importance: ImportanceF
    position: PositionF
    distance: DistanceF
    motion: MotionF
    first_seen_sec: float = Field(ge=0.0)
    last_seen_sec: float = Field(ge=0.0)
    overhead: bool = False

    @model_validator(mode="after")
    def _time_order(self) -> EntityItem:
        if self.last_seen_sec < self.first_seen_sec:
            lo, hi = self.last_seen_sec, self.first_seen_sec
            object.__setattr__(self, "first_seen_sec", lo)
            object.__setattr__(self, "last_seen_sec", hi)
        return self


class EntitiesResponse(_StrictModel):
    entities: list[EntityItem] = Field(default_factory=list, max_length=16)


# --------------------------------------------------------------------------- events
class EventItem(_StrictModel):
    """One temporally-grounded happening (layer 3)."""

    description: str = Field(min_length=1, max_length=300)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    involves: list[str] = Field(default_factory=list, max_length=8)
    caused_by: str | None = None

    @model_validator(mode="after")
    def _time_order(self) -> EventItem:
        if self.end_sec < self.start_sec:
            object.__setattr__(self, "end_sec", self.start_sec)
        return self


class EventsResponse(_StrictModel):
    events: list[EventItem] = Field(default_factory=list, max_length=12)


# --------------------------------------------------------------------------- judgment
class RiskItem(_StrictModel):
    """One risk observation (layer 6)."""

    risk_type: RiskTypeF
    severity: SeverityF
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    description: str = Field(min_length=1, max_length=300)
    related_entities: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _time_order(self) -> RiskItem:
        if self.end_sec < self.start_sec:
            object.__setattr__(self, "end_sec", self.start_sec)
        return self


class JudgmentResponse(_StrictModel):
    walkability: WalkabilityF
    actions: list[ActionF] = Field(default_factory=list, max_length=8)
    risks: list[RiskItem] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- caption
class CaptionResponse(_StrictModel):
    """Text-only synthesis: 2-3 first-person English sentences."""

    caption: str = Field(min_length=1, max_length=700)


# --------------------------------------------------------------------------- qa
class QAItem(_StrictModel):
    qa_type: QATypeF
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=600)
    evidence_entities: list[str] = Field(default_factory=list, max_length=8)
    evidence_start_sec: float | None = Field(default=None, ge=0.0)
    evidence_end_sec: float | None = Field(default=None, ge=0.0)
    answer_type: AnswerTypeF


class QAResponse(_StrictModel):
    qa: list[QAItem] = Field(default_factory=list, max_length=16)


# --------------------------------------------------------------------------- curate
class CurateClassification(_StrictModel):
    """Curate-stage cheap keep/drop classification."""

    keep: bool
    viewpoint: str = Field(default="unknown", max_length=40)
    reason: str = Field(default="", max_length=300)


SUBTASK_MODELS: dict[str, type[_StrictModel]] = {
    "scene": SceneResponse,
    "entities": EntitiesResponse,
    "events": EventsResponse,
    "judgment": JudgmentResponse,
    "caption": CaptionResponse,
    "qa": QAResponse,
}


def get_subtask_model(name: str) -> type[_StrictModel]:
    """Look up the Pydantic response model for a sub-task name."""
    try:
        return SUBTASK_MODELS[name]
    except KeyError as exc:
        raise KeyError(f"unknown sub-task: {name!r}") from exc


__all__ = [
    "SUBTASK_MODELS",
    "CaptionResponse",
    "CurateClassification",
    "EntitiesResponse",
    "EntityItem",
    "EventItem",
    "EventsResponse",
    "JudgmentResponse",
    "QAItem",
    "QAResponse",
    "RiskItem",
    "SceneResponse",
    "get_subtask_model",
]


# Silence unused-import lint when consumers only pull the module.
_ = Any
