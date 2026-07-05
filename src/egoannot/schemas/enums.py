"""Controlled vocabularies as StrEnum.

All values are English tokens (see SPEC.md). Natural-language content
(labels, descriptions, captions, Q/A text) is Chinese but is not enumerated.

Every enum defines a ``fallback`` classmethod that coerces unknown or
loosely-cased values to the nearest sensible member, plus a boolean
``coerced`` flag returned alongside so callers can log the coercion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

_E = TypeVar("_E", bound="_CoerceMixin")


class _CoerceMixin(StrEnum):
    """Utility mixin providing a safe-default ``coerce`` classmethod."""

    @classmethod
    def _default(cls: type[_E]) -> _E:
        """Return the fallback member used when a value can't be coerced.

        Concrete enums override this to point at a semantically neutral
        member (typically ``unknown`` / ``other`` / ``low``).
        """
        raise NotImplementedError

    @classmethod
    def coerce(cls: type[_E], value: object) -> tuple[_E, bool]:
        """Coerce arbitrary input to an enum member.

        Returns ``(member, was_coerced)``. ``was_coerced`` is ``True`` when
        the input did not match a known value exactly.
        """
        if isinstance(value, cls):
            return value, False
        if not isinstance(value, str):
            return cls._default(), True
        needle = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == needle:
                return member, needle != value
        return cls._default(), True


class Category(_CoerceMixin):
    person = "person"
    vehicle = "vehicle"
    bicycle = "bicycle"
    animal = "animal"
    obstacle = "obstacle"
    structure = "structure"
    sign = "sign"
    traffic_light = "traffic_light"
    other = "other"

    @classmethod
    def _default(cls) -> Category:
        return cls.other


class Importance(_CoerceMixin):
    high = "high"
    medium = "medium"
    low = "low"

    @classmethod
    def _default(cls) -> Importance:
        return cls.low


class Position(_CoerceMixin):
    front = "front"
    front_left = "front_left"
    front_right = "front_right"
    left = "left"
    right = "right"
    rear = "rear"

    @classmethod
    def _default(cls) -> Position:
        return cls.front


class Distance(_CoerceMixin):
    near = "near"
    mid = "mid"
    far = "far"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> Distance:
        return cls.unknown


class Motion(_CoerceMixin):
    static = "static"
    approaching = "approaching"
    receding = "receding"
    crossing = "crossing"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> Motion:
        return cls.unknown


class Severity(_CoerceMixin):
    high = "high"
    medium = "medium"
    low = "low"

    @classmethod
    def _default(cls) -> Severity:
        return cls.low


class RiskType(_CoerceMixin):
    person_crossing = "person_crossing"
    vehicle_approaching = "vehicle_approaching"
    obstacle_ahead = "obstacle_ahead"
    path_blocked = "path_blocked"
    surface_change = "surface_change"
    overhead_obstacle = "overhead_obstacle"
    other = "other"

    @classmethod
    def _default(cls) -> RiskType:
        return cls.other


class QAType(_CoerceMixin):
    scene = "scene"
    entity = "entity"
    event = "event"
    motion = "motion"
    risk = "risk"
    walkability = "walkability"

    @classmethod
    def _default(cls) -> QAType:
        return cls.scene


class Walkability(_CoerceMixin):
    passable = "passable"
    passable_with_caution = "passable_with_caution"
    not_passable = "not_passable"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> Walkability:
        return cls.unknown


class Action(_CoerceMixin):
    slow_down = "slow_down"
    stop = "stop"
    keep_left = "keep_left"
    keep_right = "keep_right"
    observe = "observe"
    wait = "wait"
    proceed = "proceed"
    detour = "detour"

    @classmethod
    def _default(cls) -> Action:
        return cls.observe


class LocationType(_CoerceMixin):
    outdoor = "outdoor"
    indoor = "indoor"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> LocationType:
        return cls.unknown


class Lighting(_CoerceMixin):
    normal = "normal"
    dim = "dim"
    bright = "bright"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> Lighting:
        return cls.unknown


class Weather(_CoerceMixin):
    clear = "clear"
    rain = "rain"
    snow = "snow"
    overcast = "overcast"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> Weather:
        return cls.unknown


class CrowdLevel(_CoerceMixin):
    low = "low"
    medium = "medium"
    high = "high"

    @classmethod
    def _default(cls) -> CrowdLevel:
        return cls.low


class CameraMotion(_CoerceMixin):
    walking = "walking"
    standing = "standing"
    turning = "turning"
    mixed = "mixed"
    unknown = "unknown"

    @classmethod
    def _default(cls) -> CameraMotion:
        return cls.unknown


class AnswerType(_CoerceMixin):
    open = "open"
    boolean = "boolean"
    choice = "choice"

    @classmethod
    def _default(cls) -> AnswerType:
        return cls.open


class Split(_CoerceMixin):
    train = "train"
    val = "val"
    test = "test"

    @classmethod
    def _default(cls) -> Split:
        return cls.train


class SourceDataset(_CoerceMixin):
    jaad = "jaad"
    advio = "advio"
    scand = "scand"
    navware = "navware"
    egoblind = "egoblind"
    generic = "generic"

    @classmethod
    def _default(cls) -> SourceDataset:
        return cls.jaad


class VideoStatus(_CoerceMixin):
    pending = "pending"
    curated = "curated"
    frames_done = "frames_done"
    tasks_done = "tasks_done"
    assembled = "assembled"
    failed = "failed"

    @classmethod
    def _default(cls) -> VideoStatus:
        return cls.pending


class TaskName(_CoerceMixin):
    scene = "scene"
    entities = "entities"
    events = "events"
    judgment = "judgment"
    caption = "caption"
    qa = "qa"

    @classmethod
    def _default(cls) -> TaskName:
        return cls.scene


CORE_TASKS: tuple[TaskName, ...] = (
    TaskName.scene,
    TaskName.entities,
    TaskName.events,
    TaskName.caption,
    TaskName.qa,
)
"""Tasks that must not all fail; ``judgment`` is optional and excluded."""


__all__ = [
    "CORE_TASKS",
    "Action",
    "AnswerType",
    "CameraMotion",
    "Category",
    "CrowdLevel",
    "Distance",
    "Importance",
    "Lighting",
    "LocationType",
    "Motion",
    "Position",
    "QAType",
    "RiskType",
    "Severity",
    "SourceDataset",
    "Split",
    "TaskName",
    "VideoStatus",
    "Walkability",
    "Weather",
]
