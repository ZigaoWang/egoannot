"""SQLAlchemy 2.0 declarative models.

The database is a disposable, regenerable batch artifact — no migration
framework, no versioning. Schema is created via ``Base.metadata.create_all``
at engine bootstrap.

Every stage keys off the ``Video.status`` column so re-runs skip rows that
have already advanced past their target stage. Sub-task results are stored
in ``TaskResult`` with a unique (video_id, segment_idx, task_name) key,
which makes ``annotate`` naturally idempotent.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    """Common declarative base."""


class Video(Base):
    """One raw or curated clip.

    ``status`` transitions:
        pending -> curated -> frames_done -> tasks_done -> assembled
        (or -> failed at any stage that raises fatally)
    """

    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. VID_000001
    source_dataset: Mapped[str] = mapped_column(String(16), index=True)
    source_path: Mapped[str] = mapped_column(Text)

    selected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    select_reason: Mapped[str] = mapped_column(Text, default="")
    split: Mapped[str] = mapped_column(String(8), default="train")

    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    resolution_w: Mapped[int] = mapped_column(Integer, default=0)
    resolution_h: Mapped[int] = mapped_column(Integer, default=0)

    frame_dir: Mapped[str] = mapped_column(Text, default="")
    # Effective sampling rate = num_candidate_frames / duration_sec, computed
    # once frame extraction + segmentation is done. Float, not int, because
    # short clips with 12 frames / 20s yield rates like 0.6 fps.
    candidate_fps: Mapped[float] = mapped_column(Float, default=0.0)
    # Total frames actually forwarded to the model across all segments
    # (per_segment * num_segments). NOT the raw 10-fps extraction count.
    num_candidate_frames: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )

    segments: Mapped[list[Segment]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        order_by="Segment.idx",
    )
    task_results: Mapped[list[TaskResult]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
    )
    annotation: Mapped[Annotation | None] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (Index("ix_videos_status_selected", "status", "selected"),)


class Segment(Base):
    """One time window within a video that gets its own sub-task set."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)

    video: Mapped[Video] = relationship(back_populates="segments")

    __table_args__ = (
        UniqueConstraint("video_id", "idx", name="uq_segments_video_idx"),
    )


class TaskResult(Base):
    """One validated sub-task response for one (video, segment)."""

    __tablename__ = "task_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    segment_idx: Mapped[int] = mapped_column(Integer)
    task_name: Mapped[str] = mapped_column(String(16))

    raw_response: Mapped[str] = mapped_column(Text, default="")
    parsed_json: Mapped[str] = mapped_column(Text, default="")
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    video: Mapped[Video] = relationship(back_populates="task_results")

    __table_args__ = (
        UniqueConstraint(
            "video_id", "segment_idx", "task_name", name="uq_task_results_unit"
        ),
        Index("ix_task_results_video_task", "video_id", "task_name"),
    )


class Annotation(Base):
    """Final assembled JSON, one row per video."""

    __tablename__ = "annotations"

    video_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True
    )
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    video: Mapped[Video] = relationship(back_populates="annotation")


__all__ = ["Annotation", "Base", "Segment", "TaskResult", "Video"]
