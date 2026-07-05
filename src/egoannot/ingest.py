"""Dataset ingestion.

The pipeline supports several egocentric-navigation datasets. Each has
its own on-disk layout, so we abstract behind :class:`DatasetAdapter` — a
callable that yields :class:`DiscoveredVideo` records for a given root
directory.

Real adapters:

- ``JAADAdapter``   — dashcam reference layout (JAAD_clips/*.mp4).
- ``ADVIOAdapter``  — handheld walking benchmark (advio-NN/iphone/frames.mov).
- ``GenericVideoFolderAdapter`` — recursive glob for any plain-video
  dataset (EgoBlind, SANPO, custom drops). One adapter, configurable
  glob; registered against ``egoblind`` and ``generic`` by default.

SCAND and NavWare remain STUBS that raise ``NotImplementedError`` with
an explicit "TODO: confirm on-disk layout" message; layouts were not
available at build time.

Ingestion is idempotent: re-running ``ingest`` over the same root will
skip videos already present in the ``videos`` table (matched by the
tuple ``(source_path, chunk_start_sec, chunk_end_sec)`` so chunk siblings
that share a parent file are still deduplicated correctly). When the
adapter emits a ``dataset_key``, the video id is derived deterministically
from it (``sha1`` -> 6-digit mod), so a given recording (or chunk of it)
gets the SAME VID across runs regardless of ingest order.

Long-recording chunking
-----------------------

Chunking happens AT INGEST TIME rather than at the frames stage. Rationale:

- Once a chunk is registered as its own Video row, it flows through the
  rest of the pipeline (curate, frames, annotate, assemble, export) as
  an ordinary short clip — no downstream code needs to know that it was
  ever part of something larger.
- Chunk boundaries are recorded on the Video row (``chunk_start_sec``,
  ``chunk_end_sec``), so ``frames`` can seek directly to the slice via
  ffmpeg ``-ss`` / ``-t`` without decoding the discarded portion.
- Extracted frames are chunk-relative (t=0 at chunk start). This keeps
  the segment math and every ``time_span`` in the assembled annotation
  meaningful with respect to that specific chunk — not the parent
  recording.

Controlled by ``frames.chunk_long_videos`` (default true) and
``frames.chunk_sec`` (default 40).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy import func, select

from .config import get_settings
from .db import Video, session_scope
from .media.frames import FrameExtractionError, plan_chunks, probe_video
from .schemas.enums import SourceDataset, VideoStatus

_log = structlog.stdlib.get_logger(__name__)

_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})
_ADVIO_RECORDING_RE = re.compile(r"^advio-\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveredVideo:
    """One video candidate emitted by an adapter."""

    source_path: Path
    # Optional stable per-recording key (e.g. "advio-01"). When set,
    # ingest derives a deterministic VID from ``sha1(key)`` so re-ingesting
    # produces the same ids regardless of ordering.
    dataset_key: str = ""
    # A hint the adapter can pass forward (e.g. official train/val/test
    # split); assembly honours a non-empty value over the deterministic
    # hash bucket.
    split_hint: str = ""


class DatasetAdapter(Protocol):
    """A callable that yields DiscoveredVideo records rooted at a path."""

    def discover(self, root: Path) -> Iterator[DiscoveredVideo]: ...


class JAADAdapter:
    """Real adapter for JAAD (Joint Attention in Autonomous Driving).

    JAAD ships as a folder of short mp4 clips named ``video_XXXX.mp4`` at
    720p. Some releases group them under ``JAAD_clips/`` with an optional
    ``split_ids/{train,val,test}_all.txt`` sibling listing. We honour the
    split lists if present; otherwise leave the split empty for the
    deterministic hash bucket to assign later.
    """

    def discover(self, root: Path) -> Iterator[DiscoveredVideo]:
        if not root.exists():
            _log.warning("ingest_root_missing", dataset="jaad", root=str(root))
            return
        split_map: dict[str, str] = {}
        split_dir = root / "split_ids"
        if split_dir.is_dir():
            for split_name in ("train", "val", "test"):
                p = split_dir / f"{split_name}_all.txt"
                if p.exists():
                    for line in p.read_text(encoding="utf-8").splitlines():
                        name = line.strip()
                        if name:
                            split_map[name] = split_name

        candidates = [root / "JAAD_clips", root / "clips", root]
        seen: set[Path] = set()
        for parent in candidates:
            if not parent.is_dir():
                continue
            for f in sorted(parent.iterdir()):
                if not f.is_file() or f.suffix.lower() not in _VIDEO_SUFFIXES:
                    continue
                if f in seen:
                    continue
                seen.add(f)
                yield DiscoveredVideo(
                    source_path=f,
                    dataset_key=f"jaad/{f.stem}",
                    split_hint=split_map.get(f.stem, ""),
                )


class ADVIOAdapter:
    """Real adapter for ADVIO (handheld walking egocentric benchmark).

    See module docstring for the on-disk layout. Only the iPhone RGB
    stream (``advio-NN/iphone/frames.mov``) is registered; Tango fisheye
    frames and CSV sidecars are hard-ignored.
    """

    def discover(self, root: Path) -> Iterator[DiscoveredVideo]:
        if not root.exists():
            _log.warning("ingest_root_missing", dataset="advio", root=str(root))
            return

        for rec_dir in sorted(root.iterdir()):
            if not rec_dir.is_dir():
                continue
            if not _ADVIO_RECORDING_RE.match(rec_dir.name):
                continue
            iphone_video = rec_dir / "iphone" / "frames.mov"
            if not iphone_video.is_file():
                _log.warning(
                    "advio_missing_iphone_video",
                    recording=rec_dir.name,
                    expected=str(iphone_video),
                )
                continue
            yield DiscoveredVideo(
                source_path=iphone_video,
                dataset_key=rec_dir.name.lower(),
                split_hint="",
            )


@dataclass(frozen=True)
class GenericVideoFolderAdapter:
    """Recursive glob-based adapter for any plain-video dataset.

    EgoBlind, SANPO, and custom user drops all fit this pattern: a folder
    (or nested folders) of video files, no bespoke naming convention.
    Instantiate with a dataset name and a glob (default ``**/*.mp4``).

    The ``dataset_key`` for each video is
    ``"<dataset_name>/<relative_path_without_ext>"``, so ids stay
    deterministic across re-ingests and stable if the parent folder is
    relocated (only the relative-to-root portion matters).
    """

    dataset_name: str
    glob: str = "**/*.mp4"
    extra_suffixes: tuple[str, ...] = field(default_factory=tuple)

    def discover(self, root: Path) -> Iterator[DiscoveredVideo]:
        if not root.exists():
            _log.warning(
                "ingest_root_missing", dataset=self.dataset_name, root=str(root)
            )
            return
        allowed = set(_VIDEO_SUFFIXES) | {s.lower() for s in self.extra_suffixes}
        seen: set[Path] = set()
        for path in sorted(root.glob(self.glob)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                continue
            if path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(root).with_suffix("")
            yield DiscoveredVideo(
                source_path=path,
                dataset_key=f"{self.dataset_name}/{rel.as_posix()}",
                split_hint="",
            )


class _StubAdapter:
    """Placeholder for datasets whose on-disk layout is unconfirmed."""

    def __init__(self, dataset_name: str) -> None:
        self._name = dataset_name

    def discover(self, root: Path) -> Iterator[DiscoveredVideo]:
        raise NotImplementedError(
            f"TODO: confirm on-disk layout for {self._name!r}; "
            "implement an adapter modeled on JAADAdapter/ADVIOAdapter or "
            "instantiate GenericVideoFolderAdapter with a suitable glob, "
            "and register it in egoannot.ingest.ADAPTERS. Root that was "
            f"requested: {root}"
        )


ADAPTERS: dict[SourceDataset, DatasetAdapter] = {
    SourceDataset.jaad: JAADAdapter(),
    SourceDataset.advio: ADVIOAdapter(),
    SourceDataset.scand: _StubAdapter("scand"),
    SourceDataset.navware: _StubAdapter("navware"),
    # EgoBlind ships plain video files; the exact folder shape is
    # user-controlled after downloading from Google Drive. The generic
    # recursive glob handles it (and any comparable plain-video dataset).
    SourceDataset.egoblind: GenericVideoFolderAdapter(
        dataset_name="egoblind", glob="**/*.mp4"
    ),
    # A general-purpose entry point for one-off drops; users pass a root
    # containing arbitrary mp4/mov/mkv/avi/webm files.
    SourceDataset.generic: GenericVideoFolderAdapter(
        dataset_name="generic",
        glob="**/*",
        extra_suffixes=(".mp4", ".mov", ".mkv", ".avi", ".webm"),
    ),
}


def _stable_vid_from_key(key: str) -> str:
    """Deterministic 6-digit VID from ``sha1(key)`` mod 1_000_000."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    return f"VID_{h % 1_000_000:06d}"


def _next_sequential_vid(offset: int = 0) -> str:
    """Return the next available ``VID_%06d`` id via a table-count offset."""
    with session_scope() as session:
        count = session.execute(select(func.count(Video.id))).scalar_one()
    return f"VID_{int(count) + 1 + offset:06d}"


def _resolve_video_id(dataset_key: str, source_path: str, offset: int) -> str:
    """Pick a VID, handling id collisions.

    If ``dataset_key`` is set, derive deterministically; on collision
    with a different source_path/chunk, bump linearly until free.
    """
    if not dataset_key:
        return _next_sequential_vid(offset=offset)

    vid = _stable_vid_from_key(dataset_key)
    for _ in range(1000):
        with session_scope() as session:
            existing = session.execute(
                select(Video).where(Video.id == vid)
            ).scalar_one_or_none()
        if existing is None or (
            existing.source_path == source_path
            and existing.chunk_start_sec == 0.0
            and existing.chunk_end_sec == 0.0
        ):
            return vid
        _log.warning(
            "vid_collision",
            key=dataset_key,
            wanted=vid,
            existing_source=existing.source_path,
            new_source=source_path,
        )
        n_bump = int(vid.removeprefix("VID_")) + 1
        vid = f"VID_{n_bump % 1_000_000:06d}"
    raise RuntimeError(
        f"unable to resolve a free VID after 1000 probes; key={dataset_key}"
    )


def ingest_dataset(dataset: str, root: Path) -> int:
    """Discover videos under ``root``, chunk long recordings, and register.

    Returns the number of newly-inserted Video rows (which may exceed the
    number of source files when chunking is enabled). Existing rows
    (matched by ``(source_path, chunk_start_sec, chunk_end_sec)``) are
    left untouched — this is a pure add-if-missing operation.
    """
    ds = _coerce_dataset(dataset)
    adapter = ADAPTERS[ds]

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ingest root does not exist: {root}")

    settings = get_settings()
    inserted = 0
    seq_offset = 0
    for candidate in adapter.discover(root):
        try:
            meta = probe_video(candidate.source_path)
        except FrameExtractionError as exc:
            _log.warning(
                "ingest_probe_failed",
                source_path=str(candidate.source_path),
                err=str(exc),
            )
            continue

        chunks = plan_chunks(
            meta.duration_sec,
            chunk_sec=settings.frames.chunk_sec,
            enabled=settings.frames.chunk_long_videos,
        )
        if not chunks:
            continue

        n_chunks = len(chunks)
        source_str = str(candidate.source_path)

        for chunk_idx, (start, end) in enumerate(chunks):
            # Zero on both fields signals "no chunk" (whole file) so
            # single-chunk recordings behave exactly as before.
            store_start = start if n_chunks > 1 else 0.0
            store_end = end if n_chunks > 1 else 0.0

            chunk_key = (
                candidate.dataset_key
                if n_chunks == 1
                else f"{candidate.dataset_key}#c{chunk_idx:03d}"
            )

            with session_scope() as session:
                existing = session.execute(
                    select(Video).where(
                        Video.source_path == source_str,
                        Video.chunk_start_sec == store_start,
                        Video.chunk_end_sec == store_end,
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    continue

                video_id = _resolve_video_id(chunk_key, source_str, offset=seq_offset)
                seq_offset += 1
                session.add(
                    Video(
                        id=video_id,
                        source_dataset=ds.value,
                        source_path=source_str,
                        split=candidate.split_hint,
                        status=VideoStatus.pending.value,
                        chunk_start_sec=store_start,
                        chunk_end_sec=store_end,
                    )
                )
                inserted += 1

    _log.info("ingest_done", dataset=ds.value, root=str(root), inserted=inserted)
    return inserted


def _coerce_dataset(name: str) -> SourceDataset:
    member, coerced = SourceDataset.coerce(name)
    if coerced:
        _log.warning(
            "ingest_dataset_coerced",
            raw=name,
            coerced_to=member.value,
            note="check the --dataset flag; unknown values default to jaad",
        )
    return member


__all__ = [
    "ADAPTERS",
    "ADVIOAdapter",
    "DatasetAdapter",
    "DiscoveredVideo",
    "GenericVideoFolderAdapter",
    "JAADAdapter",
    "ingest_dataset",
]
