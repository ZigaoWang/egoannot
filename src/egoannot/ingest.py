"""Dataset ingestion.

The pipeline supports four egocentric-navigation datasets: JAAD, ADVIO,
SCAND, NavWare. Each has its own on-disk layout, so we abstract behind
:class:`DatasetAdapter` — a callable that yields :class:`DiscoveredVideo`
records for a given root directory.

Real adapters: JAAD (dashcam, included as reference), ADVIO (handheld
walking egocentric — primary content fit). SCAND and NavWare remain
STUBS that raise ``NotImplementedError`` with an explicit "TODO: confirm
on-disk layout" message.

Ingestion is idempotent: re-running ``ingest`` over the same root will
skip videos already present in the ``videos`` table (matched by
``source_path``). When the adapter emits a ``dataset_key``, the video id
is derived deterministically from it (``sha1`` -> 6-digit mod), so a
given recording gets the SAME VID across runs regardless of ingest order.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy import func, select

from .db import Video, session_scope
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

        # JAAD clips live under JAAD_clips/ or clips/, but we also handle
        # a flat layout for user convenience.
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

    ADVIO ships as a tree of recording folders named ``advio-01``,
    ``advio-02``, ..., each containing two synchronised camera streams:

        <root>/
            advio-01/
                iphone/frames.mov      <- RGB, ~narrow FOV, walking POV  (USED)
                iphone/frames.csv      <- IMU/pose sidecar               (ignored)
                tango/frames.mov       <- fisheye Tango stream           (IGNORED)
                tango/frames.csv                                          (ignored)
                ...

    Per SPEC, only the iPhone RGB stream is a walking-egocentric fit; the
    Tango fisheye has the wrong FOV for our downstream use. We
    hard-ignore ``tango/`` and all sidecar CSVs.

    ADVIO does not ship a train/val/test split; assembly falls back to
    the deterministic sha1 bucketing (same behaviour as JAAD without
    ``split_ids/``). Recordings are minutes long, so most will be
    multi-segment.
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


class _StubAdapter:
    """Placeholder for datasets whose on-disk layout is unconfirmed.

    Rather than guess a layout and silently mis-ingest, we refuse and
    surface a TODO to the operator. Once the actual layout is known,
    replace this stub with a real adapter following ``JAADAdapter``.
    """

    def __init__(self, dataset_name: str) -> None:
        self._name = dataset_name

    def discover(self, root: Path) -> Iterator[DiscoveredVideo]:
        raise NotImplementedError(
            f"TODO: confirm on-disk layout for {self._name!r}; "
            "implement an adapter modeled on JAADAdapter/ADVIOAdapter and "
            "register it in egoannot.ingest.ADAPTERS. Root that was requested: "
            f"{root}"
        )


ADAPTERS: dict[SourceDataset, DatasetAdapter] = {
    SourceDataset.jaad: JAADAdapter(),
    SourceDataset.advio: ADVIOAdapter(),
    SourceDataset.scand: _StubAdapter("scand"),
    SourceDataset.navware: _StubAdapter("navware"),
}


def _stable_vid_from_key(key: str) -> str:
    """Deterministic 6-digit VID from ``sha1(key)`` mod 1_000_000."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    return f"VID_{h % 1_000_000:06d}"


def _next_sequential_vid(offset: int = 0) -> str:
    """Return the next available ``VID_%06d`` id via a table-count offset.

    Used when the adapter does not provide a ``dataset_key`` — the id is
    stable per (source_path, first-insertion-order) but not across
    reordered ingestions.
    """
    with session_scope() as session:
        count = session.execute(select(func.count(Video.id))).scalar_one()
    return f"VID_{int(count) + 1 + offset:06d}"


def _resolve_video_id(candidate: DiscoveredVideo, offset: int) -> str:
    """Pick a VID for one candidate, handling id collisions.

    If ``dataset_key`` is set, derive deterministically; on collision
    with a different source_path, bump by one until free. If ``dataset_key``
    is empty, fall back to the sequential scheme.
    """
    if not candidate.dataset_key:
        return _next_sequential_vid(offset=offset)

    vid = _stable_vid_from_key(candidate.dataset_key)
    src = str(candidate.source_path)
    for _ in range(1000):
        with session_scope() as session:
            existing = session.execute(
                select(Video).where(Video.id == vid)
            ).scalar_one_or_none()
        if existing is None or existing.source_path == src:
            return vid
        _log.warning(
            "vid_collision",
            key=candidate.dataset_key,
            wanted=vid,
            existing_source=existing.source_path,
            new_source=src,
        )
        # Deterministic linear probe: append a counter to the key and rehash.
        n_bump = int(vid.removeprefix("VID_")) + 1
        vid = f"VID_{n_bump % 1_000_000:06d}"
    raise RuntimeError(
        f"unable to resolve a free VID after 1000 probes; key={candidate.dataset_key}"
    )


def ingest_dataset(dataset: str, root: Path) -> int:
    """Discover videos under ``root`` and register them in the database.

    Returns the number of newly-inserted rows. Existing rows (matched by
    ``source_path``) are left untouched — this is a pure add-if-missing
    operation.
    """
    ds = _coerce_dataset(dataset)
    adapter = ADAPTERS[ds]

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ingest root does not exist: {root}")

    inserted = 0
    idx = 0
    for candidate in adapter.discover(root):
        with session_scope() as session:
            existing = session.execute(
                select(Video).where(Video.source_path == str(candidate.source_path))
            ).scalar_one_or_none()
            if existing is not None:
                continue

            video_id = _resolve_video_id(candidate, offset=idx)
            idx += 1
            session.add(
                Video(
                    id=video_id,
                    source_dataset=ds.value,
                    source_path=str(candidate.source_path),
                    split=candidate.split_hint,
                    status=VideoStatus.pending.value,
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
    "JAADAdapter",
    "ingest_dataset",
]
