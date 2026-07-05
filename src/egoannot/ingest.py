"""Dataset ingestion.

The pipeline supports four egocentric-navigation datasets: JAAD, ADVIO,
SCAND, NavWare. Each has its own on-disk layout, so we abstract behind
:class:`DatasetAdapter` — a callable that yields :class:`DiscoveredVideo`
records for a given root directory.

Only the JAAD adapter is implemented against a real layout. The other
three are STUBS that raise ``NotImplementedError`` with an explicit
``TODO: confirm on-disk layout for <dataset>`` message; layouts vary by
release and were not available at build time.

Ingestion is idempotent: re-running ``ingest`` over the same root will
skip videos already present in the ``videos`` table (matched by
``source_path``).
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class DiscoveredVideo:
    """One video candidate emitted by an adapter."""

    source_path: Path
    # A hint the adapter can pass forward (e.g. official train/val/test
    # split from JAAD's split_ids/); orchestrator/assembly honour a
    # non-empty value over the deterministic hash bucket.
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
                    source_path=f, split_hint=split_map.get(f.stem, "")
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
            "implement an adapter modeled on JAADAdapter and register it in "
            "egoannot.ingest.ADAPTERS. Root that was requested: "
            f"{root}"
        )


ADAPTERS: dict[SourceDataset, DatasetAdapter] = {
    SourceDataset.jaad: JAADAdapter(),
    SourceDataset.advio: _StubAdapter("advio"),
    SourceDataset.scand: _StubAdapter("scand"),
    SourceDataset.navware: _StubAdapter("navware"),
}


def _next_video_id(offset: int = 0) -> str:
    """Return the next available ``VID_%06d`` id.

    We assign ids at ingest time (rather than curate time as the SPEC
    hints) because it simplifies the resumability model: once a Video row
    exists, its id is stable and downstream tables can safely reference
    it. Curate only flips ``selected``.
    """
    with session_scope() as session:
        count = session.execute(select(func.count(Video.id))).scalar_one()
    return f"VID_{int(count) + 1 + offset:06d}"


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

            video_id = _next_video_id(offset=idx)
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
    "DatasetAdapter",
    "DiscoveredVideo",
    "JAADAdapter",
    "ingest_dataset",
]
