"""Write assembled annotations to disk.

Every video whose ``Annotation`` row exists is written to
``<outputs_dir>/<video_id>.json``. When ``--jsonl`` is set, the
combined ``all.jsonl`` file is (re)written from the same rows.

Files are written atomically (write to a ``.tmp`` sibling, then rename)
so that a crash mid-write cannot leave a partial file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import structlog
from sqlalchemy import select

from .config import get_settings
from .db import Annotation, Video, session_scope

_log = structlog.stdlib.get_logger(__name__)


def _outputs_dir() -> Path:
    settings = get_settings()
    d = settings.paths.outputs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def export_video(video_id: str) -> Path:
    """Write one annotation to ``outputs/<video_id>.json`` and return its path."""
    with session_scope() as session:
        ann = session.get(Annotation, video_id)
        if ann is None:
            raise KeyError(f"no assembled annotation for {video_id}")
        payload_json = ann.payload_json
    dst = _outputs_dir() / f"{video_id}.json"
    _atomic_write(dst, payload_json)
    _log.info("export_video_written", video_id=video_id, path=str(dst))
    return dst


def export_all(*, jsonl: bool = False) -> tuple[list[Path], Path | None]:
    """Write every assembled annotation to disk.

    Returns (list_of_per_video_paths, jsonl_path_or_None).
    """
    with session_scope() as session:
        rows = session.execute(
            select(Video.id, Annotation.payload_json)
            .join(Annotation, Annotation.video_id == Video.id)
            .order_by(Video.id)
        ).all()

    dir_ = _outputs_dir()
    per_video: list[Path] = []
    lines: list[str] = []
    for vid, payload_json in rows:
        dst = dir_ / f"{vid}.json"
        _atomic_write(dst, payload_json)
        per_video.append(dst)
        if jsonl:
            # Collapse the JSON object onto one line for jsonl.
            data = json.loads(payload_json)
            lines.append(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    jsonl_path: Path | None = None
    if jsonl:
        jsonl_path = dir_ / "all.jsonl"
        _atomic_write(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))
    _log.info(
        "export_all_done",
        count=len(per_video),
        outputs_dir=str(dir_),
        jsonl=str(jsonl_path) if jsonl_path else None,
    )
    return per_video, jsonl_path


__all__ = ["export_all", "export_video"]


# Kept for type-checker friendliness; unused local import guard removed.
_ = Iterable
