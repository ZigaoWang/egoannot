"""Adapter discovery tests. No ffmpeg or real videos required."""

from __future__ import annotations

import re
from pathlib import Path

from egoannot.db import Video, session_scope
from egoannot.ingest import (
    ADVIOAdapter,
    GenericVideoFolderAdapter,
    JAADAdapter,
    ingest_dataset,
)


def _touch(p: Path, size: int = 16) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)
    return p


# ---------- JAAD --------------------------------------------------------


def test_jaad_adapter_flat_layout(tmp_path: Path) -> None:
    root = tmp_path / "jaad"
    (root / "JAAD_clips").mkdir(parents=True)
    _touch(root / "JAAD_clips" / "video_0001.mp4")
    _touch(root / "JAAD_clips" / "video_0002.mp4")
    _touch(root / "JAAD_clips" / "readme.txt")  # decoy
    (root / "split_ids").mkdir()
    (root / "split_ids" / "train_all.txt").write_text("video_0001\nvideo_0002\n")

    found = list(JAADAdapter().discover(root))
    names = sorted(d.source_path.name for d in found)
    assert names == ["video_0001.mp4", "video_0002.mp4"]
    # Split hint honoured when provided by JAAD.
    assert {d.source_path.stem: d.split_hint for d in found} == {
        "video_0001": "train",
        "video_0002": "train",
    }
    # Dataset key is folder-based, stable.
    assert {d.dataset_key for d in found} == {"jaad/video_0001", "jaad/video_0002"}


# ---------- ADVIO -------------------------------------------------------


def test_advio_adapter_picks_iphone_only(tmp_path: Path) -> None:
    root = tmp_path / "advio"
    for n in (1, 2, 3):
        rec = root / f"advio-{n:02d}"
        # The one video we want.
        _touch(rec / "iphone" / "frames.mov", size=32)
        # Decoys that MUST be ignored.
        _touch(rec / "iphone" / "frames.csv")
        _touch(rec / "tango" / "frames.mov")
        _touch(rec / "tango" / "frames.csv")
        _touch(rec / "arkit_pose.csv")

    # Extra top-level clutter to ignore.
    _touch(root / "README.md")
    (root / "not-advio-folder").mkdir()
    _touch(root / "not-advio-folder" / "iphone" / "frames.mov")

    found = list(ADVIOAdapter().discover(root))
    paths = sorted(str(d.source_path) for d in found)

    # Exactly three: advio-01/iphone/frames.mov ... advio-03/...
    assert len(found) == 3
    for d in found:
        assert d.source_path.name == "frames.mov"
        assert d.source_path.parent.name == "iphone"
        # Never a tango frame.
        assert "tango" not in d.source_path.parts
        # No CSV sneaking through.
        assert d.source_path.suffix == ".mov"
        # Stable per-recording key.
        assert d.dataset_key.startswith("advio-")
        # ADVIO ships no split; hint stays empty for hash bucketing.
        assert d.split_hint == ""

    keys = sorted(d.dataset_key for d in found)
    assert keys == ["advio-01", "advio-02", "advio-03"]

    # No path leaked from the non-advio-* sibling.
    assert all("not-advio-folder" not in p for p in paths)


def test_advio_missing_iphone_video_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "advio"
    # advio-01 has iphone/frames.mov; advio-02 does NOT (only tango).
    _touch(root / "advio-01" / "iphone" / "frames.mov")
    _touch(root / "advio-02" / "tango" / "frames.mov")

    found = list(ADVIOAdapter().discover(root))
    keys = [d.dataset_key for d in found]
    assert keys == ["advio-01"]


def test_advio_missing_root_yields_nothing(tmp_path: Path) -> None:
    found = list(ADVIOAdapter().discover(tmp_path / "does_not_exist"))
    assert found == []


# ---------- ingest_dataset determinism ---------------------------------


def test_advio_ingest_ids_are_deterministic(
    tmp_path: Path, tmp_pipeline, monkeypatch
) -> None:
    root = tmp_path / "advio"
    for n in (1, 2, 3):
        _touch(root / f"advio-{n:02d}" / "iphone" / "frames.mov")

    # Bypass ffprobe: the "videos" here are zero-byte stubs. Return a
    # short synthetic duration so chunking keeps everything single-chunk.
    from egoannot import ingest as ingest_mod
    from egoannot.media.frames import VideoMeta

    def _fake_probe(_path: Path) -> VideoMeta:
        return VideoMeta(duration_sec=20.0, fps=30.0, width=640, height=480)

    monkeypatch.setattr(ingest_mod, "probe_video", _fake_probe)

    inserted_first = ingest_dataset("advio", root)
    assert inserted_first == 3
    with session_scope() as s:
        rows = s.execute(
            Video.__table__.select().order_by(Video.source_path)
        ).all()
        first_ids = [r.id for r in rows]

    # Re-ingest: no new rows, ids unchanged.
    inserted_second = ingest_dataset("advio", root)
    assert inserted_second == 0
    with session_scope() as s:
        rows = s.execute(
            Video.__table__.select().order_by(Video.source_path)
        ).all()
        second_ids = [r.id for r in rows]

    assert first_ids == second_ids
    # All ids match the strict VID_\d{6} shape (needed downstream).
    for vid in first_ids:
        assert re.match(r"^VID_\d{6}$", vid), vid


def test_advio_ingest_chunks_long_recording(
    tmp_path: Path, tmp_pipeline, monkeypatch
) -> None:
    """A 260s ADVIO clip splits into 7 chunks; ids are chunk-specific."""
    root = tmp_path / "advio"
    _touch(root / "advio-01" / "iphone" / "frames.mov")

    from egoannot import ingest as ingest_mod
    from egoannot.media.frames import VideoMeta

    def _fake_probe(_path: Path) -> VideoMeta:
        return VideoMeta(duration_sec=260.0, fps=30.0, width=1920, height=1080)

    monkeypatch.setattr(ingest_mod, "probe_video", _fake_probe)

    inserted = ingest_dataset("advio", root)
    assert inserted == 7
    with session_scope() as s:
        rows = s.execute(
            Video.__table__.select().order_by(Video.chunk_start_sec)
        ).all()
        windows = [(r.chunk_start_sec, r.chunk_end_sec) for r in rows]
    assert windows == [
        (0.0, 40.0),
        (40.0, 80.0),
        (80.0, 120.0),
        (120.0, 160.0),
        (160.0, 200.0),
        (200.0, 240.0),
        (240.0, 260.0),
    ]
    # Every chunk has a unique deterministic id.
    assert len({r[0] for r in windows}) == 7



# ---------- Generic video folder adapter -------------------------------


def test_generic_adapter_picks_videos_from_nested_tree(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _touch(root / "top.mp4")
    _touch(root / "day1" / "clip_a.mp4")
    _touch(root / "day1" / "clip_b.mp4")
    _touch(root / "day2" / "session_1" / "video.mp4")
    # Decoys — must be ignored.
    _touch(root / "day1" / "notes.txt")
    _touch(root / "day1" / "thumb.jpg")
    _touch(root / "annotations.csv")
    _touch(root / "day2" / "session_1" / "capture.log")

    adapter = GenericVideoFolderAdapter(dataset_name="egoblind", glob="**/*.mp4")
    found = list(adapter.discover(root))
    paths = sorted(d.source_path.name for d in found)
    assert paths == ["clip_a.mp4", "clip_b.mp4", "top.mp4", "video.mp4"]

    keys = sorted(d.dataset_key for d in found)
    assert keys == [
        "egoblind/day1/clip_a",
        "egoblind/day1/clip_b",
        "egoblind/day2/session_1/video",
        "egoblind/top",
    ]
    for d in found:
        assert d.source_path.suffix == ".mp4"
        assert d.split_hint == ""


def test_generic_adapter_missing_root_yields_nothing(tmp_path: Path) -> None:
    adapter = GenericVideoFolderAdapter(dataset_name="egoblind")
    assert list(adapter.discover(tmp_path / "nope")) == []


def test_generic_adapter_deterministic_across_reruns(tmp_path: Path) -> None:
    root = tmp_path / "egoblind_corpus"
    _touch(root / "v_00001.mp4")
    _touch(root / "v_00002.mp4")

    adapter = GenericVideoFolderAdapter(dataset_name="egoblind")
    first = [d.dataset_key for d in adapter.discover(root)]
    second = [d.dataset_key for d in adapter.discover(root)]
    assert first == second

