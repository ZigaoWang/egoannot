"""Chunk math + frame math tests. No ffmpeg required."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from egoannot.media.frames import (
    Segment,
    plan_chunks,
    sample_segment_frames,
    segment_video,
)


def test_segment_video_single_short():
    segs = segment_video(30.0, segment_max_sec=45.0, segment_len_sec=20.0)
    assert len(segs) == 1
    assert segs[0].start_sec == 0.0
    assert segs[0].end_sec == 30.0


def test_segment_video_multi_long():
    segs = segment_video(70.0, segment_max_sec=45.0, segment_len_sec=20.0)
    assert [(s.start_sec, s.end_sec) for s in segs] == [
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
        (60.0, 70.0),
    ]


def test_segment_video_empty_when_zero():
    assert segment_video(0.0) == []


def _fake_frames(n: int) -> list[Path]:
    return [Path(f"f_{i:05d}.jpg") for i in range(1, n + 1)]


def test_sample_segment_frames_uniform():
    frames = _fake_frames(200)
    seg = Segment(idx=0, start_sec=0.0, end_sec=20.0)
    sampled = sample_segment_frames(frames, extraction_fps=10, segment=seg, per_segment=12)
    assert len(sampled) == 12
    ts = [s.timestamp_sec for s in sampled]
    assert ts == sorted(ts)
    assert all(seg.start_sec <= t < seg.end_sec for t in ts)


def test_sample_segment_frames_returns_all_when_pool_small():
    frames = _fake_frames(5)
    seg = Segment(idx=0, start_sec=0.0, end_sec=20.0)
    sampled = sample_segment_frames(frames, extraction_fps=10, segment=seg, per_segment=12)
    assert len(sampled) == 5


def test_sample_segment_frames_disjoint_window():
    frames = _fake_frames(200)
    seg = Segment(idx=1, start_sec=20.0, end_sec=40.0)
    sampled = sample_segment_frames(frames, extraction_fps=10, segment=seg, per_segment=12)
    assert sampled == []


def test_sample_segment_frames_center_of_bucket():
    frames = _fake_frames(10)
    seg = Segment(idx=0, start_sec=0.0, end_sec=1.0)
    sampled = sample_segment_frames(frames, extraction_fps=10, segment=seg, per_segment=10)
    ts = [round(s.timestamp_sec, 3) for s in sampled]
    assert ts[0] == 0.05
    assert ts[-1] == 0.95


# ------------------------------------------------------------------ chunks


def test_plan_chunks_260s_yields_seven_chunks():
    chunks = plan_chunks(260.0, chunk_sec=40.0, enabled=True)
    assert len(chunks) == 7
    # Consecutive, no gaps, no overlap.
    assert chunks[0] == (0.0, 40.0)
    for prev, curr in pairwise(chunks):
        assert prev[1] == curr[0]
    # Last chunk covers the tail.
    assert chunks[-1] == (240.0, 260.0)


def test_plan_chunks_short_recording_stays_single():
    assert plan_chunks(30.0, chunk_sec=40.0, enabled=True) == [(0.0, 30.0)]


def test_plan_chunks_exact_boundary_stays_single():
    assert plan_chunks(40.0, chunk_sec=40.0, enabled=True) == [(0.0, 40.0)]


def test_plan_chunks_just_over_boundary_yields_two():
    chunks = plan_chunks(41.0, chunk_sec=40.0, enabled=True)
    assert chunks == [(0.0, 40.0), (40.0, 41.0)]


def test_plan_chunks_disabled_returns_full_range():
    assert plan_chunks(200.0, chunk_sec=40.0, enabled=False) == [(0.0, 200.0)]


def test_plan_chunks_zero_or_negative_returns_empty():
    assert plan_chunks(0.0, chunk_sec=40.0, enabled=True) == []
    assert plan_chunks(-5.0, chunk_sec=40.0, enabled=True) == []
