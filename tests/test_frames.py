"""Tests for :mod:`egoannot.media.frames` math (no ffmpeg required)."""

from __future__ import annotations

from pathlib import Path

from egoannot.media.frames import Segment, sample_segment_frames, segment_video


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
    # Center-of-bucket timestamps should be monotonically increasing.
    ts = [s.timestamp_sec for s in sampled]
    assert ts == sorted(ts)
    # All within segment bounds.
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
    # First bucket at t=(1-0.5)/10=0.05, last at t=(200-0.5)/10=19.95.
    # Segment [20,40) contains nothing.
    assert sampled == []


def test_sample_segment_frames_center_of_bucket():
    # First frame timestamp should be 0.05s at fps=10 (i=1 -> (1-0.5)/10)
    frames = _fake_frames(10)
    seg = Segment(idx=0, start_sec=0.0, end_sec=1.0)
    sampled = sample_segment_frames(frames, extraction_fps=10, segment=seg, per_segment=10)
    ts = [round(s.timestamp_sec, 3) for s in sampled]
    assert ts[0] == 0.05
    assert ts[-1] == 0.95
