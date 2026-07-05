"""Frame extraction + sampling + base64 encoding for model calls."""

from __future__ import annotations

from .frames import (
    FrameExtractionError,
    SampledFrame,
    Segment,
    VideoMeta,
    encode_data_uri,
    extract_frames,
    probe_video,
    sample_segment_frames,
    sampled_frames_to_content,
    segment_video,
)

__all__ = [
    "FrameExtractionError",
    "SampledFrame",
    "Segment",
    "VideoMeta",
    "encode_data_uri",
    "extract_frames",
    "probe_video",
    "sample_segment_frames",
    "sampled_frames_to_content",
    "segment_video",
]
