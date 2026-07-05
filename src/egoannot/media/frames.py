"""Frame extraction, segmentation, sampling, and base64 encoding.

Pipeline:
    ffprobe                 -> VideoMeta (duration, fps, resolution)
    ffmpeg -vf fps=N,scale  -> per-video JPEGs on disk (frames/<vid>/f_%05d.jpg)
    segment_video           -> list of [start,end) segments
    sample_segment_frames   -> N SampledFrame objects with timestamps
    encode_data_uri         -> re-encoded base64 JPEG data URI for model calls

Constraints:
- vLLM max_model_len=16384. A single call carries ~12 downscaled frames plus
  a compact prompt; the JPEG re-encode with quality=80 and long-side<=1280
  keeps each frame under the token budget.
- Every path emitted lives under ``settings.paths.data_dir`` (never /).
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog
from PIL import Image

from ..config import get_settings

_log = structlog.stdlib.get_logger(__name__)


class FrameExtractionError(RuntimeError):
    """Raised when ffprobe/ffmpeg fails or output is unusable."""


@dataclass(frozen=True)
class VideoMeta:
    duration_sec: float
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class Segment:
    idx: int
    start_sec: float
    end_sec: float

    @property
    def length_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class SampledFrame:
    """One frame chosen from the extracted set for a model call."""

    path: Path
    timestamp_sec: float


def _require_binary(name: str) -> str:
    binpath = shutil.which(name)
    if binpath is None:
        raise FrameExtractionError(f"required binary not found on PATH: {name}")
    return binpath


def probe_video(video_path: Path) -> VideoMeta:
    """Extract duration/fps/resolution via ffprobe."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FrameExtractionError(f"ffprobe failed on {video_path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")

    stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if stream is None:
        raise FrameExtractionError(f"no video stream in {video_path}")

    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den) else 0.0
    else:
        fps = float(rate)

    if duration <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise FrameExtractionError(
            f"invalid metadata for {video_path}: duration={duration} fps={fps} "
            f"resolution={width}x{height}"
        )

    return VideoMeta(duration_sec=duration, fps=fps, width=width, height=height)


def _scale_filter(meta: VideoMeta, max_long_side: int) -> str | None:
    """Build an ffmpeg scale filter that caps the long side, keeping aspect.

    Returns None if the source already fits under the cap.
    """
    long_side = max(meta.width, meta.height)
    if long_side <= max_long_side:
        return None
    if meta.width >= meta.height:
        return f"scale={max_long_side}:-2"
    return f"scale=-2:{max_long_side}"


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    meta: VideoMeta,
    *,
    fps: int | None = None,
    max_long_side: int | None = None,
    jpeg_quality: int | None = None,
) -> list[Path]:
    """Extract candidate frames at ``fps`` into ``frame_dir``, downscaling.

    Idempotent: if ``frame_dir`` already contains at least one frame, this
    function is a no-op and returns the sorted existing list. Callers who
    want a fresh extraction should remove the directory first.
    """
    settings = get_settings()
    fps = fps if fps is not None else settings.frames.fps
    max_long_side = max_long_side if max_long_side is not None else settings.frames.max_long_side
    jpeg_quality = jpeg_quality if jpeg_quality is not None else settings.frames.jpeg_quality

    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("f_*.jpg"))
    if existing:
        _log.info(
            "frames_reused", video=str(video_path), count=len(existing), frame_dir=str(frame_dir)
        )
        return existing

    ffmpeg = _require_binary("ffmpeg")
    filters = [f"fps={fps}"]
    scale = _scale_filter(meta, max_long_side)
    if scale is not None:
        filters.append(scale)
    vf = ",".join(filters)

    out_pattern = frame_dir / "f_%05d.jpg"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", str(_pil_to_ffmpeg_quality(jpeg_quality)),
        str(out_pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FrameExtractionError(f"ffmpeg failed on {video_path}: {proc.stderr.strip()}")

    frames = sorted(frame_dir.glob("f_*.jpg"))
    if not frames:
        raise FrameExtractionError(f"no frames extracted from {video_path}")
    _log.info(
        "frames_extracted", video=str(video_path), count=len(frames),
        fps=fps, max_long_side=max_long_side, frame_dir=str(frame_dir),
    )
    return frames


def _pil_to_ffmpeg_quality(pil_quality: int) -> int:
    """Map a PIL-style quality (1-100, 100=best) to ffmpeg -q:v (2-31, 2=best).

    Rough inverse-linear mapping; exact value doesn't matter because model
    calls re-encode via PIL anyway.
    """
    pil_quality = max(1, min(100, pil_quality))
    return max(2, min(31, round(31 - (pil_quality * 0.29))))


def segment_video(
    duration_sec: float,
    *,
    segment_max_sec: float | None = None,
    segment_len_sec: float | None = None,
) -> list[Segment]:
    """Partition the clip into non-overlapping segments per SPEC rules."""
    settings = get_settings()
    segment_max_sec = segment_max_sec if segment_max_sec is not None else settings.frames.segment_max_sec
    segment_len_sec = segment_len_sec if segment_len_sec is not None else settings.frames.segment_len_sec

    if duration_sec <= 0:
        return []
    if duration_sec <= segment_max_sec:
        return [Segment(idx=0, start_sec=0.0, end_sec=duration_sec)]

    segs: list[Segment] = []
    idx = 0
    t = 0.0
    while t < duration_sec:
        end = min(t + segment_len_sec, duration_sec)
        segs.append(Segment(idx=idx, start_sec=t, end_sec=end))
        idx += 1
        t = end
    return segs


def sample_segment_frames(
    frames: list[Path],
    extraction_fps: int,
    segment: Segment,
    *,
    per_segment: int | None = None,
) -> list[SampledFrame]:
    """Uniformly sample ``per_segment`` frames from within ``segment``.

    Frame timestamp is derived from the 1-indexed filename position and
    the extraction fps: ``t = (i - 0.5) / fps`` (centre-of-bucket).
    Only frames whose timestamp lies in [start, end) are eligible; if the
    eligible pool is smaller than ``per_segment``, all are returned.
    """
    settings = get_settings()
    per_segment = per_segment if per_segment is not None else settings.frames.per_segment

    eligible: list[tuple[Path, float]] = []
    for p in frames:
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        ts = (idx - 0.5) / extraction_fps
        if segment.start_sec <= ts < segment.end_sec:
            eligible.append((p, ts))

    if not eligible:
        return []
    if len(eligible) <= per_segment:
        return [SampledFrame(path=p, timestamp_sec=ts) for p, ts in eligible]

    # Uniform indices across the eligible pool.
    n = len(eligible)
    step = n / per_segment
    picks = [eligible[min(int(step * (k + 0.5)), n - 1)] for k in range(per_segment)]
    return [SampledFrame(path=p, timestamp_sec=ts) for p, ts in picks]


def encode_data_uri(
    path: Path,
    *,
    max_long_side: int | None = None,
    jpeg_quality: int | None = None,
) -> str:
    """Read a frame, re-encode as JPEG under the size cap, return data URI.

    Uses PIL so the encoding matches whatever quality/size cap is configured
    at call time, independent of the on-disk cache produced by ffmpeg.
    """
    settings = get_settings()
    max_long_side = max_long_side if max_long_side is not None else settings.frames.max_long_side
    jpeg_quality = jpeg_quality if jpeg_quality is not None else settings.frames.jpeg_quality

    with Image.open(path) as raw:
        img = raw.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def sampled_frames_to_content(
    frames: list[SampledFrame],
    *,
    max_long_side: int | None = None,
    jpeg_quality: int | None = None,
) -> list[dict[str, object]]:
    """Convert sampled frames to OpenAI-compatible content blocks.

    Each frame becomes two blocks: a ``[t=SS.Ss]`` text marker for temporal
    grounding, then the ``image_url`` block with the base64 JPEG data URI.
    """
    content: list[dict[str, object]] = []
    for f in frames:
        content.append({"type": "text", "text": f"[t={f.timestamp_sec:.1f}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_data_uri(
                        f.path,
                        max_long_side=max_long_side,
                        jpeg_quality=jpeg_quality,
                    )
                },
            }
        )
    return content


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
