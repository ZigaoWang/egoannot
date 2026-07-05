"""Smoke test for DB models. Run: python scripts/smoke_db.py"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from egoannot.db import Annotation, Segment, TaskResult, Video, init_engine, session_scope


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "x.db"
    init_engine(p)

    with session_scope() as s:
        v = Video(
            id="VID_000001",
            source_dataset="jaad",
            source_path="/tmp/a.mp4",
            status="pending",
        )
        s.add(v)
        s.flush()
        s.add(Segment(video_id="VID_000001", idx=0, start_sec=0.0, end_sec=20.0))
        s.add(
            TaskResult(
                video_id="VID_000001",
                segment_idx=0,
                task_name="scene",
                raw_response="{}",
                parsed_json="{}",
                ok=True,
                attempts=1,
            )
        )
        s.add(Annotation(video_id="VID_000001", payload_json="{}"))

    with session_scope() as s:
        v = s.get(Video, "VID_000001")
        assert v is not None
        print(
            "video status=", v.status,
            "segs=", len(v.segments),
            "tasks=", len(v.task_results),
            "ann=", v.annotation is not None,
        )
        # unique constraint check
        try:
            with session_scope() as s2:
                s2.add(
                    TaskResult(
                        video_id="VID_000001",
                        segment_idx=0,
                        task_name="scene",
                        raw_response="{}",
                        parsed_json="{}",
                        ok=True,
                        attempts=1,
                    )
                )
        except Exception as e:  # noqa: BLE001
            print("unique constraint enforced:", type(e).__name__)

    print("db path=", p, "size=", p.stat().st_size)
    os.remove(p)


if __name__ == "__main__":
    main()
