"""Streamlit dashboard for monitoring and reviewing the pipeline.

Read-only against pipeline state. Never modifies the DB, never re-runs a
stage. Two views:

- Overview: status counts, failed videos, most-recent updates, aggregate
  stats over assembled annotations.
- Per-clip review: pick a video_id, view the assembled JSON, the sampled
  frames as a gallery with timestamps, and readable tables for
  key_elements + qa_pairs.

Launch:
    egoannot dashboard                           # via the Typer CLI
    streamlit run src/egoannot/dashboard.py      # direct

View remotely via an SSH tunnel:
    ssh -L 8501:localhost:8501 <host>
    open http://localhost:8501 in your local browser
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st
from sqlalchemy import desc, select

# Streamlit runs this file via `streamlit run <path>`, which loads it as
# ``__main__`` — not as ``egoannot.dashboard``. Relative imports would
# raise ``ImportError: attempted relative import with no known parent``.
# Use absolute imports so both entrypoints (Typer CLI and direct
# ``streamlit run``) work.
from egoannot.config import get_settings
from egoannot.db import Annotation, TaskResult, Video, init_engine, session_scope


def _boot_once() -> None:
    """Initialise settings + DB engine once per session."""
    if st.session_state.get("_egoannot_booted"):
        return
    settings = get_settings()
    init_engine(settings.paths.db_path)
    st.session_state["_egoannot_booted"] = True


STATUS_ORDER = (
    "pending",
    "curated",
    "frames_done",
    "tasks_done",
    "assembled",
    "failed",
)


def _load_status_counts() -> dict[str, dict[str, int]]:
    """Return {status: {'selected': n, 'not_selected': n}}."""
    out: dict[str, dict[str, int]] = {s: {"selected": 0, "not_selected": 0} for s in STATUS_ORDER}
    with session_scope() as session:
        rows = session.execute(select(Video.status, Video.selected)).all()
    for st_, sel in rows:
        out.setdefault(st_, {"selected": 0, "not_selected": 0})
        key = "selected" if sel else "not_selected"
        out[st_][key] += 1
    return out


def _load_failed(limit: int = 50) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.execute(
            select(Video.id, Video.source_dataset, Video.error, Video.updated_at)
            .where(Video.status == "failed")
            .order_by(desc(Video.updated_at))
            .limit(limit)
        ).all()
    return [
        {"video_id": r.id, "dataset": r.source_dataset, "error": r.error or "", "updated_at": str(r.updated_at)}
        for r in rows
    ]


def _load_recent(limit: int = 25) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.execute(
            select(
                Video.id,
                Video.source_dataset,
                Video.status,
                Video.selected,
                Video.updated_at,
            )
            .order_by(desc(Video.updated_at))
            .limit(limit)
        ).all()
    return [
        {
            "video_id": r.id,
            "dataset": r.source_dataset,
            "status": r.status,
            "selected": bool(r.selected),
            "updated_at": str(r.updated_at),
        }
        for r in rows
    ]


def _load_assembled_ids() -> list[str]:
    with session_scope() as session:
        rows = session.execute(
            select(Video.id)
            .join(Annotation, Annotation.video_id == Video.id)
            .order_by(Video.id)
        ).all()
    return [r.id for r in rows]


def _load_all_selected_ids() -> list[str]:
    with session_scope() as session:
        rows = session.execute(
            select(Video.id).where(Video.selected.is_(True)).order_by(Video.id)
        ).all()
    return [r.id for r in rows]


def _load_annotation(video_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        ann = session.get(Annotation, video_id)
        if ann is None:
            return None
        try:
            data: dict[str, Any] = json.loads(ann.payload_json)
        except json.JSONDecodeError:
            return None
        return data


def _load_video_row(video_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        v = session.get(Video, video_id)
        if v is None:
            return None
        return {
            "id": v.id,
            "source_dataset": v.source_dataset,
            "source_path": v.source_path,
            "duration_sec": v.duration_sec,
            "fps": v.fps,
            "resolution_w": v.resolution_w,
            "resolution_h": v.resolution_h,
            "frame_dir": v.frame_dir,
            "status": v.status,
            "selected": bool(v.selected),
            "select_reason": v.select_reason,
            "num_candidate_frames": v.num_candidate_frames,
            "candidate_fps": v.candidate_fps,
            "error": v.error,
            "updated_at": str(v.updated_at),
        }


def _aggregate_stats() -> dict[str, Any]:
    """Aggregate stats over all assembled annotations."""
    with session_scope() as session:
        payloads = session.execute(select(Annotation.payload_json)).scalars().all()
    parsed: list[dict[str, Any]] = []
    for p in payloads:
        try:
            parsed.append(json.loads(p))
        except json.JSONDecodeError:
            continue
    n = len(parsed)
    if n == 0:
        return {"count": 0}
    total_ke = sum(len(p.get("key_elements") or []) for p in parsed)
    total_qa = sum(len(p.get("qa_pairs") or []) for p in parsed)
    n_with_risks = sum(1 for p in parsed if p.get("risk_labels"))
    walk_counter: Counter[str] = Counter(p.get("walkability", "unknown") for p in parsed)
    split_counter: Counter[str] = Counter(p.get("split", "unknown") for p in parsed)
    return {
        "count": n,
        "avg_key_elements": round(total_ke / n, 2),
        "avg_qa_pairs": round(total_qa / n, 2),
        "with_risks": n_with_risks,
        "walkability": dict(walk_counter),
        "split": dict(split_counter),
    }


def _list_sampled_frames(video_row: dict[str, Any]) -> list[tuple[Path, float]]:
    """Return sampled frame paths + their center-of-bucket timestamps.

    Uses ``candidate_fps = num_candidate_frames / duration_sec`` from the
    row, so ordering + timestamps line up with what the model saw.
    """
    settings = get_settings()
    frame_dir = Path(video_row["frame_dir"]) if video_row["frame_dir"] else settings.paths.frames_dir / video_row["id"]
    frames = sorted(frame_dir.glob("f_*.jpg"))
    if not frames:
        return []
    # We store extraction fps in the model call chain but the DB only
    # keeps `candidate_fps` (effective). Reconstruct extraction fps as
    # len(frames) / duration; if duration is 0, fall back to 10 fps.
    duration = float(video_row.get("duration_sec") or 0.0)
    ext_fps = (len(frames) / duration) if duration > 0 else float(settings.frames.fps)
    ts = [(p, ((_frame_index(p) - 0.5) / ext_fps) if ext_fps > 0 else 0.0) for p in frames]
    # Uniformly downsample to at most 24 tiles for display responsiveness.
    if len(ts) > 24:
        step = len(ts) / 24
        picks = [ts[min(int(step * (k + 0.5)), len(ts) - 1)] for k in range(24)]
        return picks
    return ts


def _frame_index(path: Path) -> int:
    m = re.search(r"f_(\d+)", path.stem)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------- views


def _render_overview() -> None:
    st.header("Pipeline overview")
    counts = _load_status_counts()
    total_all = sum(v["selected"] + v["not_selected"] for v in counts.values())

    cols = st.columns(len(STATUS_ORDER))
    for col, st_name in zip(cols, STATUS_ORDER, strict=True):
        sub = counts.get(st_name, {"selected": 0, "not_selected": 0})
        col.metric(
            label=st_name,
            value=sub["selected"] + sub["not_selected"],
            delta=f"sel={sub['selected']}",
        )
    st.caption(f"Total video rows: {total_all}")

    st.subheader("Aggregate stats over assembled annotations")
    agg = _aggregate_stats()
    if agg.get("count", 0) == 0:
        st.info("No assembled annotations yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("assembled", agg["count"])
        c2.metric("avg key_elements", agg["avg_key_elements"])
        c3.metric("avg qa_pairs", agg["avg_qa_pairs"])
        c4.metric("with risks", agg["with_risks"])
        st.write("walkability distribution:", agg["walkability"])
        st.write("split distribution:", agg["split"])

    st.subheader("Failed videos")
    failed = _load_failed()
    if not failed:
        st.success("No failed videos.")
    else:
        st.dataframe(failed, use_container_width=True)

    st.subheader("Most recently updated")
    recent = _load_recent()
    if not recent:
        st.info("No videos ingested yet.")
    else:
        st.dataframe(recent, use_container_width=True)


def _render_review() -> None:
    st.header("Per-clip review")

    assembled_ids = _load_assembled_ids()
    all_selected = _load_all_selected_ids()
    options = assembled_ids or all_selected
    if not options:
        st.info("No videos to review yet. Ingest and curate first.")
        return
    default_idx = 0
    video_id = st.selectbox("video_id", options=options, index=default_idx)
    if video_id is None:
        return

    row = _load_video_row(video_id)
    if row is None:
        st.error(f"Video row for {video_id} not found.")
        return

    top = st.columns(4)
    top[0].metric("status", row["status"])
    top[1].metric("duration_sec", f"{row['duration_sec']:.2f}")
    top[2].metric("num_candidate_frames", row["num_candidate_frames"])
    top[3].metric("dataset", row["source_dataset"])
    st.caption(f"source_path: {row['source_path']}")
    if row.get("error"):
        st.error(f"error: {row['error']}")

    payload = _load_annotation(video_id)
    tab_json, tab_tables, tab_frames, tab_video = st.tabs(
        ["Assembled JSON", "Tables", "Sampled frames", "Source video"]
    )

    with tab_json:
        if payload is None:
            st.info("No assembled annotation for this video yet.")
        else:
            st.json(payload)

    with tab_tables:
        if payload is None:
            st.info("Assemble the annotation to see tables.")
        else:
            st.subheader("key_elements")
            ke_rows = [
                {
                    "label": k.get("label"),
                    "category": k.get("category"),
                    "importance": k.get("importance"),
                    "time_span": k.get("time_span"),
                    "position": k.get("position"),
                    "distance": k.get("distance"),
                    "motion": k.get("motion"),
                    "overhead": k.get("overhead"),
                }
                for k in (payload.get("key_elements") or [])
            ]
            st.dataframe(ke_rows or [{}], use_container_width=True)

            st.subheader("risk_labels")
            r_rows = [
                {
                    "type": r.get("type"),
                    "severity": r.get("severity"),
                    "time_span": r.get("time_span"),
                    "description": r.get("description"),
                }
                for r in (payload.get("risk_labels") or [])
            ]
            st.dataframe(r_rows or [{}], use_container_width=True)

            st.subheader("qa_pairs")
            q_rows = [
                {
                    "qid": q.get("qid"),
                    "type": q.get("type"),
                    "question": q.get("question"),
                    "answer": q.get("answer"),
                    "evidence_elements": q.get("evidence_elements"),
                    "evidence_time_span": q.get("evidence_time_span"),
                    "answer_type": q.get("answer_type"),
                }
                for q in (payload.get("qa_pairs") or [])
            ]
            st.dataframe(q_rows or [{}], use_container_width=True)

            st.subheader("caption")
            st.write(payload.get("caption", ""))
            st.caption(
                f"sampling_interval_sec = {payload.get('sampling_interval_sec')} "
                f"(time_spans are reliable within +/- this)"
            )

    with tab_frames:
        pairs = _list_sampled_frames(row)
        if not pairs:
            st.info("No extracted frames on disk for this video.")
        else:
            cols_per_row = 4
            for i in range(0, len(pairs), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, (path, ts) in zip(cols, pairs[i : i + cols_per_row], strict=False):
                    with col:
                        st.image(str(path), caption=f"t={ts:.2f}s", use_container_width=True)

    with tab_video:
        src = Path(row["source_path"])
        if src.exists():
            try:
                with src.open("rb") as fh:
                    st.video(fh.read())
            except OSError as e:
                st.warning(f"Could not read source video: {e}")
        else:
            st.info(f"Source video not found at {src}. This is expected on a review-only workstation.")


# --------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="egoannot dashboard", layout="wide")
    _boot_once()

    st.sidebar.title("egoannot")
    view = st.sidebar.radio("View", options=("Overview", "Per-clip review"), index=0)
    if st.sidebar.button("Refresh"):
        st.rerun()
    st.sidebar.caption(f"DB: {get_settings().paths.db_path}")

    if view == "Overview":
        _render_overview()
    else:
        _render_review()


# Streamlit calls the module top-level on `streamlit run`, so invoke main
# unconditionally when not imported.
if __name__ == "__main__" or True:  # noqa: SIM222 -- streamlit entrypoint pattern
    # Guard: only render if streamlit's runtime is present (i.e. we are
    # being executed by `streamlit run`).
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            main()
    except Exception:
        # Non-streamlit import (tests, mypy) — do nothing.
        pass


_ = TaskResult  # keep import for future task-drill-down; no runtime effect.
