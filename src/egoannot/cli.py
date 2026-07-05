"""Typer command-line interface.

Every stage is idempotent and resumable. Common flags:
    --concurrency N   asyncio semaphore for VLM calls
    --mock            use MockVLMClient (no network / GPU)
    --video-id X      operate on one row instead of the full corpus

The stage order under ``run --all``:
    ingest (skipped inside run) -> curate -> frames -> annotate -> assemble -> export
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import cast

import structlog
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from .assemble import AssembleError, MergeClient, MergeClientFactory, assemble_video
from .config import get_settings
from .curate import curate_all
from .db import Segment, Video, init_engine, session_scope
from .export import export_all
from .ingest import ingest_dataset
from .logging import configure_logging
from .media.frames import FrameExtractionError, extract_frames, probe_video, segment_video
from .orchestrator import annotate_video
from .schemas.enums import VideoStatus
from .vlm.client import VLMClient
from .vlm.mock import MockVLMClient

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Batch pipeline that annotates egocentric navigation videos with a "
        "locally served Qwen3-VL model. Every stage is idempotent and "
        "resumable via the SQLite pipeline.db under data/."
    ),
)
console = Console()
_log = structlog.stdlib.get_logger(__name__)


def _boot(log_level: str | None = None) -> None:
    """Common startup: configure logging, ensure paths, open DB."""
    settings = get_settings()
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        level=log_level or settings.runtime.log_level,
        log_dir=settings.paths.log_dir,
        run_id=uuid.uuid4().hex[:8],
    )
    init_engine(settings.paths.db_path)


def _make_client(use_mock: bool):  # type: ignore[no-untyped-def]
    return MockVLMClient() if use_mock else VLMClient()


def _make_factory(use_mock: bool) -> MergeClientFactory:
    """Return a fresh-client factory whose static type widens to the
    ``MergeClient`` protocol; both concrete clients structurally match it.
    """
    if use_mock:
        def _mock_factory() -> MergeClient:
            return MockVLMClient()
        return _mock_factory

    def _real_factory() -> MergeClient:
        return VLMClient()
    return _real_factory


# --------------------------------------------------------------------- ingest


@app.command()
def ingest(
    dataset: str = typer.Option(..., "--dataset", help="jaad|advio|scand|navware"),
    path: Path = typer.Option(..., "--path", help="Root directory containing the dataset"),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Register raw videos from a dataset root."""
    _boot(log_level)
    inserted = ingest_dataset(dataset, path)
    console.print(f"[green]ingest[/green]: dataset={dataset} inserted={inserted}")


# --------------------------------------------------------------------- curate


@app.command()
def curate(
    target: int = typer.Option(0, "--target", help="Target kept count (0 = config default)"),
    concurrency: int = typer.Option(0, "--concurrency"),
    mock: bool = typer.Option(False, "--mock"),
    force: bool = typer.Option(False, "--force", help="Re-evaluate already-curated videos"),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Two-pass curate to ~target-count kept videos."""
    _boot(log_level)
    settings = get_settings()
    tgt = target or settings.curate.target_count
    conc = concurrency or settings.curate.model_pass_concurrency

    async def _run() -> tuple[int, int]:
        client = _make_client(mock)
        try:
            return await curate_all(client=client, target_count=tgt, concurrency=conc, force=force)
        finally:
            await client.aclose()

    kept, dropped = asyncio.run(_run())
    console.print(f"[green]curate[/green]: kept={kept} dropped={dropped} target={tgt}")


# --------------------------------------------------------------------- frames


@app.command()
def frames(
    video_id: str | None = typer.Option(None, "--video-id"),
    all_: bool = typer.Option(False, "--all", help="Process every curated video"),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Extract candidate frames and persist per-video metadata + segments.

    Frames are also extracted lazily by ``annotate``; running this stage
    ahead of time is optional but useful for verifying disk I/O before
    firing off long GPU work.
    """
    _boot(log_level)
    settings = get_settings()
    ids = _resolve_ids(video_id, all_, require_curated=True)
    if not ids:
        console.print("[yellow]frames[/yellow]: no matching videos.")
        return
    done = 0
    for vid in ids:
        with session_scope() as session:
            v = session.get(Video, vid)
            if v is None:
                continue
            source = Path(v.source_path)
        try:
            meta = probe_video(source)
            frame_dir = settings.paths.frames_dir / vid
            frames_paths = extract_frames(source, frame_dir, meta)
            segs = segment_video(meta.duration_sec)
            with session_scope() as session:
                v = session.get(Video, vid)
                if v is None:
                    continue
                v.duration_sec = meta.duration_sec
                v.fps = meta.fps
                v.resolution_w = meta.width
                v.resolution_h = meta.height
                v.frame_dir = str(frame_dir)
                total = settings.frames.per_segment * len(segs)
                v.num_candidate_frames = total
                v.candidate_fps = total / meta.duration_sec if meta.duration_sec > 0 else 0.0
                # Replace segments in place.
                existing = session.execute(
                    select(Segment).where(Segment.video_id == vid)
                ).scalars().all()
                for s in existing:
                    session.delete(s)
                session.flush()
                for m in segs:
                    session.add(
                        Segment(
                            video_id=vid,
                            idx=m.idx,
                            start_sec=m.start_sec,
                            end_sec=m.end_sec,
                        )
                    )
                v.status = VideoStatus.frames_done.value
            done += 1
            console.print(f"[green]frames[/green] {vid}: {len(frames_paths)} frames, {len(segs)} segs")
        except FrameExtractionError as e:
            console.print(f"[red]frames FAIL[/red] {vid}: {e}")
            with session_scope() as session:
                v = session.get(Video, vid)
                if v is not None:
                    v.status = VideoStatus.failed.value
                    v.error = f"frames: {e}"
    console.print(f"[green]frames[/green]: done={done}/{len(ids)}")


# --------------------------------------------------------------------- annotate


@app.command()
def annotate(
    video_id: str | None = typer.Option(None, "--video-id"),
    all_: bool = typer.Option(False, "--all"),
    no_risk: bool = typer.Option(False, "--no-risk", help="Skip the judgment sub-task"),
    concurrency: int = typer.Option(0, "--concurrency"),
    mock: bool = typer.Option(False, "--mock"),
    force: bool = typer.Option(False, "--force"),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Run the six sub-tasks per segment for each selected video."""
    _boot(log_level)
    settings = get_settings()
    conc = concurrency or settings.runtime.concurrency
    ids = _resolve_ids(video_id, all_, require_curated=True)
    if not ids:
        console.print("[yellow]annotate[/yellow]: no matching videos.")
        return

    async def _run_all() -> None:
        sem = asyncio.Semaphore(conc)
        client = _make_client(mock)
        try:
            async def _one(vid: str) -> None:
                async with sem:
                    status = await annotate_video(
                        vid, client=client, skip_risk=no_risk, force=force
                    )
                    console.print(f"[green]annotate[/green] {vid}: status={status}")
            await asyncio.gather(*[_one(v) for v in ids])
        finally:
            await client.aclose()

    asyncio.run(_run_all())


# --------------------------------------------------------------------- assemble


@app.command()
def assemble(
    video_id: str | None = typer.Option(None, "--video-id"),
    all_: bool = typer.Option(False, "--all"),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use MockVLMClient for the multi-segment caption-merge call.",
    ),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Assemble final annotations from validated sub-task rows."""
    _boot(log_level)
    ids = _resolve_ids(video_id, all_, require_status_at_least="tasks_done")
    if not ids:
        console.print("[yellow]assemble[/yellow]: no matching videos.")
        return
    # A fresh client is created per merge call inside assemble_video.
    # Single-segment videos never touch the factory.
    factory = _make_factory(mock)
    for vid in ids:
        try:
            payload = assemble_video(vid, merge_client_factory=factory)
            console.print(
                f"[green]assemble[/green] {vid}: qa={len(payload['qa_pairs'])} "
                f"key_elements={len(payload['key_elements'])} "
                f"walkability={payload['walkability']}"
            )
        except AssembleError as e:
            console.print(f"[red]assemble FAIL[/red] {vid}: {e}")


# --------------------------------------------------------------------- export


@app.command()
def export(
    jsonl: bool = typer.Option(False, "--jsonl", help="Also write outputs/all.jsonl"),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Write per-video annotation JSON files (and optionally all.jsonl)."""
    _boot(log_level)
    per_video, jsonl_path = export_all(jsonl=jsonl)
    console.print(f"[green]export[/green]: wrote {len(per_video)} json files")
    if jsonl_path is not None:
        console.print(f"[green]export[/green]: jsonl={jsonl_path}")


# --------------------------------------------------------------------- run --all


@app.command("run")
def run_all(
    all_: bool = typer.Option(False, "--all", help="Chain curate -> frames -> annotate -> assemble -> export"),
    no_risk: bool = typer.Option(False, "--no-risk"),
    concurrency: int = typer.Option(0, "--concurrency"),
    mock: bool = typer.Option(False, "--mock"),
    jsonl: bool = typer.Option(True, "--jsonl/--no-jsonl", help="Emit outputs/all.jsonl at the end"),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Chain all stages. Must be invoked with ``--all`` to guard against typos."""
    if not all_:
        console.print("[red]run[/red]: pass --all to chain the full pipeline")
        raise typer.Exit(2)
    _boot(log_level)
    settings = get_settings()
    conc = concurrency or settings.runtime.concurrency

    async def _pipeline() -> None:
        client = _make_client(mock)
        try:
            kept, dropped = await curate_all(client=client, concurrency=conc)
            console.print(f"[cyan]curate[/cyan]: kept={kept} dropped={dropped}")
            # frames + annotate for every kept, curated video.
            ids = _resolve_ids(None, True, require_curated=True)
            sem = asyncio.Semaphore(conc)

            async def _one(vid: str) -> None:
                async with sem:
                    status = await annotate_video(
                        vid, client=client, skip_risk=no_risk
                    )
                    console.print(f"[cyan]annotate[/cyan] {vid}: {status}")
            await asyncio.gather(*[_one(v) for v in ids])
        finally:
            await client.aclose()

    asyncio.run(_pipeline())

    # assemble + export (sync). Multi-segment videos will spin up a
    # short-lived merge client per video; single-segment videos skip it.
    factory = _make_factory(mock)
    ids = _resolve_ids(None, True, require_status_at_least="tasks_done")
    for vid in ids:
        try:
            assemble_video(vid, merge_client_factory=factory)
        except AssembleError as e:
            console.print(f"[red]assemble FAIL[/red] {vid}: {e}")
    export_all(jsonl=jsonl)
    console.print("[green]run --all[/green]: pipeline complete.")


# --------------------------------------------------------------------- status


@app.command()
def status(
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Print counts of Video rows by status + selection."""
    _boot(log_level)
    with session_scope() as session:
        rows = session.execute(
            select(Video.status, Video.selected, func.count(Video.id))
            .group_by(Video.status, Video.selected)
            .order_by(Video.status)
        ).all()
    table = Table(title="pipeline status")
    table.add_column("status")
    table.add_column("selected")
    table.add_column("count", justify="right")
    total = 0
    for st, sel, count in rows:
        table.add_row(str(st), "yes" if sel else "no", str(count))
        total += int(count)
    table.add_section()
    table.add_row("total", "", str(total))
    console.print(table)


# --------------------------------------------------------------------- retry-failed


@app.command("retry-failed")
def retry_failed(
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Reset every ``failed`` video back to ``curated`` so annotate re-runs it."""
    _boot(log_level)
    with session_scope() as session:
        rows = session.execute(
            select(Video).where(Video.status == VideoStatus.failed.value)
        ).scalars().all()
        reset = 0
        for v in rows:
            v.status = VideoStatus.curated.value
            v.error = None
            reset += 1
    console.print(f"[green]retry-failed[/green]: reset {reset} videos to 'curated'.")


# --------------------------------------------------------------------- dashboard


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8501, "--port"),
    headless: bool = typer.Option(True, "--headless/--no-headless"),
) -> None:
    """Launch the Streamlit review + monitoring dashboard.

    View remotely via SSH tunnel:
        ssh -L 8501:localhost:8501 <host>
        open http://localhost:8501
    """
    import subprocess
    import sys

    from . import dashboard as _dash_mod

    module_file = Path(_dash_mod.__file__)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(module_file),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    console.print(f"[green]dashboard[/green]: launching {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


# --------------------------------------------------------------------- helpers


def _resolve_ids(
    video_id: str | None,
    all_flag: bool,
    *,
    require_curated: bool = False,
    require_status_at_least: str | None = None,
) -> list[str]:
    """Turn --video-id / --all into a concrete list of ids."""
    with session_scope() as session:
        if video_id is not None:
            v = session.get(Video, video_id)
            return [v.id] if v is not None else []
        if not all_flag:
            return []
        stmt = select(Video.id).order_by(Video.id)
        if require_curated:
            stmt = stmt.where(Video.selected.is_(True))
        if require_status_at_least is not None:
            order = {
                VideoStatus.pending.value: 0,
                VideoStatus.curated.value: 1,
                VideoStatus.frames_done.value: 2,
                VideoStatus.tasks_done.value: 3,
                VideoStatus.assembled.value: 4,
                VideoStatus.failed.value: -1,
            }
            need = order.get(require_status_at_least, 0)
            # Fetch statuses and filter in Python for readability.
            rows = session.execute(
                select(Video.id, Video.status)
                .where(Video.selected.is_(True))
                .order_by(Video.id)
            ).all()
            return [r.id for r in rows if order.get(r.status, -1) >= need]
        return list(session.execute(stmt).scalars().all())


# Utility: dump the effective config in JSON. Handy for debugging.
@app.command("config-dump")
def config_dump() -> None:
    """Print the effective merged configuration as JSON."""
    settings = get_settings()
    console.print_json(json.dumps(settings.model_dump(mode="json"), default=str))


if __name__ == "__main__":
    app()
