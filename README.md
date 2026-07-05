# egoannot

Batch pipeline that generates structured navigation annotations for egocentric
(first-person) video, using a locally served **Qwen3-VL-32B-Instruct** vLLM
server. Ingest raw clips from public datasets (JAAD, ADVIO, SCAND, NavWare),
curate to ~500 good clips, and produce one strictly-validated JSON annotation
per clip. The vision model answers narrow sub-questions; Python owns the final
schema, all identifiers, all timestamps, and all cross-references — so the
output is reliable at scale even when individual model calls fail.

The full technical design is in [`SPEC.md`](SPEC.md).

---

## Table of contents

- [What this produces](#what-this-produces)
- [Architecture overview](#architecture-overview)
- [How the model is served](#how-the-model-is-served)
- [Install](#install)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Datasets](#datasets)
- [Reference annotation](#reference-annotation)
- [Development](#development)
- [Known limitations](#known-limitations)
- [License](#license)

---

## What this produces

One JSON file per video, following a strict schema owned by Python. Fields:

- Identity + media: `video_id`, `split`, `video_path`, `frame_dir`,
  `duration_sec`, `fps`, `candidate_fps`, `num_candidate_frames`,
  `sampling_interval_sec`, `resolution`.
- Scene: `scene_category`, `environment` (location, lighting, weather,
  crowd_level, camera_motion).
- Content: `caption`, `key_elements[]`, `risk_labels[]`, `walkability`,
  `acceptable_actions[]`, `qa_pairs[]`.
- `privacy` flags.

Every time span is clamped to `[0, duration_sec]` and reliable within
`sampling_interval_sec`. See the [reference annotation](#reference-annotation)
at the bottom of this file.

## Architecture overview

**Six-layer understanding model.** For each clip, we solve six conceptual
layers:

1. Scene & localization — where and what kind of place.
2. Key entities — the meaningful people/objects.
3. Events — what happens over time.
4. Motion & trends — folded into entities + events.
5. Ego-motion — folded into the scene call.
6. Judgment — walkability, risks, recommended actions. Optional.

**Model answers sub-tasks; code owns the schema.** The vision model NEVER
emits the final annotation JSON. It answers small strictly-typed
sub-questions (`scene`, `entities`, `events`, `judgment`, `caption`, `qa`)
per segment. Every response is validated with a Pydantic v2 model. Python
merges validated sub-task rows into the final schema, generates all ids and
qids, clamps time spans, and validates the assembled dict before writing.
A failed sub-task never kills a video; assembly falls back to safe
defaults.

**Resumable, idempotent SQLite state.** Every stage keys off
`Video.status` in a SQLite DB (`data/pipeline.db`). Rows already past a
stage are skipped. `TaskResult` rows are unique by
`(video_id, segment_idx, task_name)`, so re-runs upsert in place. A run
that crashes mid-batch resumes from wherever it stopped.

**Pipeline order:**

```
ingest -> curate -> frames -> annotate -> assemble -> export
                                 |
                                 +-> per segment:
                                     [scene, entities, events, judgment]  (image, parallel)
                                     [caption, qa]                        (text-only, parallel)
```

## How the model is served

The pipeline is a thin async HTTP client. It does **not** launch the model.
Serve `Qwen3-VL-32B-Instruct` externally with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3-VL-32B-Instruct \
    --served-model-name Qwen3-VL-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --enforce-eager
```

**Environment constraints we hit** (record for reproducibility):

- vLLM `0.11.0`.
- PyTorch built for CUDA 12.6 (`torch --index-url https://download.pytorch.org/whl/cu126`).
- Host driver NVIDIA 535.129.03 exposes CUDA runtime `12.2`. Torch built
  against 12.6 remains binary-compatible thanks to CUDA minor-version
  compatibility, but keep torch and vllm within a compatible tuple. A
  different driver may need a different `cuXXX` wheel.
- 16384-token context ceiling. ~12 frames at 1280 long-side + a compact
  prompt uses ~11k prompt tokens, leaving ~5k for output. See
  `SPEC.md` for the budget analysis.

The pipeline reads the server URL from
`vlm.base_url` (default `http://localhost:8000/v1`). Override via
`EGOANNOT_VLM__BASE_URL`.

## Install

Requires Python 3.11 and `ffmpeg`/`ffprobe` on `PATH`.

```bash
# Preferred: uv (fast, deterministic)
uv pip install -e ".[dev]"
pre-commit install

# Or with plain pip
pip install -e ".[dev]"
```

## Configuration

Configuration is layered: **defaults < `config.yaml` < env vars (`EGOANNOT_*`) < `.env`**.

- Defaults for paths are RELATIVE to the current working directory
  (`./data`, `./logs`) so the code is portable.
- In production, override `paths.data_dir` and `paths.log_dir` to point
  at a data volume with several GB free (raw videos + extracted frames
  dominate).
- Copy `.env.example` to `.env` for local overrides.
- Every nested key uses `__` in env vars, e.g.
  `EGOANNOT_VLM__BASE_URL=http://localhost:8000/v1`.

`config-dump` prints the effective merged configuration as JSON:

```bash
egoannot config-dump
```

## Usage

Every command is idempotent and resumable. Common flags: `--mock`
(offline `MockVLMClient`), `--concurrency N`, `--video-id X`,
`--log-level {DEBUG,INFO,WARNING,ERROR}`.

### 1. `ingest` — register raw videos

```bash
egoannot ingest --dataset jaad --path /path/to/JAAD_root
```

Adapters know how to enumerate each dataset. Skips clips already
registered (matched by absolute source path).

### 2. `curate` — two-pass filter to ~500 kept clips

```bash
egoannot curate --target 500 --concurrency 4
```

Pass 1 is metadata-only (drops clips outside `[15, 90]s`, corrupt
metadata, tiny resolution). Pass 2 fires one cheap classifier call per
survivor asking "is this a forward-facing eye-level walking viewpoint?"
Kept videos are balanced across datasets via round-robin.

### 3. `frames` — extract candidate frames

```bash
egoannot frames --all
```

Extracts at `frames.fps` (default 10) into `data/frames/<video_id>/`,
downscales so the long side ≤ `frames.max_long_side` (default 1280),
segments the clip, and records per-segment metadata. Frames are also
extracted lazily by `annotate` if you skip this step.

### 4. `annotate` — run the six sub-tasks per segment

```bash
egoannot annotate --all --concurrency 4
# offline dry run:
egoannot annotate --all --mock
# skip the judgment sub-task:
egoannot annotate --all --no-risk
```

For each segment: image-bearing sub-tasks (`scene`, `entities`, `events`,
`judgment`) run in parallel; then text-only synthesis
(`caption`, `qa`) runs on the validated subset. Failed sub-tasks
persist with `ok=False` and assembly falls back to defaults.

### 5. `assemble` — deterministic final JSON

```bash
egoannot assemble --all
```

### 6. `export` — write per-video JSON files and combined JSONL

```bash
egoannot export --jsonl
```

### Convenience commands

- `egoannot run --all` chains `curate -> frames -> annotate -> assemble
  -> export`.
- `egoannot status` prints counts of `Video` rows by status.
- `egoannot retry-failed` resets `failed` rows back to `curated`.
- `egoannot dashboard` launches the Streamlit review UI (see below).

## Dashboard

A read-only Streamlit dashboard for monitoring a running batch and
reviewing per-clip annotations. It never modifies the DB or re-runs a
stage.

```bash
egoannot dashboard
# opens on http://127.0.0.1:8501
```

Two views:

- **Overview**: status counts per stage, aggregate stats over assembled
  annotations (avg key_elements, avg qa_pairs, walkability distribution),
  failed videos table, recently updated videos table. A **Refresh** button
  in the sidebar re-queries the DB for live monitoring.
- **Per-clip review**: pick a `video_id`, view the assembled JSON,
  readable tables for `key_elements`, `risk_labels`, `qa_pairs`, and the
  sampled frames as a labelled gallery so you can visually compare frames
  against the annotation. Embeds the source video when present locally.

**Remote viewing.** The dashboard binds to `127.0.0.1:8501` by default.
From your laptop:

```bash
ssh -L 8501:localhost:8501 <remote-host>
# then in a second terminal on the remote:
egoannot dashboard
# then open http://localhost:8501 in your local browser
```

## Datasets

**This repository ships no data.** You must obtain each dataset yourself,
under that dataset's own license.

| Dataset  | Viewpoint                    | License                                   | Adapter status                                        |
| -------- | ---------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| JAAD     | dashcam                      | CC BY 4.0 (attribution)                   | Real adapter (`JAADAdapter`); reference implementation. |
| ADVIO    | handheld walking egocentric  | **CC BY-NC 4.0 (non-commercial)**         | Real adapter (`ADVIOAdapter`).                        |
| EgoBlind | blind-user egocentric        | **CC BY-NC-SA 4.0 (non-commercial, share-alike)** | Registered via `GenericVideoFolderAdapter`. |
| SANPO    | walking egocentric           | see dataset terms                         | Register via `GenericVideoFolderAdapter` (glob).      |
| SCAND    | walking egocentric           | see dataset terms                         | Stub (`NotImplementedError`). Bespoke layout — write a dedicated adapter. |
| NavWare  | walking egocentric           | see dataset terms                         | Stub. Bespoke layout — write a dedicated adapter.     |

**JAAD is dashcam-only** and is included as a reference adapter for
plumbing the pipeline end-to-end. The walking-egocentric fits are
**ADVIO** (real handheld iPhone footage across malls, metro stations,
stairs, indoor and outdoor scenes — the primary content fit for this
pipeline) plus SCAND and NavWare once their adapters are implemented.

### ADVIO

Download from Zenodo yourself. Licensed **CC BY-NC 4.0** —
non-commercial only, stricter than JAAD's plain CC-BY. Downstream
artefacts inherit the restriction.

Expected on-disk layout:

```
<advio-root>/
    advio-01/
        iphone/frames.mov      # RGB, narrow-FOV, walking POV — USED
        iphone/frames.csv      # sensor sidecar — ignored
        tango/frames.mov       # fisheye Tango stream — IGNORED (wrong FOV)
        tango/frames.csv       # ignored
        ...
```

Only `iphone/frames.mov` per recording is registered; Tango fisheye and
CSVs are hard-ignored. Video ids derive deterministically from
`advio-NN`.

ADVIO recordings are minutes long, so they trigger the long-recording
chunking below by default.

### EgoBlind

Real blind-user egocentric footage — the best content fit for this
pipeline. Licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike):
non-commercial use only and every downstream artefact must be licensed
under the same terms. Check compatibility before running.

Download from the [official repository's Google Drive link](https://github.com/doc-doc/EgoBlind).
The repo ships no data — obtain videos yourself.

Expected on-disk layout: **any directory of `.mp4` files** (flat or
nested). The `GenericVideoFolderAdapter` recurses via `**/*.mp4`, so
after unzipping the archive to a folder just point `ingest` at it:

```bash
egoannot ingest --dataset egoblind --path /path/to/egoblind_videos
```

Video ids derive deterministically from `egoblind/<relative_path>`, so
moving the corpus root does not change ids.

### Adding a new dataset

Two options depending on layout:

- **Plain-video datasets** (a folder of `.mp4`, possibly nested):
  register with `GenericVideoFolderAdapter(dataset_name="mydataset",
  glob="**/*.mp4")`. No new code beyond registration.
- **Bespoke layouts** (paired streams, sidecar files, per-clip folders):
  implement `class DatasetAdapter(Protocol)` yielding
  `DiscoveredVideo(source_path, dataset_key, split_hint)`.

Register in `egoannot.ingest.ADAPTERS`.

### Long-recording chunking

Source recordings longer than `frames.chunk_sec` (default 40 s) are
split at ingest into consecutive chunks. Each chunk becomes its own
`Video` row with `chunk_start_sec` / `chunk_end_sec`; siblings share the
source file. Frame extraction seeks the chunk window directly, so no
copies are made. All timestamps and `time_span` values in the assembled
annotation are chunk-relative (t=0 at the chunk start).

Toggle with `frames.chunk_long_videos: false` if you want each
recording annotated as a single (potentially multi-segment) video.

## Reference annotation

```json
{
  "video_id": "VID_000001",
  "split": "train",
  "video_path": "videos/VID_000001.mp4",
  "frame_dir": "frames/VID_000001",
  "duration_sec": 32.4,
  "fps": 30,
  "candidate_fps": 2.0,
  "num_candidate_frames": 65,
  "sampling_interval_sec": 0.5,
  "resolution": [1920, 1080],
  "scene_category": ["outdoor_sidewalk", "dynamic_obstacle"],
  "environment": {
    "location_type": "outdoor", "lighting": "normal", "weather": "clear",
    "crowd_level": "low", "camera_motion": "walking"
  },
  "caption": "I am walking along a sidewalk in first-person view. A pedestrian is crossing ahead from the right; a parked car sits on the right shoulder. The path is mostly passable but I should slow down.",
  "key_elements": [
    {"label": "pedestrian", "category": "person", "importance": "high", "time_span": [8.0, 14.5], "position": "front", "distance": "near", "motion": "crossing", "overhead": false},
    {"label": "parked_car", "category": "vehicle", "importance": "medium", "time_span": [5.0, 20.0], "position": "right", "distance": "mid", "motion": "static", "overhead": false}
  ],
  "risk_labels": [
    {"type": "person_crossing", "severity": "medium", "time_span": [8.0, 14.5], "description": "A pedestrian is crossing in front; slow down and observe."}
  ],
  "walkability": "passable_with_caution",
  "acceptable_actions": ["slow_down", "keep_left", "observe"],
  "qa_pairs": [
    {"qid": "VID_000001_Q001", "type": "risk", "question": "Are there any dynamic obstacles ahead?", "answer": "Yes, a pedestrian is crossing ahead.", "evidence_elements": ["pedestrian"], "evidence_time_span": [8.0, 14.5], "answer_type": "open"},
    {"qid": "VID_000001_Q002", "type": "walkability", "question": "Can the walker proceed at normal speed?", "answer": "No; slow down and observe the crossing pedestrian.", "evidence_elements": ["pedestrian"], "evidence_time_span": [8.0, 14.5], "answer_type": "open"}
  ],
  "privacy": {"face_blurred": true, "plate_blurred": true, "contains_sensitive_info": false}
}
```

## Development

```bash
make lint       # ruff check
make fmt        # ruff --fix + black
make typecheck  # mypy (strict)
make test       # pytest
make check      # all three
```

The pre-commit hooks in `.pre-commit-config.yaml` run lint / format / mypy
on every commit. Enable with `pre-commit install`.

Test coverage:

- `tests/test_schemas.py` — Pydantic validation of every sub-task and the
  final annotation, plus enum coercion.
- `tests/test_frames.py` — frame math (segmentation, uniform sampling,
  center-of-bucket timestamps).
- `tests/test_assemble.py` — canned `TaskResult` rows -> schema-valid JSON,
  time-span clamping, defaults, deterministic split.
- `tests/test_orchestrator_mock.py` — full pipeline end-to-end via
  `MockVLMClient`.

## Known limitations

- **Time-span accuracy is bounded by the sampling interval.**
  `sampling_interval_sec = duration_sec / num_candidate_frames`. In our
  runs the model was consistently accurate to within ~1–2 sampling
  intervals on event boundaries, biased late on decay. The annotation
  carries this field explicitly so downstream consumers know the
  temporal resolution.
- **Attribute hallucination on VLM output.** The model occasionally
  invents plausible-but-unseen attributes (e.g. a "cane" that is not in
  frame). Enums are coerced/defaulted to keep the schema valid, but
  free-text descriptions still reflect the model's biases. Post-process
  or manually spot-check before shipping downstream.
- **Ego-motion misclassification on non-walking footage.** JAAD is
  dashcam; the model sometimes reports `camera_motion=walking` when the
  camera is vehicle-mounted. If you feed non-walking footage, add a
  clause to the scene prompt or override the field at assembly time.
- **Cross-segment merge is implemented but under-exercised on real
  footage.** For long (>45s) clips the assembler merges entities by
  label, unions actions, keeps the worst walkability, joins captions with
  `"; "`. Behaviour is deterministic but the merge policy has not been
  tuned against production data yet.

## License

The source code in this repository is **All Rights Reserved**. See
[`LICENSE`](LICENSE).

The datasets (JAAD, ADVIO, SCAND, NavWare) are governed by their own
licenses and must be obtained separately. Model weights (Qwen3-VL) are
governed by their own licenses. This repository does not redistribute
either.
