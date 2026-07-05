# Egocentric Video Annotation Pipeline — Technical Spec

## Goal
Ingest first-person (egocentric) navigation videos from public datasets (JAAD,
ADVIO, SCAND, NavWare), curate down to ~500 high-quality clips, and generate one
structured JSON annotation per clip using a locally served Qwen3-VL-32B-Instruct
model. The model answers narrow, strictly-typed sub-questions; Python validates
every response, persists it, and deterministically assembles the final annotation.
All state is persisted in SQLite so the job is fully resumable and idempotent.

The reference JSON at the end of this document is the target shape to follow. It is
a well-motivated standard, not a frozen contract: additional justified fields are
acceptable; omitting required fields is not.

## Core principle
The vision model NEVER emits the final annotation JSON. It only answers small
sub-tasks, each returning a minimal, easily-validated JSON object. Python owns the
final schema, all identifiers, all timestamps, all cross-references, and all
normalization. This separation is what makes the output reliable at scale.

## Language policy
All output — code, code comments, log lines, config, CLI help, prompts, prompt
templates, and natural-language DATA VALUES (captions, entity labels, event
descriptions, questions, answers, risk descriptions) — is written in ENGLISH.
Categorical values use the English enum tokens defined below.

## Understanding model (the six layers that define "understanding a video")
1. Scene & localization — where this is, what kind of place, conditions.
2. Key entities — the meaningful people/objects present.
3. Events — what happens over time (the layer generic captioners miss).
4. Motion & trends — what moves/changes and in what direction (folded into
   entities + events as per-item fields, not a separate call).
5. Ego-motion — what the camera/wearer does (folded into the scene call).
6. Judgment & meaning — anything noteworthy/abnormal/requiring a response;
   walkability and risk live here. Optional (skippable).

## Deployed environment (already provisioned — build against this exactly)
- Inference server: vLLM 0.11.0, OpenAI-compatible, ALREADY RUNNING.
  - Base URL: `http://localhost:8000/v1`
  - Model id: `Qwen3-VL-32B-Instruct`
  - Launched with `--max-model-len 16384 --enforce-eager`.
  - Local frames are sent as base64 JPEG data URIs inside OpenAI `image_url`
    content blocks. Never file paths, never remote URLs.
- Runtime: conda env `qwen`, Python 3.11. Pipeline is a thin HTTP client of the
  server; no torch/transformers/vllm dependency.
- Filesystem: the pipeline reads/writes exclusively under
  `paths.data_dir` (default `./data`). Point that at a volume with
  several GB free before running a real batch. Nothing is written outside
  the configured `data_dir` / `log_dir`.

## Technology stack
- Packaging: `pyproject.toml`, managed with `uv`. Pinned deps.
- Python 3.11. CLI: Typer. Config: pydantic-settings. Validation: Pydantic v2.
- Persistence: SQLAlchemy 2.0 (typed, declarative) over SQLite.
- HTTP: httpx (async) + tenacity for retry/backoff.
- Media: ffmpeg/ffprobe via subprocess; Pillow for resize + JPEG encoding.
- Logging: structlog, JSON-line output.
- Concurrency: asyncio with a bounded semaphore.
- Quality: ruff + black + mypy + pytest, enforced via pre-commit and Makefile.

## Persistence model
- `Video`: id (VID_%06d), source_dataset, source_path, selected, select_reason,
  split, duration_sec, fps, resolution_{w,h}, frame_dir, candidate_fps
  (float; effective num_candidate_frames/duration), num_candidate_frames
  (total frames sent to the model across all segments), status
  (pending|curated|frames_done|tasks_done|assembled|failed), error, timestamps.
- `Segment`: (video_id, idx, start_sec, end_sec).
- `TaskResult`: unique (video_id, segment_idx, task_name); raw_response,
  parsed_json, ok, attempts.
- `Annotation`: (video_id, payload_json).

## Controlled vocabularies (English tokens, enforced in code)
- category: person, vehicle, bicycle, animal, obstacle, structure, sign,
  traffic_light, other
- importance: high, medium, low
- position: front, front_left, front_right, left, right, rear
- distance: near, mid, far, unknown
- motion: static, approaching, receding, crossing, unknown
- severity: high, medium, low
- risk_type: person_crossing, vehicle_approaching, obstacle_ahead, path_blocked,
  surface_change, overhead_obstacle, other
- qa_type: scene, entity, event, motion, risk, walkability
- walkability: passable, passable_with_caution, not_passable, unknown
- action: slow_down, stop, keep_left, keep_right, observe, wait, proceed, detour
- location_type: outdoor, indoor, unknown
- lighting: normal, dim, bright, unknown
- weather: clear, rain, snow, overcast, unknown
- crowd_level: low, medium, high
- camera_motion: walking, standing, turning, mixed, unknown
- answer_type: open, boolean, choice

On enum mismatch, code coerces to the nearest valid token or a safe default and
logs the coercion; it does not fail the video.

## Curate stage
1. Metadata pass (no model): keep only clips whose duration is inside
   `[curate.min_sec, curate.max_sec]` (default 15-90) with sane resolution and
   ffprobe-readable metadata.
2. Model pass (survivors only): sample 3-4 frames, issue ONE cheap classifier
   query — is this a forward-facing eye-level walking/navigation viewpoint?
   Response: `{ "keep": bool, "viewpoint": str, "reason": str }`.
3. Selection: take up to `curate.target_count` (default 500), balanced across
   datasets via round-robin. Mark `selected=true`, transition to `curated`.

## Frame handling
1. ffprobe -> VideoMeta persisted on the Video row.
2. Extract at `frames.fps` (default 10) to `frames/<video_id>/f_%05d.jpg`,
   downscaling so the long side <= `frames.max_long_side` (default 1280).
3. Segmentation: `duration <= frames.segment_max_sec` -> one segment; else
   non-overlapping windows of `frames.segment_len_sec` (default 20).
4. Per segment, uniformly sample `frames.per_segment` (default 12) frames; each
   frame's timestamp is `(i - 0.5) / extraction_fps` (centre-of-bucket).
5. For model calls, re-encode frames as base64 JPEG data URIs at
   `frames.jpeg_quality` (default 80). Precede each with a `[t=X.Xs]` text marker.

## Sub-tasks
Each returns a strict Pydantic-validated JSON object. Prompts instruct the model
(in English) to return only one JSON object with no prose and no markdown fences.

Image-bearing (send frames): `scene`, `entities`, `events`, `judgment` (optional).
Text-only synthesis (consume validated prior results as compact JSON, no images):
`caption`, `qa`.

Per-video failure conditions: frame extraction fails, or every image sub-task
fails across every segment. Otherwise assembly falls back to defaults for any
missing sub-task.

## Assembly (Python owns the schema)
- Identity/media fields from the Video row.
- environment/scene_category from scene.
- key_elements from entities, merged by label across segments (widest time_span,
  max importance).
- caption: per-segment captions joined with "; ".
- risk_labels, walkability (worst), acceptable_actions (union) from judgment.
- qa_pairs from qa; qids generated deterministically `{video_id}_Q{n:03d}`.
- privacy from per-dataset config flags.
- split: deterministic sha1(video_id) 80/10/10 unless dataset provides one.
- sampling_interval_sec = duration_sec / num_candidate_frames (time resolution
  the annotation is valid within).
- All time_spans are clamped to `[0, duration_sec]`.

## CLI (Typer) — every command idempotent and resumable
- `ingest --dataset <name> --path <dir>`
- `curate [--target 500] [--concurrency N] [--mock] [--force]`
- `frames [--video-id X | --all]`
- `annotate [--video-id X | --all] [--no-risk] [--concurrency N] [--mock] [--force]`
- `assemble [--video-id X | --all]`
- `export [--jsonl]`
- `run --all [--no-risk] [--concurrency N] [--mock] [--jsonl/--no-jsonl]`
- `status`
- `retry-failed`
- `config-dump`

## Concurrency & robustness
- Bounded asyncio semaphore, `runtime.concurrency` videos in flight (default 4).
- Per HTTP call: `vlm.timeout_sec` timeout, <=`vlm.max_retries` retries on 5xx /
  timeout (tenacity, exponential backoff). One additional retry on validation
  failure with a corrective user message; images stay attached on the retry.
- A failed sub-task never kills a video (ok=False persisted; assembly defaults).
- Raw model responses are persisted for debugging.

## Testing & quality
- MockVLMClient returns deterministic valid JSON so the full pipeline runs
  end-to-end without the server (`--mock`).
- pytest suite: schema validation (valid + invalid), assembly, frame math,
  mock end-to-end orchestrator run.
- ruff + black + mypy clean; enforced via pre-commit and the Makefile.

## Out of scope
No fine-tuning, no web UI, no Postgres, no migration framework, no
torch/transformers/vllm dependency. The model never emits the final schema.
Nothing is written outside `/data/`.

## Reference final annotation (target shape; standard, not frozen)

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
