"""Per-sub-task modules. Each wraps :func:`base.run_image_subtask` or
:func:`base.run_text_subtask` with the right response model and prompt params.
"""

from __future__ import annotations

from . import caption, entities, events, judgment, qa, scene

__all__ = ["caption", "entities", "events", "judgment", "qa", "scene"]
