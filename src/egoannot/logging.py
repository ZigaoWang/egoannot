"""Structured logging.

structlog is configured to emit JSON lines. Each pipeline unit binds
``run_id``, ``video_id``, and ``task_name`` into the logger context so log
lines can be filtered/grepped without reparsing free-form messages.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def _timestamper(_: Any, __: str, event_dict: EventDict) -> EventDict:
    event_dict["ts"] = datetime.now(tz=UTC).isoformat(timespec="milliseconds")
    return event_dict


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Configure stdlib logging + structlog once at program start.

    Console handler emits human-readable rendering; a rotating file handler
    (if ``log_dir`` given) emits JSON lines suitable for `jq`.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(log_level)
    handlers.append(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = f"egoannot-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}.jsonl"
        file_handler = logging.FileHandler(log_dir / fname, encoding="utf-8")
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(log_level)

    shared_processors: list[Processor] = [
        _timestamper,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Console: human-readable; file handler picks up the same processors via
    # ProcessorFormatter and renders as JSON.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    console_fmt = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )
    json_fmt = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    console.setFormatter(console_fmt)
    for h in handlers[1:]:
        h.setFormatter(json_fmt)

    if run_id is not None:
        structlog.contextvars.bind_contextvars(run_id=run_id)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Callers pass ``__name__``."""
    return structlog.stdlib.get_logger(name)


def bind(**kwargs: Any) -> None:
    """Bind key/value pairs into the ambient logger context."""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind(*keys: str) -> None:
    """Remove keys from the ambient logger context."""
    structlog.contextvars.unbind_contextvars(*keys)
