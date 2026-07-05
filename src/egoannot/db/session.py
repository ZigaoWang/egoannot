"""SQLite engine + session factory.

One process-wide engine; sessions are short-lived and managed via a
context manager. SQLite pragmas (WAL, foreign keys) are set on connect
so writers and readers can coexist during a run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

_lock = RLock()
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _apply_pragmas(dbapi_connection: Any, _: Any) -> None:
    cur = dbapi_connection.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
    finally:
        cur.close()


def init_engine(db_path: Path | None = None, *, echo: bool = False) -> Engine:
    """Create the process-wide engine and schema. Idempotent."""
    global _engine, _SessionLocal

    with _lock:
        if _engine is not None:
            return _engine

        if db_path is None:
            db_path = get_settings().paths.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"sqlite:///{db_path}"
        engine = create_engine(
            url,
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        event.listen(engine, "connect", _apply_pragmas)

        Base.metadata.create_all(engine)

        _engine = engine
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        return engine


def get_engine() -> Engine:
    if _engine is None:
        return init_engine()
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a transactional session; commit on success, rollback on error."""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Close and clear the engine. Used by tests between runs."""
    global _engine, _SessionLocal
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionLocal = None


__all__ = ["dispose_engine", "get_engine", "init_engine", "session_scope"]
