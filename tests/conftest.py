"""Shared pytest fixtures.

Every test that touches the DB gets a fresh SQLite file under the test's
tmp_path, and the settings cache is reset so ``config.get_settings()``
picks up the temp paths without polluting other tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from egoannot.config import reset_settings_cache
from egoannot.db import dispose_engine, init_engine


@pytest.fixture()
def tmp_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolated data_dir + log_dir + fresh DB for one test."""
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("EGOANNOT_PATHS__ROOT", str(tmp_path))
    monkeypatch.setenv("EGOANNOT_PATHS__DATA_DIR", str(data_dir))
    monkeypatch.setenv("EGOANNOT_PATHS__LOG_DIR", str(log_dir))

    reset_settings_cache()
    dispose_engine()

    from egoannot.config import get_settings

    settings = get_settings()
    init_engine(settings.paths.db_path)

    yield tmp_path

    dispose_engine()
    reset_settings_cache()
