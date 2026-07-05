"""Runtime configuration.

Layered defaults:
    hard-coded defaults  <  config.yaml  <  environment variables (EGOANNOT_*)  <  .env

The YAML file is optional; if missing, defaults + env vars still yield a
complete settings object. Nested keys use `__` in env vars, e.g.
``EGOANNOT_VLM__BASE_URL`` overrides ``settings.vlm.base_url``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class PathsSettings(BaseModel):
    # Defaults are RELATIVE to the current working directory so the code is
    # portable. In production, override via config.yaml or EGOANNOT_PATHS__*
    # env vars to point at a large data volume (e.g. /data/keyframe on a
    # multi-GB filesystem).
    root: Path = Path(".")
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    raw_subdir: str = "raw"
    videos_subdir: str = "videos"
    frames_subdir: str = "frames"
    outputs_subdir: str = "outputs"
    db_filename: str = "pipeline.db"

    @field_validator("root", "data_dir", "log_dir")
    @classmethod
    def _must_live_under_data(cls, value: Path) -> Path:
        # Guard: the deployed host has ~3 GB free on `/`; batch artifacts must
        # never spill outside safe roots. We allow `/data/` (prod), `/tmp/`
        # and `/private/tmp/` (macOS test tempdirs), `/var/`, and relative
        # paths. Anything else is rejected.
        resolved = value.expanduser()
        if not resolved.is_absolute():
            return resolved
        s = str(resolved)
        safe_prefixes = ("/data/", "/tmp/", "/private/tmp/", "/var/", "/private/var/")
        if not s.startswith(safe_prefixes):
            raise ValueError(
                f"path {resolved} must live under one of {safe_prefixes}"
            )
        return resolved

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / self.raw_subdir

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / self.videos_subdir

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / self.frames_subdir

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / self.outputs_subdir


class VLMSettings(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen3-VL-32B-Instruct"
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    max_retries: int = 2
    temperature: float = 0.1
    top_p: float = 0.9
    max_output_tokens: int = 768


class RuntimeSettings(BaseModel):
    concurrency: int = 4
    log_level: str = "INFO"
    skip_risk: bool = False
    use_mock: bool = False


class CurateSettings(BaseModel):
    min_sec: float = 15.0
    max_sec: float = 90.0
    target_count: int = 500
    metadata_pass_concurrency: int = 8
    model_pass_concurrency: int = 4
    classifier_frames: int = 3


class FramesSettings(BaseModel):
    fps: int = 10
    max_long_side: int = 1280
    jpeg_quality: int = 80
    segment_max_sec: float = 45.0
    segment_len_sec: float = 20.0
    per_segment: int = 12


class TasksSettings(BaseModel):
    max_entities: int = 8
    max_events: int = 6
    num_qa: int = 5


class PrivacyFlags(BaseModel):
    face_blurred: bool = False
    plate_blurred: bool = False
    contains_sensitive_info: bool = False


class PrivacySettings(BaseModel):
    jaad: PrivacyFlags = Field(default_factory=lambda: PrivacyFlags(face_blurred=True, plate_blurred=True))
    advio: PrivacyFlags = Field(default_factory=PrivacyFlags)
    scand: PrivacyFlags = Field(default_factory=lambda: PrivacyFlags(face_blurred=True, plate_blurred=True))
    navware: PrivacyFlags = Field(default_factory=PrivacyFlags)

    def for_dataset(self, dataset: str) -> PrivacyFlags:
        key = dataset.lower()
        return getattr(self, key, PrivacyFlags())


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a top-level mapping")
    return data


class Settings(BaseSettings):
    """Top-level settings container. Access via :func:`get_settings`."""

    paths: PathsSettings = Field(default_factory=PathsSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    curate: CurateSettings = Field(default_factory=CurateSettings)
    frames: FramesSettings = Field(default_factory=FramesSettings)
    tasks: TasksSettings = Field(default_factory=TasksSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)

    model_config = SettingsConfigDict(
        env_prefix="EGOANNOT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence (highest first): init kwargs > env > dotenv > yaml > defaults.
        yaml_source = _YamlConfigSource(settings_cls)
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)


class _YamlConfigSource(PydanticBaseSettingsSource):
    """Loads config.yaml from the current working directory."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = _load_yaml(Path("config.yaml"))

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cache. Used by tests that mutate env vars."""
    get_settings.cache_clear()
