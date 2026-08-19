"""Configuration loading and project-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML mapping and fail early for missing or malformed files."""

    resolved_path = config_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved_path}")
    with resolved_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved_path}")
    return config


def resolve_path(raw_path: str | Path, config_directory: Path) -> Path:
    """Resolve a configured path relative to the configuration file."""

    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (config_directory / path).resolve()
