"""Preprocessing orchestration without modeling or feature transformation."""

from __future__ import annotations

from typing import Any

from search_ads_system.data.interfaces import PreprocessConfig
from search_ads_system.data.schema import inspect_schema


def run_preprocessing(config: PreprocessConfig) -> dict[str, Any]:
    """Run the currently supported preprocessing step: schema inspection."""

    return inspect_schema(config.dataset)
