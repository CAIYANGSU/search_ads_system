"""Deterministic, model-agnostic feature engineering for unified click data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_ads_system.data.interfaces import FeatureConfig
from search_ads_system.data.storage import iter_csv_parts, prepare_output_directory, write_csv_part

LOGGER = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {
    "event_id",
    "click_timestamp",
    "conversion_label",
    "conversion_value_eur",
    "conversion_delay_seconds",
    "clicks_last_7d",
    "product_price",
}


@dataclass(frozen=True)
class FeatureResult:
    """Summary and column contract for a completed feature-generation run."""

    rows_written: int
    parts_written: int
    output_directory: Path
    feature_columns: tuple[str, ...]


def build_features(
    unified_data_directory: Path,
    output_directory: Path,
    metadata_path: Path,
    config: FeatureConfig,
    chunk_size: int,
    *,
    overwrite: bool = False,
) -> FeatureResult:
    """Build features in a streaming pass and persist a reproducible feature contract."""

    prepare_output_directory(output_directory, overwrite=overwrite)
    row_count = 0
    feature_columns: tuple[str, ...] | None = None
    parts_written = 0
    for part_number, chunk in enumerate(iter_csv_parts(unified_data_directory, chunk_size)):
        features = engineer_features(chunk, config)
        if feature_columns is None:
            feature_columns = tuple(features.columns)
        elif tuple(features.columns) != feature_columns:
            raise ValueError("Feature columns changed between chunks")
        write_csv_part(features, output_directory, part_number)
        row_count += len(features)
        parts_written += 1
    if feature_columns is None:
        raise ValueError("No unified data rows were available for feature generation")
    _write_metadata(metadata_path, feature_columns, config)
    return FeatureResult(row_count, parts_written, output_directory, feature_columns)


def engineer_features(chunk: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Transform a canonical chunk without fitting global or target-derived state."""

    missing = (_REQUIRED_COLUMNS | set(config.categorical_columns)) - set(chunk.columns)
    if missing:
        raise ValueError(f"Unified data is missing feature columns: {sorted(missing)}")

    features = chunk.loc[
        :, ["event_id", "conversion_label", "conversion_value_eur", "conversion_delay_seconds"]
    ].copy()
    timestamp = pd.to_numeric(chunk["click_timestamp"], errors="coerce")
    date_time = pd.to_datetime(timestamp, unit="s", utc=True, errors="coerce")
    features["click_hour_utc"] = date_time.dt.hour.astype("Int8")
    features["click_day_of_week_utc"] = date_time.dt.dayofweek.astype("Int8")
    features["click_timestamp_missing"] = timestamp.isna().astype("int8")

    for source_column, feature_name in (("product_price", "product_price"), ("clicks_last_7d", "clicks_last_7d")):
        values = pd.to_numeric(chunk[source_column], errors="coerce")
        features[f"{feature_name}_missing"] = values.isna().astype("int8")
        features[feature_name] = values.fillna(0.0).astype("float32")
        features[f"log1p_{feature_name}"] = np.log1p(values.clip(lower=0).fillna(0.0)).astype("float32")

    for column in config.categorical_columns:
        features[f"cat_{column}"] = (
            chunk[column].astype("string").fillna(config.missing_category_token).astype("string")
        )
    return features


def _write_metadata(path: Path, columns: tuple[str, ...], config: FeatureConfig) -> None:
    payload: dict[str, Any] = {
        "feature_version": "1.0",
        "feature_columns": list(columns),
        "label_columns": ["conversion_label", "conversion_value_eur", "conversion_delay_seconds"],
        "categorical_source_columns": list(config.categorical_columns),
        "missing_category_token": config.missing_category_token,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    LOGGER.info("Feature metadata written to %s", path)
