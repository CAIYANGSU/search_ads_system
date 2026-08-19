"""Streaming exploratory data analysis for canonical click-conversion data."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from search_ads_system.data.interfaces import EdaConfig
from search_ads_system.data.storage import iter_csv_parts

LOGGER = logging.getLogger(__name__)

_NUMERIC_COLUMNS = (
    "click_timestamp",
    "conversion_value_eur",
    "conversion_delay_seconds",
    "clicks_last_7d",
    "product_price",
)


@dataclass
class _NumericAccumulator:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return
        self.count += len(numeric)
        self.total += float(numeric.sum())
        minimum = float(numeric.min())
        maximum = float(numeric.max())
        self.minimum = minimum if self.minimum is None else min(self.minimum, minimum)
        self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "non_null_count": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
        }


def run_eda(
    unified_data_directory: Path,
    summary_path: Path,
    categories_path: Path,
    config: EdaConfig,
    chunk_size: int,
) -> dict[str, Any]:
    """Compute data-quality, label, numeric, and selected-category summaries."""

    numeric = {column: _NumericAccumulator() for column in _NUMERIC_COLUMNS}
    missing_counts: Counter[str] = Counter()
    category_counts = {column: Counter[str]() for column in config.categorical_columns}
    row_count = 0
    conversion_count = 0

    for chunk in iter_csv_parts(unified_data_directory, chunk_size):
        _validate_columns(chunk, set(_NUMERIC_COLUMNS) | {"conversion_label"} | set(config.categorical_columns))
        row_count += len(chunk)
        conversion_count += int(pd.to_numeric(chunk["conversion_label"], errors="raise").sum())
        for column in chunk.columns:
            missing_counts.setdefault(column, 0)
            missing_counts[column] += int(chunk[column].isna().sum())
        for column, accumulator in numeric.items():
            accumulator.add(chunk[column])
        for column, counts in category_counts.items():
            values = chunk[column].astype("string").fillna("__MISSING__")
            counts.update(values.value_counts().to_dict())

    summary = {
        "row_count": row_count,
        "conversion_count": conversion_count,
        "conversion_rate": conversion_count / row_count if row_count else 0.0,
        "missingness": {
            column: {
                "missing_count": count,
                "missing_rate": count / row_count if row_count else 0.0,
            }
            for column, count in sorted(missing_counts.items())
        },
        "numeric_statistics": {column: accumulator.to_dict() for column, accumulator in numeric.items()},
    }
    _write_json(summary, summary_path)
    _write_category_counts(category_counts, categories_path, config.top_k)
    LOGGER.info("EDA summary written to %s", summary_path)
    return summary


def _validate_columns(chunk: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(chunk.columns)
    if missing:
        raise ValueError(f"Unified data is missing EDA columns: {sorted(missing)}")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_category_counts(counts: dict[str, Counter[str]], path: Path, top_k: int) -> None:
    rows = [
        {"column": column, "value": value, "count": count}
        for column, counter in counts.items()
        for value, count in counter.most_common(top_k)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["column", "value", "count"]).to_csv(path, index=False)
