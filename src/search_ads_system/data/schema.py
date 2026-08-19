"""Streaming schema checks for configured datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_ads_system.data.dataset import iter_delimited_chunks
from search_ads_system.data.interfaces import DelimitedDatasetConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class _TypeState:
    numeric_possible: bool = True
    integer_possible: bool = True
    values_seen: int = 0

    def observe(self, values: pd.Series) -> None:
        if values.empty:
            return
        self.values_seen += len(values)
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            self.numeric_possible = False
            self.integer_possible = False
            return
        if self.integer_possible:
            numeric_values = numeric.to_numpy(dtype=float)
            if not np.equal(numeric_values, np.floor(numeric_values)).all():
                self.integer_possible = False

    @property
    def dtype(self) -> str:
        if self.values_seen == 0:
            return "unknown"
        if not self.numeric_possible:
            return "string"
        return "int64" if self.integer_possible else "float64"


def inspect_schema(config: DelimitedDatasetConfig) -> dict[str, Any]:
    """Return file, row, column, dtype, missingness, and label-distribution stats."""

    column_names: list[str] | None = None
    missing_counts: dict[str, int] = {}
    type_states: dict[str, _TypeState] = {}
    label_counts: dict[str, Counter[str]] = {label: Counter() for label in config.label_columns}
    row_count = 0

    for chunk in iter_delimited_chunks(config):
        if column_names is None:
            column_names = list(chunk.columns)
            _validate_observed_schema(config, column_names)
            missing_counts = {column: 0 for column in column_names}
            type_states = {column: _TypeState() for column in column_names}

        row_count += len(chunk)
        LOGGER.info("Inspected %s source rows so far", row_count)
        for column in column_names:
            missing = _missing_mask(chunk[column], config, column)
            missing_counts[column] += int(missing.sum())
            type_states[column].observe(chunk.loc[~missing, column])

        for label in config.label_columns:
            label_values = chunk.loc[:, label].astype("string")
            label_counts[label].update(label_values.value_counts(dropna=False).to_dict())

    if column_names is None:
        column_names = list(config.column_names)
        missing_counts = {column: 0 for column in column_names}
        type_states = {column: _TypeState() for column in column_names}

    file_size = config.path.stat().st_size
    return {
        "dataset_path": str(config.path),
        "file_size_bytes": file_size,
        "file_size_human": _format_size(file_size),
        "row_count": row_count,
        "column_count": len(column_names),
        "columns": [
            {
                "name": column,
                "dtype": type_states[column].dtype,
                "missing_count": missing_counts[column],
                "missing_rate": round(missing_counts[column] / row_count, 8) if row_count else 0.0,
            }
            for column in column_names
        ],
        "label_distribution": {
            label: {str(value): count for value, count in sorted(counts.items())}
            for label, counts in label_counts.items()
        },
    }


def _validate_observed_schema(config: DelimitedDatasetConfig, observed_columns: list[str]) -> None:
    if config.column_names and tuple(observed_columns) != config.column_names:
        raise ValueError(
            "Observed columns do not match the explicit column_names contract. "
            f"Expected {len(config.column_names)} columns, received {len(observed_columns)}."
        )
    unknown_labels = set(config.label_columns) - set(observed_columns)
    if unknown_labels:
        raise ValueError(f"Configured labels are absent from source data: {sorted(unknown_labels)}")


def _missing_mask(chunk: pd.Series, config: DelimitedDatasetConfig, column: str) -> pd.Series:
    tokens = set(config.missing_value_tokens)
    tokens.update(config.missing_value_tokens_by_column.get(column, ()))
    values = chunk.astype("string")
    return values.isna() | values.isin(tokens)


def _format_size(size_in_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(size_in_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")
