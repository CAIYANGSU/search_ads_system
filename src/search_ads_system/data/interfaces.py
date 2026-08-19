"""Explicit, source-agnostic contracts for delimited datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DelimitedDatasetConfig:
    """Configuration required to load a delimited dataset without guessing schema."""

    path: Path
    delimiter: str
    has_header: bool
    encoding: str
    chunk_size: int
    column_names: tuple[str, ...]
    label_columns: tuple[str, ...]
    missing_value_tokens: tuple[str, ...]
    missing_value_tokens_by_column: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for schema inspection and future preprocessing steps."""

    dataset: DelimitedDatasetConfig
    schema_report_path: Path


def parse_preprocess_config(raw_config: dict[str, Any], config_path: Path) -> PreprocessConfig:
    """Parse the explicit preprocessing contract from a loaded YAML mapping."""

    try:
        preprocessing = raw_config["preprocessing"]
        dataset = preprocessing["dataset"]
    except KeyError as error:
        raise ValueError("config.yaml must define preprocessing.dataset") from error

    config_directory = config_path.parent.resolve()
    dataset_path = _resolve_path(dataset["path"], config_directory)
    report_path = _resolve_path(preprocessing["schema_report_path"], config_directory)
    missing_values = dataset.get("missing_value_tokens", {})
    by_column = {
        column: tuple(str(token) for token in tokens)
        for column, tokens in missing_values.get("by_column", {}).items()
    }
    parsed = DelimitedDatasetConfig(
        path=dataset_path,
        delimiter=str(dataset.get("delimiter", "\t")),
        has_header=bool(dataset.get("has_header", True)),
        encoding=str(dataset.get("encoding", "utf-8")),
        chunk_size=int(dataset.get("chunk_size", 100_000)),
        column_names=tuple(str(name) for name in dataset.get("column_names", [])),
        label_columns=tuple(str(name) for name in dataset.get("label_columns", [])),
        missing_value_tokens=tuple(str(token) for token in missing_values.get("default", [""])),
        missing_value_tokens_by_column=by_column,
    )
    _validate_dataset_config(parsed)
    return PreprocessConfig(dataset=parsed, schema_report_path=report_path)


def _resolve_path(raw_path: str, config_directory: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else config_directory / path


def _validate_dataset_config(config: DelimitedDatasetConfig) -> None:
    if not config.path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {config.path}")
    if not config.delimiter:
        raise ValueError("dataset.delimiter must not be empty")
    if config.chunk_size <= 0:
        raise ValueError("dataset.chunk_size must be greater than zero")
    if len(config.column_names) != len(set(config.column_names)):
        raise ValueError("dataset.column_names must be unique")
    declared_columns = set(config.column_names)
    unknown_labels = set(config.label_columns) - declared_columns
    if config.column_names and unknown_labels:
        raise ValueError(f"label_columns are not declared in column_names: {sorted(unknown_labels)}")
    unknown_missing_columns = set(config.missing_value_tokens_by_column) - declared_columns
    if config.column_names and unknown_missing_columns:
        raise ValueError(
            "missing_value_tokens.by_column refers to undeclared columns: "
            f"{sorted(unknown_missing_columns)}"
        )
