"""Typed configuration contracts for the data-processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search_ads_system.common.config import resolve_path


@dataclass(frozen=True)
class DelimitedDatasetConfig:
    """Configuration required to read a delimited file without guessing its schema."""

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
class OutputPaths:
    """Every generated artifact path; all must be under the configured outputs root."""

    root: Path
    schema_report: Path
    unified_data: Path
    eda_summary: Path
    eda_categories: Path
    feature_data: Path
    feature_metadata: Path


@dataclass(frozen=True)
class EdaConfig:
    """Settings for bounded-memory exploratory statistics."""

    categorical_columns: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class FeatureConfig:
    """Settings for deterministic, model-agnostic feature generation."""

    categorical_columns: tuple[str, ...]
    missing_category_token: str


@dataclass(frozen=True)
class PreprocessConfig:
    """Complete configuration for ingestion, conversion, EDA, and features."""

    dataset: DelimitedDatasetConfig
    outputs: OutputPaths
    eda: EdaConfig
    features: FeatureConfig

    @property
    def schema_report_path(self) -> Path:
        """Compatibility alias for the original schema-inspection entry point."""

        return self.outputs.schema_report


def parse_preprocess_config(raw_config: dict[str, Any], config_path: Path) -> PreprocessConfig:
    """Parse and validate the data-pipeline portion of a YAML configuration."""

    try:
        preprocessing = raw_config["preprocessing"]
        dataset = preprocessing["dataset"]
        paths = raw_config["paths"]
    except KeyError as error:
        raise ValueError("Configuration must define paths and preprocessing.dataset") from error

    config_directory = config_path.parent.resolve()
    missing_values = dataset.get("missing_value_tokens", {})
    by_column = {
        str(column): tuple(str(token) for token in tokens)
        for column, tokens in missing_values.get("by_column", {}).items()
    }
    parsed_dataset = DelimitedDatasetConfig(
        path=resolve_path(str(dataset["path"]), config_directory),
        delimiter=str(dataset.get("delimiter", "\t")),
        has_header=bool(dataset.get("has_header", True)),
        encoding=str(dataset.get("encoding", "utf-8")),
        chunk_size=int(dataset.get("chunk_size", 100_000)),
        column_names=tuple(str(name) for name in dataset.get("column_names", [])),
        label_columns=tuple(str(name) for name in dataset.get("label_columns", [])),
        missing_value_tokens=tuple(str(token) for token in missing_values.get("default", [""])),
        missing_value_tokens_by_column=by_column,
    )
    output_root = resolve_path(str(paths["outputs_dir"]), config_directory)
    outputs = OutputPaths(
        root=output_root,
        schema_report=resolve_path(str(paths["schema_report"]), config_directory),
        unified_data=resolve_path(str(paths["unified_data"]), config_directory),
        eda_summary=resolve_path(str(paths["eda_summary"]), config_directory),
        eda_categories=resolve_path(str(paths["eda_categories"]), config_directory),
        feature_data=resolve_path(str(paths["feature_data"]), config_directory),
        feature_metadata=resolve_path(str(paths["feature_metadata"]), config_directory),
    )
    eda_raw = preprocessing.get("eda", {})
    feature_raw = preprocessing.get("features", {})
    parsed = PreprocessConfig(
        dataset=parsed_dataset,
        outputs=outputs,
        eda=EdaConfig(
            categorical_columns=tuple(str(value) for value in eda_raw.get("categorical_columns", [])),
            top_k=int(eda_raw.get("top_k", 20)),
        ),
        features=FeatureConfig(
            categorical_columns=tuple(str(value) for value in feature_raw.get("categorical_columns", [])),
            missing_category_token=str(feature_raw.get("missing_category_token", "__MISSING__")),
        ),
    )
    _validate_dataset_config(parsed.dataset)
    _validate_output_paths(parsed.outputs)
    if parsed.eda.top_k <= 0:
        raise ValueError("preprocessing.eda.top_k must be greater than zero")
    return parsed


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


def _validate_output_paths(outputs: OutputPaths) -> None:
    root = outputs.root.resolve()
    for name, path in vars(outputs).items():
        if name == "root":
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"paths.{name} must be within paths.outputs_dir: {path}") from error
