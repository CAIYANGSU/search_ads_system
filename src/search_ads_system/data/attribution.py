"""Streaming Criteo Attribution impression data contract and strict temporal split."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pandas as pd

from search_ads_system.data.dataset import iter_delimited_chunks
from search_ads_system.data.interfaces import AttributionPreprocessConfig, DelimitedDatasetConfig
from search_ads_system.data.storage import iter_csv_parts, prepare_output_directory, write_csv_part


ATTRIBUTION_SOURCE_COLUMNS: tuple[str, ...] = (
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click", "click_pos", "click_nb", "cost", "cpo", "time_since_last_click",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
)
ATTRIBUTION_COLUMNS: tuple[str, ...] = (
    "event_id", "source_row_number", "timestamp", "user_id", "campaign_id", "click", "conversion",
    "click_and_conversion", "cost", "cpo", "attribution", "conversion_timestamp", "conversion_id",
    "click_pos", "click_nb", "time_since_last_click", "cat1", "cat2", "cat3", "cat4", "cat5",
    "cat6", "cat7", "cat8", "cat9",
)
SAFE_IMPRESSION_TIME_FEATURES: tuple[str, ...] = (
    "timestamp", "user_id", "campaign_id", "time_since_last_click", "cat1", "cat2", "cat3", "cat4",
    "cat5", "cat6", "cat7", "cat8", "cat9",
)
POST_EVENT_ATTRIBUTION_FIELDS: tuple[str, ...] = (
    "conversion_timestamp", "conversion_id", "attribution", "click_pos", "click_nb",
)
COST_ACCOUNTING_FIELDS: tuple[str, ...] = ("cost",)
LEAKAGE_RISK_FIELDS: tuple[str, ...] = ("click", "conversion", "click_and_conversion", "cpo")


@dataclass(frozen=True)
class AttributionBuildResult:
    rows_written: int
    parts_written: int
    output_directory: Path
    metadata_path: Path


@dataclass(frozen=True)
class AttributionSplitResult:
    boundaries: dict[str, int]
    split_directories: dict[str, Path]


@dataclass
class _LabelStats:
    rows: int = 0
    click_positive: int = 0
    conversion_positive: int = 0
    ctcvr_positive: int = 0
    click_0_conversion_0: int = 0
    click_1_conversion_0: int = 0
    click_1_conversion_1: int = 0
    click_0_conversion_1: int = 0
    timestamp_min: int | None = None
    timestamp_max: int | None = None

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        timestamp = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
        click = pd.to_numeric(frame["click"], errors="raise").astype("int64")
        conversion = pd.to_numeric(frame["conversion"], errors="raise").astype("int64")
        joint = pd.to_numeric(frame["click_and_conversion"], errors="raise").astype("int64")
        self.rows += len(frame)
        self.click_positive += int(click.sum())
        self.conversion_positive += int(conversion.sum())
        self.ctcvr_positive += int(joint.sum())
        self.click_0_conversion_0 += int(((click == 0) & (conversion == 0)).sum())
        self.click_1_conversion_0 += int(((click == 1) & (conversion == 0)).sum())
        self.click_1_conversion_1 += int(((click == 1) & (conversion == 1)).sum())
        self.click_0_conversion_1 += int(((click == 0) & (conversion == 1)).sum())
        observed_min, observed_max = int(timestamp.min()), int(timestamp.max())
        self.timestamp_min = observed_min if self.timestamp_min is None else min(self.timestamp_min, observed_min)
        self.timestamp_max = observed_max if self.timestamp_max is None else max(self.timestamp_max, observed_max)

    def as_dict(self, *, users: int | None = None, campaigns: int | None = None) -> dict[str, Any]:
        rate = lambda value: value / self.rows if self.rows else 0.0
        result: dict[str, Any] = {
            "rows": self.rows,
            "timestamp_min": self.timestamp_min,
            "timestamp_max": self.timestamp_max,
            "click_positives": self.click_positive,
            "conversion_positives": self.conversion_positive,
            "click_and_conversion_positives": self.ctcvr_positive,
            "ctr_rate": rate(self.click_positive),
            "conversion_rate": rate(self.conversion_positive),
            "ctcvr_rate": rate(self.ctcvr_positive),
            "label_cross_table": {
                "click_0_conversion_0": self.click_0_conversion_0,
                "click_1_conversion_0": self.click_1_conversion_0,
                "click_1_conversion_1": self.click_1_conversion_1,
                "click_0_conversion_1": self.click_0_conversion_1,
            },
        }
        if users is not None:
            result["users"] = users
        if campaigns is not None:
            result["campaigns"] = campaigns
        return result


def normalize_attribution_chunk(
    raw_chunk: pd.DataFrame, source_row_offset: int, dataset_config: DelimitedDatasetConfig
) -> pd.DataFrame:
    """Normalize one raw Attribution chunk without changing its original outcomes."""

    missing = set(ATTRIBUTION_SOURCE_COLUMNS) - set(raw_chunk.columns)
    if missing:
        raise ValueError(f"Attribution source is missing required columns: {sorted(missing)}")
    normalized = pd.DataFrame(index=raw_chunk.index)
    source_rows = pd.Series(
        range(source_row_offset, source_row_offset + len(raw_chunk)), index=raw_chunk.index, dtype="Int64"
    )
    normalized["source_row_number"] = source_rows
    normalized["event_id"] = "criteo-attribution-" + source_rows.astype("string").str.zfill(12)
    normalized["timestamp"] = _required_integer(raw_chunk["timestamp"], "timestamp", dataset_config)
    normalized["user_id"] = _nullable_string(raw_chunk["uid"], "uid", dataset_config)
    normalized["campaign_id"] = _nullable_string(raw_chunk["campaign"], "campaign", dataset_config)
    normalized["click"] = _binary_or_integer(raw_chunk["click"], "click", dataset_config, binary=True).astype("Int8")
    normalized["conversion"] = _binary_or_integer(
        raw_chunk["conversion"], "conversion", dataset_config, binary=True
    ).astype("Int8")
    normalized["click_and_conversion"] = (
        (normalized["click"] == 1) & (normalized["conversion"] == 1)
    ).astype("Int8")
    for column in ("cost", "cpo"):
        normalized[column] = pd.to_numeric(_replace_missing(raw_chunk[column], column, dataset_config), errors="coerce").astype("Float64")
    for source_name, target_name in (
        ("attribution", "attribution"), ("conversion_timestamp", "conversion_timestamp"),
        ("conversion_id", "conversion_id"), ("click_pos", "click_pos"), ("click_nb", "click_nb"),
        ("time_since_last_click", "time_since_last_click"),
    ):
        normalized[target_name] = _nullable_integer(raw_chunk[source_name], source_name, dataset_config)
    for category in (f"cat{index}" for index in range(1, 10)):
        normalized[category] = _nullable_string(raw_chunk[category], category, dataset_config)
    return normalized.loc[:, ATTRIBUTION_COLUMNS]


def build_attribution_impressions(
    config: AttributionPreprocessConfig, *, overwrite: bool = False
) -> AttributionBuildResult:
    """Stream raw impression rows into the independent Attribution schema."""

    prepare_output_directory(config.processed_data, overwrite=overwrite)
    stats = _LabelStats()
    rows_written = 0
    parts_written = 0
    for part_number, raw_chunk in enumerate(iter_delimited_chunks(config.dataset)):
        normalized = normalize_attribution_chunk(raw_chunk, rows_written, config.dataset)
        write_csv_part(normalized, config.processed_data, part_number)
        stats.update(normalized)
        rows_written += len(normalized)
        parts_written += 1
    metadata = {
        "contract_version": "1.0",
        "raw_rows": rows_written,
        "processed_rows": rows_written,
        "parts_written": parts_written,
        "processed_data": str(config.processed_data),
        "summary": stats.as_dict(),
        "search_conversion_join": False,
        "labels": {
            "click": "Raw 0/1 clicked-impression label.",
            "conversion": "Raw 0/1 conversion-within-window label; it is retained unchanged.",
            "click_and_conversion": "Derived exactly as click AND conversion.",
        },
    }
    _write_json(config.build_metadata_path, metadata)
    return AttributionBuildResult(rows_written, parts_written, config.processed_data, config.build_metadata_path)


def split_attribution_temporally(
    config: AttributionPreprocessConfig, *, overwrite: bool = False
) -> AttributionSplitResult:
    """Create disjoint, timestamp-strict Past/Future-A/Future-B CSV partitions."""

    boundaries = _compute_strict_boundaries(config.processed_data, config.dataset.chunk_size, config.past_ratio, config.future_a_ratio)
    directories = {
        "past": config.temporal_output_dir / "split" / "past",
        "future_a": config.temporal_output_dir / "split" / "future_a",
        "future_b": config.temporal_output_dir / "split" / "future_b",
    }
    _assert_split_outputs_ready(directories.values(), overwrite)
    for directory in directories.values():
        prepare_output_directory(directory, overwrite=overwrite)
    part_numbers = {name: 0 for name in directories}
    stats = {name: _LabelStats() for name in directories}
    for chunk in iter_csv_parts(config.processed_data, config.dataset.chunk_size):
        timestamp = pd.to_numeric(chunk["timestamp"], errors="raise")
        partitions = {
            "past": chunk.loc[timestamp < boundaries["future_a_start"]],
            "future_a": chunk.loc[(timestamp >= boundaries["future_a_start"]) & (timestamp < boundaries["future_b_start"])],
            "future_b": chunk.loc[timestamp >= boundaries["future_b_start"]],
        }
        if sum(len(partition) for partition in partitions.values()) != len(chunk):
            raise AssertionError("Temporal partitioning failed to assign every impression exactly once")
        for name, partition in partitions.items():
            if partition.empty:
                continue
            write_csv_part(partition, directories[name], part_numbers[name])
            part_numbers[name] += 1
            stats[name].update(partition)
    if any(stats[name].rows == 0 for name in directories):
        raise ValueError("Strict temporal split produced an empty partition; choose a dataset with more timestamp support")
    _assert_temporal_order(stats["past"], stats["future_a"], stats["future_b"])
    _write_json(
        config.temporal_output_dir / "split" / "metadata.json",
        {
            "contract_version": "1.0",
            "boundaries": boundaries,
            "part_counts": part_numbers,
            "split_summary": {name: value.as_dict() for name, value in stats.items()},
            "strict_order": "max(past.timestamp) < min(future_a.timestamp) < min(future_b.timestamp)",
            "search_conversion_join": False,
        },
    )
    return AttributionSplitResult(boundaries=boundaries, split_directories=directories)


def build_attribution_audit(config: AttributionPreprocessConfig) -> dict[str, Any]:
    """Write model-free label, leakage, and temporal-contract diagnostics."""

    split_directories = {
        "past": config.temporal_output_dir / "split" / "past",
        "future_a": config.temporal_output_dir / "split" / "future_a",
        "future_b": config.temporal_output_dir / "split" / "future_b",
    }
    processed = _collect_stats(config.processed_data, config.dataset.chunk_size)
    splits = {name: _collect_stats(path, config.dataset.chunk_size) for name, path in split_directories.items()}
    _assert_temporal_order_from_dicts(splits["past"], splits["future_a"], splits["future_b"])
    audit = {
        "audit_version": "1.0",
        "raw_rows": processed["rows"],
        "processed_rows": processed["rows"],
        "timestamp_range": {"min": processed["timestamp_min"], "max": processed["timestamp_max"]},
        "label_diagnostics": processed,
        "temporal_split": {
            "contract": "timestamp-strict: max(Past) < min(Future-A) < min(Future-B); no random split",
            "past_ratio_target": config.past_ratio,
            "future_a_ratio_within_future_target": config.future_a_ratio,
            "splits": splits,
        },
        "esmm_label_contract": {
            "y_ctr": "click",
            "y_ctcvr": "click_and_conversion = click AND conversion",
            "pCTR": "P(click | impression)",
            "pCTCVR": "P(click AND conversion | impression)",
            "pCVR": "pCTCVR / pCTR; derive only downstream with a pCTR numerical guard",
            "raw_conversion_semantics": "The raw conversion label remains an impression-window outcome and is not used as a full-impression CVR target.",
            "click_0_conversion_1_policy": "Rows are retained unchanged. Their existence records the raw attribution/post-view semantics; they receive click_and_conversion=0.",
        },
        "feature_eligibility": {
            "A_safe_impression_time_features": list(SAFE_IMPRESSION_TIME_FEATURES),
            "B_post_event_or_attribution_only": list(POST_EVENT_ATTRIBUTION_FIELDS),
            "C_cost_accounting_fields": list(COST_ACCOUNTING_FIELDS),
            "D_label_derived_or_leakage_risk": list(LEAKAGE_RISK_FIELDS),
            "default_esmm_features": list(SAFE_IMPRESSION_TIME_FEATURES),
        },
        "cost_cpo_semantics": {
            "cost": "README describes transformed price paid by Criteo. Retained for business analysis, excluded from default ESMM inputs.",
            "cpo": "README describes transformed cost-per-order for attributed conversions. It may depend on conversion/attribution outcome and is classified as leakage risk; excluded from default ESMM inputs.",
        },
        "touchpoint_semantics": {
            "retained_fields": ["user_id", "timestamp", "conversion_id", "conversion_timestamp", "attribution", "click_pos", "click_nb", "time_since_last_click"],
            "policy": "Retained for attribution analysis. Only time_since_last_click is in the safe set, contingent on serving-time availability; outcome-linked touchpoint fields are excluded.",
        },
        "search_conversion_join": False,
    }
    _write_json(config.audit_path, audit)
    _write_markdown(config.audit_path.with_suffix(".md"), audit)
    return audit


def _replace_missing(values: pd.Series, column: str, config: DelimitedDatasetConfig) -> pd.Series:
    tokens = set(config.missing_value_tokens)
    tokens.update(config.missing_value_tokens_by_column.get(column, ()))
    text = values.astype("string")
    return text.mask(text.isin(tokens), pd.NA)


def _nullable_string(values: pd.Series, column: str, config: DelimitedDatasetConfig) -> pd.Series:
    return _replace_missing(values, column, config).astype("string")


def _binary_or_integer(
    values: pd.Series, column: str, config: DelimitedDatasetConfig, *, binary: bool
) -> pd.Series:
    integers = _required_integer(values, column, config)
    if binary and not set(integers.unique()).issubset({0, 1}):
        raise ValueError(f"Attribution label {column} must be binary")
    return integers


def _required_integer(values: pd.Series, column: str, config: DelimitedDatasetConfig) -> pd.Series:
    numeric = pd.to_numeric(_replace_missing(values, column, config), errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Attribution field {column} contains missing or non-numeric values")
    if (numeric % 1 != 0).any():
        raise ValueError(f"Attribution field {column} must be integral")
    return numeric.astype("Int64")


def _nullable_integer(values: pd.Series, column: str, config: DelimitedDatasetConfig) -> pd.Series:
    numeric = pd.to_numeric(_replace_missing(values, column, config), errors="coerce")
    non_missing = numeric.dropna()
    if (non_missing % 1 != 0).any():
        raise ValueError(f"Attribution field {column} must be integral when present")
    return numeric.astype("Int64")


def _compute_strict_boundaries(
    directory: Path, chunk_size: int, past_ratio: float, future_a_ratio: float
) -> dict[str, int]:
    total = 0
    previous: int | None = None
    for chunk in iter_csv_parts(directory, chunk_size):
        timestamp = pd.to_numeric(chunk["timestamp"], errors="raise").astype("int64")
        if previous is not None and int(timestamp.iloc[0]) < previous:
            raise ValueError("Attribution timestamp order is not non-decreasing")
        if (timestamp.diff().dropna() < 0).any():
            raise ValueError("Attribution timestamp order is not non-decreasing")
        previous = int(timestamp.iloc[-1])
        total += len(chunk)
    past_target = int(total * past_ratio)
    future_b_target = past_target + int((total - past_target) * future_a_ratio)
    if not (0 < past_target < future_b_target < total):
        raise ValueError("Dataset is too small to form non-empty Past, Future-A, and Future-B partitions")
    targets = {past_target: "future_a_start", future_b_target: "future_b_start"}
    bases: dict[str, int] = {}
    starts: dict[str, int] = {}
    row_index = 0
    for chunk in iter_csv_parts(directory, chunk_size):
        for timestamp in pd.to_numeric(chunk["timestamp"], errors="raise").astype("int64"):
            for target, name in targets.items():
                if row_index == target - 1:
                    bases[name] = int(timestamp)
                elif name in bases and name not in starts and int(timestamp) > bases[name]:
                    starts[name] = int(timestamp)
            row_index += 1
    if set(starts) != {"future_a_start", "future_b_start"}:
        raise ValueError("Timestamp ties prevent a strict three-way split; no split was written")
    if not starts["future_a_start"] < starts["future_b_start"]:
        raise ValueError("Timestamp support cannot form a strict Future-A/Future-B boundary")
    return starts


def _assert_split_outputs_ready(directories: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    occupied = [str(directory) for directory in directories if directory.exists() and any(directory.iterdir())]
    if occupied:
        raise FileExistsError(f"Temporal split output is not empty: {occupied}. Use --overwrite to replace generated parts.")


def _assert_temporal_order(past: _LabelStats, future_a: _LabelStats, future_b: _LabelStats) -> None:
    if not (past.timestamp_max < future_a.timestamp_min < future_b.timestamp_min):
        raise ValueError("Strict temporal order failed: max(Past) < min(Future-A) < min(Future-B) is required")


def _assert_temporal_order_from_dicts(past: dict[str, Any], future_a: dict[str, Any], future_b: dict[str, Any]) -> None:
    if not (past["timestamp_max"] < future_a["timestamp_min"] < future_b["timestamp_min"]):
        raise ValueError("Strict temporal order failed: max(Past) < min(Future-A) < min(Future-B) is required")


def _collect_stats(directory: Path, chunk_size: int) -> dict[str, Any]:
    stats = _LabelStats()
    with tempfile.TemporaryDirectory(prefix="attribution-audit-", dir=directory.parent) as temporary_directory:
        connection = sqlite3.connect(Path(temporary_directory) / "distinct.sqlite")
        try:
            connection.execute("CREATE TABLE users (identifier TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE campaigns (identifier TEXT PRIMARY KEY)")
            for chunk in iter_csv_parts(directory, chunk_size):
                stats.update(chunk)
                _insert_distinct(connection, "users", chunk["user_id"])
                _insert_distinct(connection, "campaigns", chunk["campaign_id"])
            connection.commit()
            users = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            campaigns = int(connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0])
        finally:
            connection.close()
    return stats.as_dict(users=users, campaigns=campaigns)


def _insert_distinct(connection: sqlite3.Connection, table: str, values: pd.Series) -> None:
    unique_values = values.dropna().astype("string").unique().tolist()
    connection.executemany(f"INSERT OR IGNORE INTO {table} (identifier) VALUES (?)", ((value,) for value in unique_values))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_markdown(path: Path, audit: dict[str, Any]) -> None:
    labels = audit["label_diagnostics"]
    splits = audit["temporal_split"]["splits"]
    feature_groups = audit["feature_eligibility"]
    text = "\n".join((
        "# Attribution Impression Data Audit",
        "",
        "This is an Attribution-only, impression-level contract. No Search Conversion join, synthetic impression negative, or model training is performed.",
        "",
        "## Labels",
        "",
        f"- Raw/processed impressions: {audit['raw_rows']} / {audit['processed_rows']}",
        f"- Timestamp range: {audit['timestamp_range']['min']} to {audit['timestamp_range']['max']}",
        f"- CTR: {labels['ctr_rate']:.8f}; conversion rate: {labels['conversion_rate']:.8f}; CTCVR: {labels['ctcvr_rate']:.8f}",
        f"- click=0, conversion=1: {labels['label_cross_table']['click_0_conversion_1']} (retained; click_and_conversion remains 0)",
        "- ESMM labels: y_ctr=click; y_ctcvr=click AND conversion. Raw conversion is not a full-impression CVR target.",
        "",
        "## Strict temporal split",
        "",
        "`max(Past timestamp) < min(Future-A timestamp) < min(Future-B timestamp)`; no random split.",
        *(
            f"- {name}: rows={summary['rows']}, users={summary['users']}, campaigns={summary['campaigns']}, "
            f"timestamp=[{summary['timestamp_min']}, {summary['timestamp_max']}], "
            f"click={summary['click_positives']}, conversion={summary['conversion_positives']}, "
            f"click_and_conversion={summary['click_and_conversion_positives']}"
            for name, summary in splits.items()
        ),
        "",
        "## Feature eligibility",
        "",
        f"- A. Safe impression-time features: {', '.join(feature_groups['A_safe_impression_time_features'])}",
        f"- B. Post-event / attribution-only: {', '.join(feature_groups['B_post_event_or_attribution_only'])}",
        f"- C. Cost/accounting: {', '.join(feature_groups['C_cost_accounting_fields'])}",
        f"- D. Label-derived / leakage risk: {', '.join(feature_groups['D_label_derived_or_leakage_risk'])}",
        "",
        "`cost` and `cpo` are transformed fields. Neither is a default ESMM feature; `cpo` is treated as outcome-dependent leakage risk.",
        "",
        "search_conversion_join = false",
        "",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
