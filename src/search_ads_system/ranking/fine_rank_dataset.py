"""Streaming sample construction and parquet cache for fine ranking.

Rows in the Criteo search-conversion dataset are already click interactions.
Consequently this module only emits coarse candidates whose user/product pair
is an observed interaction.  ``conversion_label`` is the click-conditioned
CVR label; candidates that were never clicked are never invented as negatives.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from search_ads_system.data.storage import iter_csv_parts

LOGGER = logging.getLogger(__name__)

FEATURE_VERSION = "fine-rank-observed-clicks-v3-bounded-dense"
DATASET_CONSTRUCTION_VERSION = "observed-clicked-interactions-v2"
VALUE_TRANSFORM_VERSION = "normalized-log1p-v1"
DENSE_FEATURES = (
    "rrf_score", "source_count", "coarse_score", "inverse_coarse_rank",
    "product_price", "log_product_price", "clicks_last_7d", "log_clicks_last_7d",
    "click_hour_utc", "click_day_of_week_utc", "product_price_missing",
    "clicks_last_7d_missing", "timestamp_missing",
)
SPARSE_FEATURES = (
    "user_id", "product_id", "product_age_group", "device_type", "product_gender",
    "product_brand", "product_category_1", "product_category_2", "product_category_3", "product_country",
)
LEAKAGE_COLUMNS = {
    "conversion_label", "conversion_value_eur", "conversion_delay_seconds", "conversion_delay_hours",
    "future_label", "future_a", "future_b", "future_timestamp", "click_timestamp",
}
DEFAULT_BUCKET_SIZES = (1_000_003, 1_000_003, 10_007, 10_007, 10_007, 100_003, 100_003, 100_003, 100_003, 10_007)
PRODUCT_COLUMNS = ("product_price", "product_age_group", "product_gender", "product_brand", "product_category_1", "product_category_2", "product_category_3", "product_country")
USER_COLUMNS = ("clicks_last_7d", "device_type", "click_timestamp")
MAX_DENSE_ABS_VALUE = 10.0


class SparseHashCache:
    """Caches stable hashes by sparse field across inference batches."""

    def __init__(self, bucket_sizes: Sequence[int], random_seed: int) -> None:
        self.bucket_sizes = tuple(int(value) for value in bucket_sizes)
        self.random_seed = random_seed
        self._values: list[dict[str, int]] = [dict() for _ in SPARSE_FEATURES]

    def encode(self, values: pd.Series, index: int) -> np.ndarray:
        normalized = _normalise_id_series(values).fillna("__MISSING__")
        codes, unique = pd.factorize(normalized, sort=False)
        cache = self._values[index]
        bucket = self.bucket_sizes[index]
        encoded_unique = np.empty(len(unique), dtype=np.int64)
        for position, value in enumerate(unique):
            text = str(value)
            hashed = cache.get(text)
            if hashed is None:
                hashed = stable_hash(text, bucket, self.random_seed + index)
                cache[text] = hashed
            encoded_unique[position] = hashed
        return encoded_unique[codes]


def assert_no_fine_rank_leakage(feature_columns: Sequence[str]) -> None:
    leaked = set(feature_columns) & LEAKAGE_COLUMNS
    if leaked:
        raise ValueError(f"Fine-rank feature list contains leakage columns: {sorted(leaked)}")
    future = [column for column in feature_columns if "future" in column.lower()]
    if future:
        raise ValueError(f"Fine-rank feature list contains Future-derived columns: {future}")


def stable_hash(value: object, bucket_size: int, seed: int = 2026) -> int:
    """Stable, process-independent feature hashing."""
    if bucket_size <= 1:
        raise ValueError("bucket_size must be greater than one")
    text = "__MISSING__" if value is None or pd.isna(value) else str(value).strip() or "__MISSING__"
    digest = hashlib.blake2b(f"{seed}\x1f{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) % bucket_size


@dataclass(frozen=True)
class FineRankDatasetSpec:
    cache_dir: Path
    candidate_path: Path
    feature_source_path: Path
    train_label_path: Path
    validation_label_path: Path | None
    mode: str
    max_train_rows: int
    chunk_size: int
    validation_fraction: float
    random_seed: int
    value_log_clip_max: float = 20.0
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"

    @property
    def index_path(self) -> Path:
        return self.cache_dir.parent / "feature_index.sqlite"

    @property
    def validation_dir(self) -> Path:
        return self.cache_dir.parent / "validation"


class FineRankFeatureStore:
    """Vectorized product/user feature lookup for fine-rank inference.

    The initial implementation assigned SQLite results into a DataFrame cell by
    cell, which makes tens of millions of candidates effectively single-core.
    This store preloads a reasonably sized index once, otherwise uses bounded
    SQLite queries followed by vectorized pandas merges.
    """

    def __init__(self, path: Path, *, memory_limit_bytes: int = 0) -> None:
        self.path = path
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.products: pd.DataFrame | None = None
        self.users: pd.DataFrame | None = None
        self.lookup_mode = "sqlite_vectorized"
        if memory_limit_bytes and path.stat().st_size <= memory_limit_bytes:
            self._preload()

    def close(self) -> None:
        self.connection.close()

    def enrich(self, candidates: pd.DataFrame) -> pd.DataFrame:
        frame = candidates.copy()
        frame["user_id"] = _normalise_id_series(frame["user_id"])
        frame["candidate_ad_id"] = _normalise_id_series(frame["candidate_ad_id"])
        frame = frame.drop(columns=["product_id", *PRODUCT_COLUMNS, *USER_COLUMNS], errors="ignore")
        if self.products is not None and self.users is not None:
            return frame.merge(self.products, how="left", left_on="candidate_ad_id", right_on="product_id", sort=False).drop(columns="product_id", errors="ignore").merge(self.users, how="left", on="user_id", sort=False)
        products = self._lookup_frame("products", "product_id", frame["candidate_ad_id"].dropna().unique(), PRODUCT_COLUMNS)
        users = self._lookup_frame("users", "user_id", frame["user_id"].dropna().unique(), USER_COLUMNS)
        return frame.merge(products, how="left", left_on="candidate_ad_id", right_on="product_id", sort=False).drop(columns="product_id", errors="ignore").merge(users, how="left", on="user_id", sort=False)

    def _preload(self) -> None:
        products = pd.read_sql_query(f"SELECT product_id, {', '.join(PRODUCT_COLUMNS)} FROM products", self.connection)
        users = pd.read_sql_query(f"SELECT user_id, {', '.join(USER_COLUMNS)} FROM users", self.connection)
        products["product_id"] = _normalise_id_series(products["product_id"])
        users["user_id"] = _normalise_id_series(users["user_id"])
        self.products = products
        self.users = users
        self.lookup_mode = "in_memory_vectorized"
        LOGGER.info("Preloaded fine-rank feature index into memory: products=%s users=%s index_bytes=%s", len(products), len(users), self.path.stat().st_size)

    def _lookup_frame(self, table: str, key: str, values: Sequence[object], columns: Sequence[str]) -> pd.DataFrame:
        query_values = [str(value) for value in values if value is not None and not pd.isna(value)]
        rows: list[pd.DataFrame] = []
        for start in range(0, len(values), 900):
            subset = query_values[start : start + 900]
            if not subset:
                continue
            marks = ",".join("?" for _ in subset)
            rows.append(pd.read_sql_query(f"SELECT {key}, {', '.join(columns)} FROM {table} WHERE {key} IN ({marks})", self.connection, params=subset))
        result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=(key, *columns))
        if not result.empty:
            result[key] = _normalise_id_series(result[key])
        return result


def cache_matches(spec: FineRankDatasetSpec) -> bool:
    if not spec.metadata_path.is_file() or not any(spec.cache_dir.glob("part-*.parquet")):
        return False
    try:
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    transform = metadata.get("value_transform", {})
    return (
        metadata.get("config_hash") == _config_hash(spec)
        and metadata.get("feature_version") == FEATURE_VERSION
        and metadata.get("dataset_construction_version") == DATASET_CONSTRUCTION_VERSION
        and transform.get("version") == VALUE_TRANSFORM_VERSION
        and transform.get("configured_prediction_log_clip_max") == spec.value_log_clip_max
    )


def build_or_reuse_cached_datasets(spec: FineRankDatasetSpec) -> dict[str, Any]:
    """Build interaction-first parquet shards, or reuse an exact matching cache.

    Coarse candidates are deliberately absent from this construction.  They
    describe inference inventory, not observed click-conditioned CVR labels.
    """
    assert_no_fine_rank_leakage(DENSE_FEATURES + SPARSE_FEATURES)
    if cache_matches(spec):
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        LOGGER.info("Reusing fine-rank dataset cache (%s rows): %s", metadata["row_count"], spec.cache_dir)
        return metadata
    LOGGER.info("Building fine-rank dataset cache because no matching metadata exists: %s", spec.cache_dir)
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    _clear_parts(spec.cache_dir)
    _clear_parts(spec.validation_dir)
    _build_index(spec)
    store = FineRankFeatureStore(spec.index_path)
    counts = {"train": 0, "validation": 0}
    parts = {"train": 0, "validation": 0}
    buffers: dict[str, list[pd.DataFrame]] = {"train": [], "validation": []}
    diagnostics = _new_diagnostics(spec)
    value_stats = _RunningLogStats()
    diagnostic_database = spec.cache_dir.parent / ".fine_rank_dataset_diagnostics.sqlite"
    if diagnostic_database.exists():
        diagnostic_database.unlink()
    diagnostic_connection = sqlite3.connect(diagnostic_database)
    _create_diagnostic_tables(diagnostic_connection)
    try:
        if spec.mode == "temporal":
            _append_observed_split(spec.train_label_path, "train", spec, store, counts, parts, buffers, diagnostics, value_stats, diagnostic_connection, enrich_from_past=True)
            if spec.validation_label_path is None:
                raise ValueError("Temporal fine-rank requires Future-B validation labels")
            _append_observed_split(spec.validation_label_path, "validation", spec, store, counts, parts, buffers, diagnostics, value_stats, diagnostic_connection, enrich_from_past=True)
        else:
            _append_full_observed_rows(spec, counts, parts, buffers, diagnostics, value_stats, diagnostic_connection)
        for split in ("train", "validation"):
            if buffers[split]:
                destination = spec.cache_dir if split == "train" else spec.validation_dir
                parts[split] = _write_part(pd.concat(buffers[split], ignore_index=True), destination, parts[split])
        if counts["train"] == 0:
            raise ValueError("No observed click interactions available for fine-rank training after validation and configured limits")
        value_transform = value_stats.finalize(spec.value_log_clip_max)
        diagnostics.update(_read_diagnostic_counts(diagnostic_connection))
        diagnostics["train_rows"] = counts["train"]
        diagnostics["validation_rows"] = counts["validation"]
        diagnostics["conversion_positive_rows"] = diagnostics["train_conversion_positive_rows"] + diagnostics["validation_conversion_positive_rows"]
        diagnostics["conversion_positive_rate"] = diagnostics["conversion_positive_rows"] / (counts["train"] + counts["validation"]) if counts["train"] + counts["validation"] else 0.0
        diagnostics["valid_conversion_value_rows"] = diagnostics["train_valid_conversion_value_rows"] + diagnostics["validation_valid_conversion_value_rows"]
        diagnostics["unique_users"] = diagnostics["train_unique_users"]
        diagnostics["unique_products"] = diagnostics["train_unique_products"]
        diagnostics["train_rows_shortfall_reason"] = _shortfall_reason(spec.max_train_rows, counts["train"], diagnostics)
        metadata = {
            "feature_version": FEATURE_VERSION,
            "dataset_construction_version": DATASET_CONSTRUCTION_VERSION,
            "feature_schema": {"dense": list(DENSE_FEATURES), "sparse": list(SPARSE_FEATURES), "bucket_sizes": list(spec.bucket_sizes)},
            "source_paths": {"inference_candidates": str(spec.candidate_path), "features": str(spec.feature_source_path), "train_labels": str(spec.train_label_path), "validation_labels": str(spec.validation_label_path) if spec.validation_label_path else None},
            "label_definition": "click-conditioned conversion_label; value target only for conversion_label=1 and non-missing conversion_value_eur",
            "split_definition": "Future-A/Future-B observed interactions with Past-only features" if spec.validation_label_path else f"stable {spec.validation_fraction:.0%} observed-interaction validation hash",
            "mode": spec.mode,
            "max_train_rows": spec.max_train_rows,
            "row_count": counts["train"], "validation_row_count": counts["validation"],
            "part_count": parts, "value_transform": value_transform, "diagnostics": diagnostics, "config_hash": _config_hash(spec),
        }
        spec.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Fine-rank cache built from observed clicks: train=%s validation=%s source_rows=%s conversion_rate=%.6f valid_value_rows=%s requested_max=%s reason=%s", counts["train"], counts["validation"], diagnostics["source_interaction_rows"], diagnostics["conversion_positive_rate"], diagnostics["valid_conversion_value_rows"], spec.max_train_rows, diagnostics["train_rows_shortfall_reason"])
        return metadata
    finally:
        store.close()
        diagnostic_connection.close()
        diagnostic_database.unlink(missing_ok=True)


class _RunningLogStats:
    """Float64 streaming moments for valid ``log1p(conversion_value)`` targets."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_squares = 0.0
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, encoded: pd.DataFrame) -> None:
        values = pd.to_numeric(encoded.loc[encoded["value_mask"].eq(1), "log_conversion_value"], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.count += len(values); self.total += float(values.sum()); self.total_squares += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min())); self.maximum = max(self.maximum, float(values.max()))

    def finalize(self, configured_clip_max: float) -> dict[str, float | int | str]:
        if self.count:
            mean = self.total / self.count
            variance = max(self.total_squares / self.count - mean * mean, 0.0)
            std = max(math.sqrt(variance), 1e-6)
            inferred_upper = self.maximum + 3.0 * std
        else:
            mean, std, inferred_upper = 0.0, 1.0, 0.0
        return {
            "version": VALUE_TRANSFORM_VERSION, "mean": float(mean), "std": float(std), "train_valid_value_count": int(self.count),
            "train_log_value_min": float(self.minimum if self.count else 0.0), "train_log_value_max": float(self.maximum if self.count else 0.0),
            "prediction_log_min": 0.0, "prediction_log_max": float(min(configured_clip_max, max(0.0, inferred_upper))),
            "configured_prediction_log_clip_max": float(configured_clip_max),
        }


def _append_full_observed_rows(spec: FineRankDatasetSpec, counts: dict[str, int], parts: dict[str, int], buffers: dict[str, list[pd.DataFrame]], diagnostics: dict[str, Any], value_stats: _RunningLogStats, diagnostic_connection: sqlite3.Connection) -> None:
    offset = 0
    for chunk in _iter_csv_chunks(spec.train_label_path, spec.chunk_size):
        prepared = _prepare_observed_rows(chunk, diagnostics, "full")
        keys = _observation_keys(prepared, offset)
        offset += len(chunk)
        validation_mask = np.fromiter((stable_hash(key, 10_000, spec.random_seed) < int(spec.validation_fraction * 10_000) for key in keys), dtype=bool, count=len(prepared))
        _append_encoded_rows(encode_feature_frame(prepared.loc[validation_mask], bucket_sizes=spec.bucket_sizes, random_seed=spec.random_seed), "validation", spec, counts, parts, buffers, diagnostics, value_stats, diagnostic_connection)
        _append_encoded_rows(encode_feature_frame(prepared.loc[~validation_mask], bucket_sizes=spec.bucket_sizes, random_seed=spec.random_seed), "train", spec, counts, parts, buffers, diagnostics, value_stats, diagnostic_connection)


def _append_observed_split(path: Path, split: str, spec: FineRankDatasetSpec, store: FineRankFeatureStore, counts: dict[str, int], parts: dict[str, int], buffers: dict[str, list[pd.DataFrame]], diagnostics: dict[str, Any], value_stats: _RunningLogStats, diagnostic_connection: sqlite3.Connection, *, enrich_from_past: bool) -> None:
    for chunk in _iter_csv_chunks(path, spec.chunk_size):
        prepared = _prepare_observed_rows(chunk, diagnostics, split)
        if enrich_from_past and not prepared.empty:
            prepared = store.enrich(prepared)
        _append_encoded_rows(encode_feature_frame(prepared, bucket_sizes=spec.bucket_sizes, random_seed=spec.random_seed), split, spec, counts, parts, buffers, diagnostics, value_stats, diagnostic_connection)


def _prepare_observed_rows(chunk: pd.DataFrame, diagnostics: dict[str, Any], source_name: str) -> pd.DataFrame:
    _require_interaction_columns(chunk)
    diagnostics["source_interaction_rows"] += len(chunk)
    diagnostics["source_rows_by_window"][source_name] = diagnostics["source_rows_by_window"].get(source_name, 0) + len(chunk)
    frame = chunk.copy()
    frame["user_id"] = frame["user_id"].map(_normalise_id)
    frame["candidate_ad_id"] = frame["product_id"].map(_normalise_id)
    valid_ids = frame["user_id"].notna() & frame["candidate_ad_id"].notna()
    diagnostics["filter_rows"]["invalid_user_or_product_id"] += int((~valid_ids).sum())
    frame = frame.loc[valid_ids].copy()
    numeric_label = pd.to_numeric(frame["conversion_label"], errors="coerce")
    valid_label = numeric_label.isin((0, 1))
    diagnostics["filter_rows"]["invalid_conversion_label"] += int((~valid_label).sum())
    frame = frame.loc[valid_label].copy()
    frame["conversion_label"] = numeric_label.loc[valid_label].astype(np.int8)
    diagnostics["filter_rows"]["observed_click_rows_after_required_fields"] += len(frame)
    raw_value = pd.to_numeric(frame.get("conversion_value_eur", pd.Series(np.nan, index=frame.index)), errors="coerce")
    valid_value = frame["conversion_label"].eq(1) & np.isfinite(raw_value) & raw_value.ge(0)
    diagnostics["filter_rows"]["conversion_rows_without_valid_value"] += int(frame["conversion_label"].eq(1).sum() - valid_value.sum())
    return frame


def _append_encoded_rows(encoded: pd.DataFrame, split: str, spec: FineRankDatasetSpec, counts: dict[str, int], parts: dict[str, int], buffers: dict[str, list[pd.DataFrame]], diagnostics: dict[str, Any], value_stats: _RunningLogStats, diagnostic_connection: sqlite3.Connection) -> None:
    if encoded.empty:
        return
    diagnostics["filter_rows"][f"{split}_rows_before_max_train_rows"] += len(encoded)
    if split == "train":
        remaining = max(0, spec.max_train_rows - counts["train"])
        if len(encoded) > remaining:
            diagnostics["filter_rows"]["excluded_by_max_train_rows"] += len(encoded) - remaining
            encoded = encoded.iloc[:remaining].copy()
    if encoded.empty:
        return
    if split == "train":
        value_stats.update(encoded)
    positives = int(encoded["conversion_label"].sum())
    valid_values = int(encoded["value_mask"].sum())
    diagnostics[f"{split}_conversion_positive_rows"] += positives
    diagnostics[f"{split}_valid_conversion_value_rows"] += valid_values
    _record_diagnostic_rows(diagnostic_connection, split, encoded)
    counts[split], parts[split] = _append_and_flush(encoded, split, buffers, counts[split], parts[split], spec)


def _observation_keys(frame: pd.DataFrame, offset: int) -> list[str]:
    stable_source = frame.get("event_id", frame.get("source_row_number", pd.Series("", index=frame.index))).astype(str)
    timestamps = frame.get("click_timestamp", pd.Series("", index=frame.index)).astype(str)
    return [f"{user}\x1f{product}\x1f{timestamp}\x1f{source}\x1f{offset + index}" for index, (user, product, timestamp, source) in enumerate(zip(frame["user_id"], frame["candidate_ad_id"], timestamps, stable_source))]


def _new_diagnostics(spec: FineRankDatasetSpec) -> dict[str, Any]:
    return {
        "source_interaction_rows": 0, "source_rows_by_window": {},
        "filter_rows": {"invalid_user_or_product_id": 0, "invalid_conversion_label": 0, "observed_click_rows_after_required_fields": 0, "conversion_rows_without_valid_value": 0, "train_rows_before_max_train_rows": 0, "validation_rows_before_max_train_rows": 0, "excluded_by_max_train_rows": 0},
        "train_conversion_positive_rows": 0, "validation_conversion_positive_rows": 0,
        "train_valid_conversion_value_rows": 0, "validation_valid_conversion_value_rows": 0,
    }


def _create_diagnostic_tables(connection: sqlite3.Connection) -> None:
    connection.executescript("CREATE TABLE users (split TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(split,value)); CREATE TABLE products (split TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(split,value));")


def _record_diagnostic_rows(connection: sqlite3.Connection, split: str, encoded: pd.DataFrame) -> None:
    connection.executemany("INSERT OR IGNORE INTO users VALUES (?, ?)", ((split, str(value)) for value in encoded["user_id"]))
    connection.executemany("INSERT OR IGNORE INTO products VALUES (?, ?)", ((split, str(value)) for value in encoded["candidate_ad_id"]))
    connection.commit()


def _read_diagnostic_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        result[f"{split}_unique_users"] = int(connection.execute("SELECT COUNT(*) FROM users WHERE split=?", (split,)).fetchone()[0])
        result[f"{split}_unique_products"] = int(connection.execute("SELECT COUNT(*) FROM products WHERE split=?", (split,)).fetchone()[0])
    return result


def _shortfall_reason(maximum: int, train_rows: int, diagnostics: Mapping[str, Any]) -> str:
    if train_rows >= maximum:
        return "reached max_train_rows limit from observed click interactions"
    available = int(diagnostics["filter_rows"]["train_rows_before_max_train_rows"])
    return f"only {available} observed click interactions were assigned to train after required-field and validation split filters; no coarse-candidate overlap filter is applied"


def encode_feature_frame(frame: pd.DataFrame, *, bucket_sizes: Sequence[int] = DEFAULT_BUCKET_SIZES, random_seed: int = 2026, hash_cache: SparseHashCache | None = None) -> pd.DataFrame:
    """Encode safe model features; labels are preserved solely as training targets."""
    assert_no_fine_rank_leakage(DENSE_FEATURES + SPARSE_FEATURES)
    output = pd.DataFrame(index=frame.index)
    output["user_id"] = frame["user_id"].astype(str)
    output["candidate_ad_id"] = frame["candidate_ad_id"].astype(str)
    price = _numeric(frame, "product_price")
    clicks = _numeric(frame, "clicks_last_7d")
    timestamp = _numeric(frame, "click_timestamp")
    rank = _numeric(frame, "rank", default=0.0)
    log_price = _bounded_positive_log(price)
    log_clicks = _bounded_positive_log(clicks)
    dense = {
        "rrf_score": _bounded_signed_log(_numeric(frame, "rrf_score")), "source_count": _bounded_positive_log(_numeric(frame, "source_count")), "coarse_score": _bounded_signed_log(_numeric(frame, "coarse_score")),
        "inverse_coarse_rank": np.clip(np.divide(1.0, np.maximum(rank, 1.0)), 0.0, 1.0), "product_price": log_price / 5.0,
        "log_product_price": log_price, "clicks_last_7d": log_clicks / 5.0,
        "log_clicks_last_7d": log_clicks,
        "click_hour_utc": np.where(np.isfinite(timestamp), (timestamp // 3600 % 24) / 23.0, 0.0),
        "click_day_of_week_utc": np.where(np.isfinite(timestamp), (timestamp // 86400 % 7) / 6.0, 0.0),
        "product_price_missing": (~np.isfinite(price)).astype(np.float32), "clicks_last_7d_missing": (~np.isfinite(clicks)).astype(np.float32), "timestamp_missing": (~np.isfinite(timestamp)).astype(np.float32),
    }
    for name, values in dense.items():
        output[f"dense__{name}"] = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), -MAX_DENSE_ABS_VALUE, MAX_DENSE_ABS_VALUE).astype(np.float32)
    for index, (name, bucket) in enumerate(zip(SPARSE_FEATURES, bucket_sizes)):
        source = output["candidate_ad_id"] if name == "product_id" else output["user_id"] if name == "user_id" else frame.get(name, pd.Series(None, index=frame.index))
        output[f"sparse__{name}"] = hash_cache.encode(source, index) if hash_cache is not None else _hash_series(source, int(bucket), random_seed + index)
    label = pd.to_numeric(frame.get("conversion_label", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0).astype(np.float32)
    value = pd.to_numeric(frame.get("conversion_value_eur", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(np.float64)
    finite_nonnegative_value = np.isfinite(value) & (value >= 0)
    output["conversion_label"] = label
    # Invalid/missing source values remain missing (never synthetic zero); the
    # diagnostics record their count and the value-loss mask stays disabled.
    output["conversion_value_eur"] = value.where(finite_nonnegative_value, np.nan)
    output["value_mask"] = ((label == 1) & finite_nonnegative_value).astype(np.float32)
    log_value = np.full(len(output), np.nan, dtype=np.float64)
    valid_value_mask = output["value_mask"].to_numpy(dtype=bool)
    log_value[valid_value_mask] = np.log1p(value.to_numpy(dtype=np.float64)[valid_value_mask])
    output["log_conversion_value"] = log_value
    return output.reset_index(drop=True)


class FineRankParquetDataset(IterableDataset[dict[str, Any]]):
    """Worker-sharded parquet reader; each worker decodes only its own shards."""

    def __init__(self, directory: Path, *, value_transform: Mapping[str, float] | None = None, include_identifiers: bool = False) -> None:
        self.parts = sorted(directory.glob("part-*.parquet"))
        self.value_transform = dict(value_transform or {"mean": 0.0, "std": 1.0})
        self.include_identifiers = include_identifiers
        if not self.parts:
            raise FileNotFoundError(f"No fine-rank parquet shards found in {directory}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        parts = self.parts if info is None else self.parts[info.id :: info.num_workers]
        columns = [f"dense__{name}" for name in DENSE_FEATURES] + [f"sparse__{name}" for name in SPARSE_FEATURES] + ["conversion_label", "conversion_value_eur", "log_conversion_value", "value_mask"]
        if self.include_identifiers:
            columns.extend(("user_id", "candidate_ad_id"))
        for part in parts:
            for batch in pq.ParquetFile(part).iter_batches(batch_size=32_768, columns=columns):
                data = batch.to_pydict()
                length = len(data["conversion_label"])
                for row in range(length):
                    mask = np.float32(data["value_mask"][row])
                    log_value = data["log_conversion_value"][row]
                    normalized_target = 0.0 if not mask or log_value is None else (float(log_value) - float(self.value_transform["mean"])) / float(self.value_transform["std"])
                    result: dict[str, Any] = {
                        "dense": np.asarray([data[f"dense__{name}"][row] for name in DENSE_FEATURES], dtype=np.float32),
                        "sparse": np.asarray([data[f"sparse__{name}"][row] for name in SPARSE_FEATURES], dtype=np.int64),
                        "label": np.float32(data["conversion_label"][row]),
                        "value": np.float32(normalized_target),
                        "observed_value": np.float64(data["conversion_value_eur"][row] if data["conversion_value_eur"][row] is not None else 0.0),
                        "value_mask": mask,
                    }
                    if self.include_identifiers:
                        result.update(user_id=data["user_id"][row], candidate_ad_id=data["candidate_ad_id"][row], coarse_score=np.float32(data["dense__coarse_score"][row]))
                    yield result


def _build_index(spec: FineRankDatasetSpec) -> None:
    if spec.index_path.exists():
        spec.index_path.unlink()
    connection = sqlite3.connect(spec.index_path)
    try:
        connection.executescript("""
            CREATE TABLE products (product_id TEXT PRIMARY KEY, product_price REAL, product_age_group TEXT, product_gender TEXT, product_brand TEXT, product_category_1 TEXT, product_category_2 TEXT, product_category_3 TEXT, product_country TEXT);
            CREATE TABLE users (user_id TEXT PRIMARY KEY, clicks_last_7d REAL, device_type TEXT, click_timestamp REAL);
        """)
        _index_features(connection, spec.feature_source_path, spec.chunk_size)
        connection.commit()
    finally:
        connection.close()


def _index_features(connection: sqlite3.Connection, path: Path, chunk_size: int) -> None:
    for chunk in _iter_csv_chunks(path, chunk_size):
        _require_interaction_columns(chunk)
        for row in chunk.itertuples(index=False):
            values = row._asdict()
            product = _normalise_id(values.get("product_id")); user = _normalise_id(values.get("user_id"))
            timestamp = _finite_or_none(values.get("click_timestamp"))
            if product:
                connection.execute("""INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(product_id) DO UPDATE SET
                    product_price=COALESCE(excluded.product_price,products.product_price), product_age_group=COALESCE(excluded.product_age_group,products.product_age_group), product_gender=COALESCE(excluded.product_gender,products.product_gender), product_brand=COALESCE(excluded.product_brand,products.product_brand), product_category_1=COALESCE(excluded.product_category_1,products.product_category_1), product_category_2=COALESCE(excluded.product_category_2,products.product_category_2), product_category_3=COALESCE(excluded.product_category_3,products.product_category_3), product_country=COALESCE(excluded.product_country,products.product_country)""", (product, _finite_or_none(values.get("product_price")), _normalise_id(values.get("product_age_group")), _normalise_id(values.get("product_gender")), _normalise_id(values.get("product_brand")), _normalise_id(values.get("product_category_1")), _normalise_id(values.get("product_category_2")), _normalise_id(values.get("product_category_3")), _normalise_id(values.get("product_country"))))
            if user:
                connection.execute("""INSERT INTO users VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET
                    clicks_last_7d=CASE WHEN excluded.click_timestamp >= users.click_timestamp THEN excluded.clicks_last_7d ELSE users.clicks_last_7d END,
                    device_type=CASE WHEN excluded.click_timestamp >= users.click_timestamp THEN excluded.device_type ELSE users.device_type END,
                    click_timestamp=MAX(users.click_timestamp, excluded.click_timestamp)""", (user, _finite_or_none(values.get("clicks_last_7d")), _normalise_id(values.get("device_type")), timestamp if timestamp is not None else -1.0))
        connection.commit()


def _append_and_flush(frame: pd.DataFrame, split: str, buffers: dict[str, list[pd.DataFrame]], count: int, part: int, spec: FineRankDatasetSpec) -> tuple[int, int]:
    buffers[split].append(frame)
    count += len(frame)
    if sum(len(item) for item in buffers[split]) >= spec.chunk_size:
        destination = spec.cache_dir if split == "train" else spec.validation_dir
        part = _write_part(pd.concat(buffers[split], ignore_index=True), destination, part)
        buffers[split] = []
    return count, part


def _write_part(frame: pd.DataFrame, directory: Path, part: int) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"part-{part:05d}.parquet", index=False, engine="pyarrow")
    return part + 1


def _clear_parts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for part in directory.glob("part-*.parquet"):
        part.unlink()


def _iter_csv_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    if path.is_dir():
        yield from iter_csv_parts(path, chunk_size)
    elif path.is_file():
        yield from pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")


def _require_candidate_columns(frame: pd.DataFrame) -> None:
    missing = {"user_id", "candidate_ad_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"Fine-rank candidate file is missing columns: {sorted(missing)}")


def _require_interaction_columns(frame: pd.DataFrame) -> None:
    missing = {"user_id", "product_id", "conversion_label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Fine-rank interaction source is missing columns: {sorted(missing)}")


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _bounded_positive_log(values: np.ndarray) -> np.ndarray:
    """Represent unbounded non-negative counts/prices in a bounded log space."""
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.log1p(np.maximum(safe, 0.0)), 0.0, MAX_DENSE_ABS_VALUE)


def _bounded_signed_log(values: np.ndarray) -> np.ndarray:
    """Preserve score ordering while preventing a finite outlier from exploding CrossNet."""
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.sign(safe) * np.log1p(np.abs(safe)), -MAX_DENSE_ABS_VALUE, MAX_DENSE_ABS_VALUE)


def _hash_series(values: pd.Series, bucket_size: int, seed: int) -> np.ndarray:
    normalized = _normalise_id_series(values).fillna("__MISSING__")
    codes, unique = pd.factorize(normalized, sort=False)
    hashed = np.fromiter((stable_hash(value, bucket_size, seed) for value in unique), dtype=np.int64, count=len(unique))
    return hashed[codes]


def _normalise_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalise_id_series(values: pd.Series) -> pd.Series:
    """Fast vectorized ID normalization used by inference joins."""
    result = values.astype("string").str.strip()
    return result.mask(result.isin(("", "nan", "None", "<NA>")), pd.NA)


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _config_hash(spec: FineRankDatasetSpec) -> str:
    values: Mapping[str, Any] = {"candidate_path": str(spec.candidate_path), "feature_source_path": str(spec.feature_source_path), "train_label_path": str(spec.train_label_path), "validation_label_path": str(spec.validation_label_path), "mode": spec.mode, "max_train_rows": spec.max_train_rows, "chunk_size": spec.chunk_size, "validation_fraction": spec.validation_fraction, "random_seed": spec.random_seed, "bucket_sizes": list(spec.bucket_sizes), "feature_version": FEATURE_VERSION}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()
