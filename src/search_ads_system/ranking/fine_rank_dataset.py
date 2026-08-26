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

FEATURE_VERSION = "fine-rank-v1"
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
    """Persistent, bounded-query SQLite lookup for Past/full product and user features."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def close(self) -> None:
        self.connection.close()

    def enrich(self, candidates: pd.DataFrame) -> pd.DataFrame:
        frame = candidates.copy()
        frame["user_id"] = frame["user_id"].map(_normalise_id)
        frame["candidate_ad_id"] = frame["candidate_ad_id"].map(_normalise_id)
        product_map = self._lookup("products", "product_id", frame["candidate_ad_id"].dropna().unique().tolist(), PRODUCT_COLUMNS)
        user_map = self._lookup("users", "user_id", frame["user_id"].dropna().unique().tolist(), USER_COLUMNS)
        for index, candidate_id in frame["candidate_ad_id"].items():
            values = product_map.get(candidate_id, ())
            for offset, column in enumerate(PRODUCT_COLUMNS):
                frame.loc[index, column] = values[offset] if values else None
        for index, user_id in frame["user_id"].items():
            values = user_map.get(user_id, ())
            for offset, column in enumerate(USER_COLUMNS):
                frame.loc[index, column] = values[offset] if values else None
        return frame

    def labels_for(self, pairs: pd.DataFrame, table: str) -> pd.DataFrame:
        if table not in {"train_labels", "validation_labels"}:
            raise ValueError("unsupported label table")
        frame = pairs.copy()
        frame["user_id"] = frame["user_id"].map(_normalise_id)
        frame["candidate_ad_id"] = frame["candidate_ad_id"].map(_normalise_id)
        keys = [(user, ad) for user, ad in zip(frame["user_id"], frame["candidate_ad_id"]) if user and ad]
        found: dict[tuple[str, str], tuple[int, float | None]] = {}
        for start in range(0, len(keys), 400):
            subset = keys[start : start + 400]
            clauses = " OR ".join("(user_id=? AND product_id=?)" for _ in subset)
            query = f"SELECT user_id, product_id, conversion_label, conversion_value_eur FROM {table} WHERE {clauses}"
            for user, product, label, value in self.connection.execute(query, [item for pair in subset for item in pair]):
                found[(user, product)] = (int(label), None if value is None else float(value))
        labels = [found.get((user, ad)) for user, ad in zip(frame["user_id"], frame["candidate_ad_id"])]
        frame["conversion_label"] = [item[0] if item is not None else np.nan for item in labels]
        frame["conversion_value_eur"] = [item[1] if item is not None else np.nan for item in labels]
        return frame.loc[frame["conversion_label"].notna()].copy()

    def _lookup(self, table: str, key: str, values: list[str], columns: Sequence[str]) -> dict[str, tuple[Any, ...]]:
        result: dict[str, tuple[Any, ...]] = {}
        for start in range(0, len(values), 900):
            subset = values[start : start + 900]
            if not subset:
                continue
            marks = ",".join("?" for _ in subset)
            query = f"SELECT {key}, {', '.join(columns)} FROM {table} WHERE {key} IN ({marks})"
            for row in self.connection.execute(query, subset):
                result[str(row[0])] = tuple(row[1:])
        return result


def cache_matches(spec: FineRankDatasetSpec) -> bool:
    if not spec.metadata_path.is_file() or not any(spec.cache_dir.glob("part-*.parquet")):
        return False
    try:
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("config_hash") == _config_hash(spec) and metadata.get("feature_version") == FEATURE_VERSION


def build_or_reuse_cached_datasets(spec: FineRankDatasetSpec) -> dict[str, Any]:
    """Build deterministic parquet shards, or reuse an exactly matching cache."""
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
    try:
        for candidates in _iter_csv_chunks(spec.candidate_path, spec.chunk_size):
            _require_candidate_columns(candidates)
            candidates = candidates.loc[:, [column for column in ("user_id", "candidate_ad_id", "rrf_score", "source_count", "coarse_score", "rank") if column in candidates]].copy()
            labeled = store.labels_for(candidates, "train_labels")
            if labeled.empty:
                continue
            enriched = store.enrich(labeled)
            encoded = encode_feature_frame(enriched, bucket_sizes=spec.bucket_sizes, random_seed=spec.random_seed)
            if spec.validation_label_path is not None:
                # Temporal validation labels must be read from Future-B and never
                # share samples with Future-A training.
                validation = store.labels_for(candidates, "validation_labels")
                if not validation.empty:
                    validation_encoded = encode_feature_frame(store.enrich(validation), bucket_sizes=spec.bucket_sizes, random_seed=spec.random_seed)
                    counts["validation"], parts["validation"] = _append_and_flush(validation_encoded, "validation", buffers, counts["validation"], parts["validation"], spec)
            else:
                key = encoded["user_id"].astype(str) + "\x1f" + encoded["candidate_ad_id"].astype(str)
                is_validation = key.map(lambda value: stable_hash(value, 10_000, spec.random_seed) < int(spec.validation_fraction * 10_000))
                validation_encoded, encoded = encoded.loc[is_validation].copy(), encoded.loc[~is_validation].copy()
                if not validation_encoded.empty:
                    counts["validation"], parts["validation"] = _append_and_flush(validation_encoded, "validation", buffers, counts["validation"], parts["validation"], spec)
            if not encoded.empty and counts["train"] < spec.max_train_rows:
                encoded = encoded.iloc[: spec.max_train_rows - counts["train"]].copy()
                counts["train"], parts["train"] = _append_and_flush(encoded, "train", buffers, counts["train"], parts["train"], spec)
            if counts["train"] >= spec.max_train_rows:
                break
        for split in ("train", "validation"):
            if buffers[split]:
                destination = spec.cache_dir if split == "train" else spec.validation_dir
                parts[split] = _write_part(pd.concat(buffers[split], ignore_index=True), destination, parts[split])
        if counts["train"] == 0:
            raise ValueError("No click-conditioned coarse-candidate/interaction overlap available for fine-rank training")
        metadata = {
            "feature_version": FEATURE_VERSION,
            "feature_schema": {"dense": list(DENSE_FEATURES), "sparse": list(SPARSE_FEATURES), "bucket_sizes": list(spec.bucket_sizes)},
            "source_paths": {"candidates": str(spec.candidate_path), "features": str(spec.feature_source_path), "train_labels": str(spec.train_label_path), "validation_labels": str(spec.validation_label_path) if spec.validation_label_path else None},
            "label_definition": "click-conditioned conversion_label; value target only for conversion_label=1 and non-missing conversion_value_eur",
            "split_definition": "Future-A/Future-B" if spec.validation_label_path else f"stable {spec.validation_fraction:.0%} pair validation hash",
            "mode": spec.mode,
            "max_train_rows": spec.max_train_rows,
            "row_count": counts["train"], "validation_row_count": counts["validation"],
            "part_count": parts, "config_hash": _config_hash(spec),
        }
        spec.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Fine-rank cache built: train=%s validation=%s (requested max=%s)", counts["train"], counts["validation"], spec.max_train_rows)
        return metadata
    finally:
        store.close()


def encode_feature_frame(frame: pd.DataFrame, *, bucket_sizes: Sequence[int] = DEFAULT_BUCKET_SIZES, random_seed: int = 2026) -> pd.DataFrame:
    """Encode safe model features; labels are preserved solely as training targets."""
    assert_no_fine_rank_leakage(DENSE_FEATURES + SPARSE_FEATURES)
    output = pd.DataFrame(index=frame.index)
    output["user_id"] = frame["user_id"].astype(str)
    output["candidate_ad_id"] = frame["candidate_ad_id"].astype(str)
    price = _numeric(frame, "product_price")
    clicks = _numeric(frame, "clicks_last_7d")
    timestamp = _numeric(frame, "click_timestamp")
    rank = _numeric(frame, "rank", default=0.0)
    dense = {
        "rrf_score": _numeric(frame, "rrf_score"), "source_count": _numeric(frame, "source_count"), "coarse_score": _numeric(frame, "coarse_score"),
        "inverse_coarse_rank": np.divide(1.0, np.maximum(rank, 1.0)), "product_price": np.nan_to_num(price, nan=0.0),
        "log_product_price": np.log1p(np.maximum(np.nan_to_num(price, nan=0.0), 0.0)), "clicks_last_7d": np.nan_to_num(clicks, nan=0.0),
        "log_clicks_last_7d": np.log1p(np.maximum(np.nan_to_num(clicks, nan=0.0), 0.0)),
        "click_hour_utc": np.where(np.isfinite(timestamp), (timestamp // 3600 % 24) / 23.0, 0.0),
        "click_day_of_week_utc": np.where(np.isfinite(timestamp), (timestamp // 86400 % 7) / 6.0, 0.0),
        "product_price_missing": (~np.isfinite(price)).astype(np.float32), "clicks_last_7d_missing": (~np.isfinite(clicks)).astype(np.float32), "timestamp_missing": (~np.isfinite(timestamp)).astype(np.float32),
    }
    for name, values in dense.items():
        output[f"dense__{name}"] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for index, (name, bucket) in enumerate(zip(SPARSE_FEATURES, bucket_sizes)):
        source = output["candidate_ad_id"] if name == "product_id" else output["user_id"] if name == "user_id" else frame.get(name, pd.Series(None, index=frame.index))
        output[f"sparse__{name}"] = [stable_hash(value, int(bucket), random_seed + index) for value in source]
    label = pd.to_numeric(frame.get("conversion_label", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0).astype(np.float32)
    value = pd.to_numeric(frame.get("conversion_value_eur", pd.Series(np.nan, index=frame.index)), errors="coerce")
    output["conversion_label"] = label
    output["conversion_value_eur"] = value.astype(np.float32)
    output["value_mask"] = ((label == 1) & value.notna()).astype(np.float32)
    return output.reset_index(drop=True)


class FineRankParquetDataset(IterableDataset[dict[str, Any]]):
    """Worker-sharded parquet reader; each worker decodes only its own shards."""

    def __init__(self, directory: Path, *, include_identifiers: bool = False) -> None:
        self.parts = sorted(directory.glob("part-*.parquet"))
        self.include_identifiers = include_identifiers
        if not self.parts:
            raise FileNotFoundError(f"No fine-rank parquet shards found in {directory}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        parts = self.parts if info is None else self.parts[info.id :: info.num_workers]
        columns = [f"dense__{name}" for name in DENSE_FEATURES] + [f"sparse__{name}" for name in SPARSE_FEATURES] + ["conversion_label", "conversion_value_eur", "value_mask"]
        if self.include_identifiers:
            columns.extend(("user_id", "candidate_ad_id"))
        for part in parts:
            for batch in pq.ParquetFile(part).iter_batches(batch_size=32_768, columns=columns):
                data = batch.to_pydict()
                length = len(data["conversion_label"])
                for row in range(length):
                    result: dict[str, Any] = {
                        "dense": np.asarray([data[f"dense__{name}"][row] for name in DENSE_FEATURES], dtype=np.float32),
                        "sparse": np.asarray([data[f"sparse__{name}"][row] for name in SPARSE_FEATURES], dtype=np.int64),
                        "label": np.float32(data["conversion_label"][row]),
                        "value": np.float32(data["conversion_value_eur"][row] if data["conversion_value_eur"][row] is not None else 0.0),
                        "value_mask": np.float32(data["value_mask"][row]),
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
            CREATE TABLE train_labels (user_id TEXT NOT NULL, product_id TEXT NOT NULL, conversion_label INTEGER NOT NULL, conversion_value_eur REAL, PRIMARY KEY (user_id, product_id));
            CREATE TABLE validation_labels (user_id TEXT NOT NULL, product_id TEXT NOT NULL, conversion_label INTEGER NOT NULL, conversion_value_eur REAL, PRIMARY KEY (user_id, product_id));
        """)
        _index_features(connection, spec.feature_source_path, spec.chunk_size)
        _index_labels(connection, spec.train_label_path, "train_labels", spec.chunk_size)
        if spec.validation_label_path is not None:
            _index_labels(connection, spec.validation_label_path, "validation_labels", spec.chunk_size)
        connection.execute("CREATE INDEX train_pair_index ON train_labels(user_id, product_id)")
        connection.execute("CREATE INDEX validation_pair_index ON validation_labels(user_id, product_id)")
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


def _index_labels(connection: sqlite3.Connection, path: Path, table: str, chunk_size: int) -> None:
    for chunk in _iter_csv_chunks(path, chunk_size):
        _require_interaction_columns(chunk)
        rows = []
        for user, product, label, value in chunk[["user_id", "product_id", "conversion_label", "conversion_value_eur"]].itertuples(index=False, name=None):
            user_id, product_id = _normalise_id(user), _normalise_id(product)
            parsed_label = _finite_or_none(label)
            if user_id and product_id and parsed_label is not None:
                rows.append((user_id, product_id, int(parsed_label == 1), _finite_or_none(value) if int(parsed_label == 1) else None))
        connection.executemany(f"""INSERT INTO {table} VALUES (?, ?, ?, ?) ON CONFLICT(user_id,product_id) DO UPDATE SET
            conversion_label=MAX({table}.conversion_label,excluded.conversion_label),
            conversion_value_eur=CASE WHEN excluded.conversion_label=1 AND excluded.conversion_value_eur IS NOT NULL THEN excluded.conversion_value_eur ELSE {table}.conversion_value_eur END""", rows)
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


def _normalise_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _config_hash(spec: FineRankDatasetSpec) -> str:
    values: Mapping[str, Any] = {"candidate_path": str(spec.candidate_path), "feature_source_path": str(spec.feature_source_path), "train_label_path": str(spec.train_label_path), "validation_label_path": str(spec.validation_label_path), "mode": spec.mode, "max_train_rows": spec.max_train_rows, "chunk_size": spec.chunk_size, "validation_fraction": spec.validation_fraction, "random_seed": spec.random_seed, "bucket_sizes": list(spec.bucket_sizes), "feature_version": FEATURE_VERSION}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()
