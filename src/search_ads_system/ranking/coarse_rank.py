"""Memory-bounded coarse ranking for fused recall candidates.

The unified Criteo rows are click interactions, including when
``conversion_label`` is zero.  Therefore a candidate is a positive when its
``(user_id, product_id)`` pair occurs in those interactions; conversion is a
sample weight only.  In particular, conversion fields never enter the model
feature matrix.

Candidate files have no request/candidate-generation timestamp.  The
timestamp split implemented here is consequently a reproducible offline
baseline, rather than a strict causal serving-time split.  This limitation is
logged whenever a model is trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import pickle
import sqlite3
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.data.storage import iter_csv_parts

LOGGER = logging.getLogger(__name__)

RECALL_FEATURE_COLUMNS = ("rrf_score", "source_count")
NUMERIC_AD_FEATURE_COLUMNS = ("product_price", "clicks_last_7d")
CATEGORICAL_AD_FEATURE_COLUMNS = (
    "product_age_group",
    "device_type",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_country",
)
FEATURE_COLUMNS = RECALL_FEATURE_COLUMNS + NUMERIC_AD_FEATURE_COLUMNS + tuple(
    f"{column}__hash" for column in CATEGORICAL_AD_FEATURE_COLUMNS
)
LEAKAGE_COLUMNS = {
    "conversion_label",
    "conversion_value_eur",
    "conversion_delay_seconds",
    "click_timestamp",
    "event_id",
    "source_row_number",
    "user_id",
    "candidate_ad_id",
    "product_id",
}
OUTPUT_COLUMNS = ("user_id", "candidate_ad_id", "coarse_score", "rank")
_MISSING_CATEGORY = "__MISSING__"
DEFAULT_FEATURES = (0.0, 0.0) + (_MISSING_CATEGORY,) * len(CATEGORICAL_AD_FEATURE_COLUMNS)
_SQLITE_QUERY_BATCH_SIZE = 900

# Process-worker state.  Each worker owns a read-only SQLite connection and
# bounded feature cache; neither is copied from the parent process.
_WORKER_MODEL: CoarseRankModel | None = None
_WORKER_STORE: SQLiteFeatureStore | None = None
_WORKER_TOP_K: int | None = None
_WORKER_THREAD_CONTROLLER: Any | None = None


@dataclass(frozen=True)
class CoarseRankConfig:
    """Configuration for a CPU-only, bounded-memory coarse rank run."""

    input_path: Path
    interaction_path: Path
    output_path: Path
    model_path: Path
    train: bool = True
    max_train_rows: int = 2_000_000
    max_users: int = 500_000
    top_k: int = 50
    chunk_size: int = 200_000
    random_seed: int = 2026
    negatives_per_positive: int = 5
    conversion_sample_weight: float = 3.0
    feature_cache_size: int = 100_000
    model_type: str = "hist_gbdt"
    # Programmatic callers stay single-process unless they opt in; YAML
    # production defaults select up to 12 workers below.
    num_workers: int = 1
    inference_batch_users: int = 1_000
    inference_batch_candidates: int = 100_000
    prefetch_batches: int = 2
    enable_gpu_if_available: bool = False


@dataclass
class CoarseRankModel:
    """Persisted model payload; feature construction is intentionally stateless."""

    estimator: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    model_name: str = "sklearn.HistGradientBoostingClassifier"
    model_type: str = "hist_gbdt"


@dataclass(frozen=True)
class TrainingData:
    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    timestamps: np.ndarray
    group_keys: np.ndarray


def parse_coarse_rank_config(raw_config: Mapping[str, Any], config_path: Path) -> CoarseRankConfig:
    """Parse ``coarse_rank`` from YAML and resolve project-relative paths."""

    paths = raw_config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Configuration must define a paths mapping")
    options = raw_config.get("coarse_rank", {})
    if not isinstance(options, Mapping):
        raise ValueError("coarse_rank configuration must be a mapping")
    root = config_path.parent.resolve()
    config = CoarseRankConfig(
        input_path=resolve_path(str(options.get("input_path", "outputs/recall_candidates/fused_candidates.csv")), root),
        interaction_path=resolve_path(str(options.get("interaction_path", paths.get("unified_data", "outputs/processed/criteo_unified"))), root),
        output_path=resolve_path(str(options.get("output_path", "outputs/ranking/coarse_rank_topk.csv")), root),
        model_path=resolve_path(str(options.get("model_path", "outputs/models/coarse_rank_model.pkl")), root),
        train=bool(options.get("train", True)),
        max_train_rows=int(options.get("max_train_rows", 2_000_000)),
        max_users=int(options.get("max_users", 500_000)),
        top_k=int(options.get("top_k", 50)),
        chunk_size=int(options.get("chunk_size", 200_000)),
        random_seed=int(options.get("random_seed", raw_config.get("project", {}).get("seed", 2026))),
        negatives_per_positive=int(options.get("negative_sampling", {}).get("negatives_per_positive", 5)),
        conversion_sample_weight=float(options.get("conversion_sample_weight", 3.0)),
        feature_cache_size=int(options.get("feature_cache_size", 100_000)),
        model_type=str(options.get("model_type", "hist_gbdt")).lower(),
        num_workers=int(options.get("num_workers", max(1, min(12, os.cpu_count() or 1)))),
        inference_batch_users=int(options.get("inference_batch_users", 1_000)),
        inference_batch_candidates=int(options.get("inference_batch_candidates", 100_000)),
        prefetch_batches=int(options.get("prefetch_batches", 2)),
        enable_gpu_if_available=bool(options.get("enable_gpu_if_available", False)),
    )
    _validate_config(config)
    output_root = resolve_path(str(paths.get("outputs_dir", "outputs")), root)
    for name, path in (("output_path", config.output_path), ("model_path", config.model_path)):
        try:
            path.relative_to(output_root)
        except ValueError as error:
            raise ValueError(f"coarse_rank.{name} must be within paths.outputs_dir") from error
    return config


def preprocess_features(frame: pd.DataFrame) -> np.ndarray:
    """Build safe numeric features without IDs, labels, or conversion outcomes.

    Categorical values use a deterministic hash rather than a fitted one-hot
    vocabulary, which keeps both model size and serving memory bounded.
    """

    assert_no_leakage_features(FEATURE_COLUMNS)
    result = np.zeros((len(frame), len(FEATURE_COLUMNS)), dtype=np.float32)
    for index, column in enumerate(RECALL_FEATURE_COLUMNS + NUMERIC_AD_FEATURE_COLUMNS):
        values = pd.to_numeric(frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce")
        result[:, index] = np.nan_to_num(values.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    base_index = len(RECALL_FEATURE_COLUMNS + NUMERIC_AD_FEATURE_COLUMNS)
    for offset, column in enumerate(CATEGORICAL_AD_FEATURE_COLUMNS):
        values = frame.get(column, pd.Series(_MISSING_CATEGORY, index=frame.index)).astype("string").fillna(_MISSING_CATEGORY)
        result[:, base_index + offset] = np.fromiter(
            (_stable_hash(str(value), seed=0) % 1_000_003 / 1_000_003 for value in values),
            dtype=np.float32,
            count=len(values),
        )
    return result


def assert_no_leakage_features(feature_columns: Sequence[str]) -> None:
    """Reject labels, post-conversion data, and identifier fields as features."""

    leaked = set(feature_columns) & LEAKAGE_COLUMNS
    if leaked:
        raise ValueError(f"Coarse-rank feature list contains leakage/ID columns: {sorted(leaked)}")


def deterministic_negative_sample(
    candidates: pd.DataFrame, *, user_id: str, count: int, random_seed: int
) -> pd.DataFrame:
    """Choose negatives by stable hash, independent of CSV chunk boundaries."""

    if count <= 0 or candidates.empty:
        return candidates.iloc[0:0].copy()
    ranked = candidates.assign(
        _sample_key=[_stable_hash(f"{user_id}\x1f{candidate}", random_seed) for candidate in candidates["candidate_ad_id"]]
    ).sort_values(["_sample_key", "candidate_ad_id"], kind="mergesort")
    return ranked.iloc[:count].drop(columns="_sample_key")


def build_labeled_user_examples(
    candidates: pd.DataFrame,
    *,
    user_id: str,
    feature_store: "SQLiteFeatureStore",
    negatives_per_positive: int,
    random_seed: int,
    conversion_sample_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Create click positives plus deterministic sampled candidate negatives."""

    if candidates.empty:
        return _empty_examples()
    rows = candidates.copy()
    rows["candidate_ad_id"] = rows["candidate_ad_id"].map(_normalise_id)
    rows = rows.dropna(subset=["candidate_ad_id"])
    interactions = feature_store.interactions_for(rows["candidate_ad_id"].tolist(), user_id)
    rows["_interaction"] = rows["candidate_ad_id"].map(interactions)
    positive_mask = rows["_interaction"].notna()
    positives = rows.loc[positive_mask].copy()
    if positives.empty:
        return _empty_examples()
    negatives = deterministic_negative_sample(
        rows.loc[~positive_mask],
        user_id=user_id,
        count=len(positives) * negatives_per_positive,
        random_seed=random_seed,
    )
    selected = pd.concat((positives, negatives), ignore_index=True)
    selected = feature_store.enrich(selected)
    labels = selected["_interaction"].notna().to_numpy(dtype=np.int8)
    weights = np.ones(len(selected), dtype=np.float32)
    for row_index, interaction in enumerate(selected["_interaction"]):
        if isinstance(interaction, tuple) and int(interaction[1]) == 1:
            weights[row_index] = conversion_sample_weight
    timestamps = [entry[0] for entry in positives["_interaction"] if entry is not None and entry[0] is not None]
    group_timestamp = int(max(timestamps)) if timestamps else -1
    return preprocess_features(selected), labels, weights, group_timestamp


def collect_training_data(config: CoarseRankConfig, feature_store: "SQLiteFeatureStore") -> TrainingData:
    """Stream candidates and retain at most ``max_train_rows`` sampled examples."""

    feature_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []
    timestamps_parts: list[np.ndarray] = []
    keys_parts: list[np.ndarray] = []
    selected_users = 0
    rows_collected = 0
    candidate_rows = 0
    for user_id, candidates in iter_candidate_groups(config.input_path, config.chunk_size):
        candidate_rows += len(candidates)
        if selected_users >= config.max_users:
            break
        features, labels, weights, timestamp = build_labeled_user_examples(
            candidates,
            user_id=user_id,
            feature_store=feature_store,
            negatives_per_positive=config.negatives_per_positive,
            random_seed=config.random_seed,
            conversion_sample_weight=config.conversion_sample_weight,
        )
        selected_users += 1
        if len(labels) == 0 or rows_collected + len(labels) > config.max_train_rows:
            continue
        feature_parts.append(features)
        labels_parts.append(labels)
        weights_parts.append(weights)
        timestamps_parts.append(np.full(len(labels), timestamp, dtype=np.int64))
        keys_parts.append(np.full(len(labels), _stable_hash(user_id, config.random_seed), dtype=np.uint64))
        rows_collected += len(labels)
        if selected_users % 10_000 == 0:
            LOGGER.info("processed_candidates=%s selected_users=%s train_rows=%s", candidate_rows, selected_users, rows_collected)
        if rows_collected >= config.max_train_rows:
            break
    if not feature_parts:
        raise ValueError("No labeled training examples: no recalled candidate matched a real click interaction")
    data = TrainingData(
        features=np.concatenate(feature_parts),
        labels=np.concatenate(labels_parts),
        sample_weights=np.concatenate(weights_parts),
        timestamps=np.concatenate(timestamps_parts),
        group_keys=np.concatenate(keys_parts),
    )
    if len(np.unique(data.labels)) != 2:
        raise ValueError("Coarse ranking requires both click-positive and sampled-negative examples")
    LOGGER.info("Collected training rows=%s positives=%s", len(data.labels), int(data.labels.sum()))
    return data


def split_train_validation(timestamps: np.ndarray, group_keys: np.ndarray, random_seed: int) -> np.ndarray:
    """Return a deterministic 20%% validation mask, preferring timestamp order."""

    valid_timestamps = timestamps[timestamps >= 0]
    if len(valid_timestamps) and np.unique(valid_timestamps).size > 1:
        cutoff = int(np.quantile(valid_timestamps, 0.8))
        mask = timestamps >= cutoff
        if mask.any() and (~mask).any():
            return mask
    # Candidate files lack a request timestamp; this is the reproducible fallback.
    return np.fromiter(
        ((_stable_hash(str(int(key)), random_seed + 17) % 10) >= 8 for key in group_keys),
        dtype=bool,
        count=len(group_keys),
    )


def train_coarse_ranker(data: TrainingData, config: CoarseRankConfig) -> tuple[CoarseRankModel, dict[str, float], set[int]]:
    """Fit the lightweight CPU model and return row metrics plus validation groups."""

    validation_mask = split_train_validation(data.timestamps, data.group_keys, config.random_seed)
    if not validation_mask.any() or (~validation_mask).sum() == 0 or len(np.unique(data.labels[validation_mask])) < 2:
        # Small development data can have a timestamp split containing one class.
        validation_mask = np.fromiter(
            ((_stable_hash(str(index), config.random_seed) % 5) == 0 for index in range(len(data.labels))),
            dtype=bool,
            count=len(data.labels),
        )
    if not validation_mask.any() or not (~validation_mask).any() or len(np.unique(data.labels[validation_mask])) < 2:
        raise ValueError("Unable to form a validation set containing click positives and negatives")
    train_rows = int((~validation_mask).sum())
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        # sklearn's internal validation split is not viable for tiny fixtures;
        # production samples are large enough to benefit from it.
        early_stopping=train_rows >= 1_000,
        random_state=config.random_seed,
    )
    estimator.fit(data.features[~validation_mask], data.labels[~validation_mask], sample_weight=data.sample_weights[~validation_mask])
    probabilities = estimator.predict_proba(data.features[validation_mask])[:, 1]
    labels = data.labels[validation_mask]
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    LOGGER.info("Validation ROC-AUC=%.6f PR-AUC=%.6f LogLoss=%.6f", metrics["roc_auc"], metrics["pr_auc"], metrics["logloss"])
    return CoarseRankModel(estimator=estimator), metrics, {int(key) for key in data.group_keys[validation_mask]}


def save_model(model: CoarseRankModel, path: Path) -> None:
    """Atomically persist the trained model checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output_file:
        pickle.dump(model, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_model(path: Path) -> CoarseRankModel:
    """Load and validate a coarse-rank checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"Coarse-rank model does not exist: {path}")
    with path.open("rb") as input_file:
        model = pickle.load(input_file)
    if not isinstance(model, CoarseRankModel) or tuple(model.feature_columns) != FEATURE_COLUMNS:
        raise ValueError("Model checkpoint has an incompatible coarse-rank feature schema")
    assert_no_leakage_features(model.feature_columns)
    return model


def iter_candidate_groups(path: Path, chunk_size: int) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield complete contiguous user groups, including groups split across chunks."""

    try:
        reader = pd.read_csv(path, usecols=["user_id", "candidate_ad_id", "rrf_score", "source_count"], chunksize=chunk_size, low_memory=False)
    except ValueError as error:
        raise ValueError(f"Fused candidates at {path} must include user_id, candidate_ad_id, rrf_score, source_count") from error
    current_user: str | None = None
    rows: list[dict[str, Any]] = []
    completed_users: set[str] = set()
    for chunk in reader:
        for raw_user_id, raw_candidate_id, rrf_score, source_count in chunk.itertuples(index=False, name=None):
            user_id = _normalise_id(raw_user_id)
            candidate_id = _normalise_id(raw_candidate_id)
            if user_id is None or candidate_id is None:
                continue
            if current_user is None:
                current_user = user_id
            elif user_id != current_user:
                if current_user in completed_users:
                    raise ValueError("Fused candidates must keep each user's rows contiguous for streaming Top-K")
                completed_users.add(current_user)
                yield current_user, pd.DataFrame(rows)
                current_user = user_id
                rows = []
            rows.append({"user_id": user_id, "candidate_ad_id": candidate_id, "rrf_score": rrf_score, "source_count": source_count})
    if current_user is not None:
        if current_user in completed_users:
            raise ValueError("Fused candidates must keep each user's rows contiguous for streaming Top-K")
        yield current_user, pd.DataFrame(rows)


class SQLiteFeatureStore:
    """On-disk interaction/feature lookup with a bounded in-process LRU cache."""

    def __init__(self, database_path: Path, cache_size: int = 100_000, *, read_only: bool = False) -> None:
        self.database_path = database_path
        self.connection = (
            sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
            if read_only
            else sqlite3.connect(database_path)
        )
        self.cache_size = cache_size
        self._feature_cache: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
        self._unknown_cache: set[str] = set()
        self.missing_feature_candidates = 0
        self.total_feature_candidates = 0

    def close(self) -> None:
        self.connection.close()

    def interactions_for(self, candidate_ids: Sequence[str], user_id: str) -> dict[str, tuple[int | None, int]]:
        if not candidate_ids:
            return {}
        result: dict[str, tuple[int | None, int]] = {}
        for batch in _batches(list(dict.fromkeys(candidate_ids)), _SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT candidate_ad_id, click_timestamp, conversion_label FROM interactions WHERE user_id = ? AND candidate_ad_id IN ({placeholders})"
            for candidate_id, timestamp, conversion in self.connection.execute(query, [user_id, *batch]):
                result[str(candidate_id)] = (None if timestamp is None else int(timestamp), int(conversion))
        return result

    def enrich(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Attach product features with a per-batch fallback map.

        The local map is deliberately independent of the LRU: an eviction while
        processing a large batch must never turn a valid unknown candidate into
        a KeyError.  Unknown products receive the same deterministic defaults
        as ordinary missing input values and remain eligible for Top-K.
        """

        enriched = candidates.copy()
        candidate_ids = [str(value) for value in enriched["candidate_ad_id"]]
        unique_ids = list(dict.fromkeys(candidate_ids))
        batch_feature_map = {
            candidate_id: self._feature_cache[candidate_id]
            for candidate_id in unique_ids
            if candidate_id in self._feature_cache
        }
        unknown_ids: set[str] = {candidate_id for candidate_id in batch_feature_map if candidate_id in self._unknown_cache}
        missing = [candidate_id for candidate_id in unique_ids if candidate_id not in batch_feature_map]
        for batch in _batches(missing, _SQLITE_QUERY_BATCH_SIZE):
            placeholders = ",".join("?" for _ in batch)
            query = "SELECT candidate_ad_id, product_price, clicks_last_7d, " + ", ".join(CATEGORICAL_AD_FEATURE_COLUMNS) + f" FROM ad_features WHERE candidate_ad_id IN ({placeholders})"
            found = {str(row[0]): tuple(row[1:]) for row in self.connection.execute(query, batch)}
            for candidate_id in batch:
                values = found.get(candidate_id, DEFAULT_FEATURES)
                if candidate_id not in found:
                    unknown_ids.add(candidate_id)
                batch_feature_map[candidate_id] = values
                self._cache_put(candidate_id, values, is_unknown=candidate_id not in found)
        self.total_feature_candidates += len(candidate_ids)
        # Count candidate rows (rather than unique IDs) to make the reported
        # rate representative of inference work.
        self.missing_feature_candidates += sum(1 for candidate_id in candidate_ids if candidate_id in unknown_ids)
        if unknown_ids:
            LOGGER.debug("Feature index misses in batch: %s", len(unknown_ids))
        values = [batch_feature_map[candidate_id] for candidate_id in candidate_ids]
        columns = list(NUMERIC_AD_FEATURE_COLUMNS + CATEGORICAL_AD_FEATURE_COLUMNS)
        feature_frame = pd.DataFrame(values, columns=columns, index=enriched.index)
        return pd.concat((enriched, feature_frame), axis=1)

    def consume_feature_stats(self) -> tuple[int, int]:
        """Return and reset per-process feature lookup counters."""

        result = self.missing_feature_candidates, self.total_feature_candidates
        self.missing_feature_candidates = 0
        self.total_feature_candidates = 0
        return result

    def _cache_put(self, candidate_id: str, values: tuple[Any, ...], *, is_unknown: bool = False) -> None:
        self._feature_cache[candidate_id] = values
        if is_unknown:
            self._unknown_cache.add(candidate_id)
        else:
            self._unknown_cache.discard(candidate_id)
        self._feature_cache.move_to_end(candidate_id)
        if len(self._feature_cache) > self.cache_size:
            evicted_id, _ = self._feature_cache.popitem(last=False)
            self._unknown_cache.discard(evicted_id)


def build_interaction_feature_index(config: CoarseRankConfig, database_path: Path) -> SQLiteFeatureStore:
    """Stream unified CSV parts into a disk index; no global DataFrame/merge is used."""

    if database_path.exists():
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE interactions (user_id TEXT NOT NULL, candidate_ad_id TEXT NOT NULL, click_timestamp INTEGER, conversion_label INTEGER NOT NULL, PRIMARY KEY (user_id, candidate_ad_id))"
        )
        columns = ", ".join(("candidate_ad_id TEXT PRIMARY KEY", "product_price REAL", "clicks_last_7d REAL") + tuple(f"{column} TEXT" for column in CATEGORICAL_AD_FEATURE_COLUMNS))
        connection.execute(f"CREATE TABLE ad_features ({columns})")
        rows_seen = 0
        for chunk in _iter_interaction_chunks(config.interaction_path, config.chunk_size):
            required = {"user_id", "product_id", "conversion_label"}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"Unified interactions are missing required columns: {sorted(missing)}")
            interaction_rows: list[tuple[Any, ...]] = []
            feature_rows: list[tuple[Any, ...]] = []
            for row in chunk.itertuples(index=False):
                values = row._asdict()
                user_id, candidate_id = _normalise_id(values.get("user_id")), _normalise_id(values.get("product_id"))
                if user_id is None or candidate_id is None:
                    continue
                timestamp = _safe_int(values.get("click_timestamp"))
                conversion = 1 if _safe_int(values.get("conversion_label")) == 1 else 0
                interaction_rows.append((user_id, candidate_id, timestamp, conversion))
                feature_rows.append((candidate_id, _safe_float(values.get("product_price")), _safe_float(values.get("clicks_last_7d"))) + tuple(_safe_category(values.get(column)) for column in CATEGORICAL_AD_FEATURE_COLUMNS))
            connection.executemany(
                "INSERT INTO interactions VALUES (?, ?, ?, ?) ON CONFLICT(user_id, candidate_ad_id) DO UPDATE SET click_timestamp = CASE WHEN excluded.click_timestamp > interactions.click_timestamp THEN excluded.click_timestamp ELSE interactions.click_timestamp END, conversion_label = MAX(interactions.conversion_label, excluded.conversion_label)",
                interaction_rows,
            )
            placeholders = ",".join("?" for _ in range(1 + len(NUMERIC_AD_FEATURE_COLUMNS) + len(CATEGORICAL_AD_FEATURE_COLUMNS)))
            connection.executemany(f"INSERT OR IGNORE INTO ad_features VALUES ({placeholders})", feature_rows)
            connection.commit()
            rows_seen += len(chunk)
            LOGGER.info("Indexed interactions=%s", rows_seen)
    finally:
        connection.close()
    return SQLiteFeatureStore(database_path, config.feature_cache_size)


def evaluate_positive_retention(
    config: CoarseRankConfig, model: CoarseRankModel, feature_store: SQLiteFeatureStore, validation_group_keys: set[int]
) -> float:
    """Score complete candidate groups and compute positive retention at Top-K."""

    positives = 0
    retained = 0
    for user_id, candidates in iter_candidate_groups(config.input_path, config.chunk_size):
        if _stable_hash(user_id, config.random_seed) not in validation_group_keys:
            continue
        labels = feature_store.interactions_for(candidates["candidate_ad_id"].tolist(), user_id)
        if not labels:
            continue
        scored = score_candidate_group(candidates, model, feature_store)
        top_ids = set(scored.iloc[: config.top_k]["candidate_ad_id"])
        positive_ids = set(labels)
        positives += len(positive_ids)
        retained += len(top_ids & positive_ids)
    value = float(retained / positives) if positives else float("nan")
    LOGGER.info("Positive retention@%s=%.6f (%s/%s)", config.top_k, value, retained, positives)
    return value


def score_candidate_group(candidates: pd.DataFrame, model: CoarseRankModel, feature_store: SQLiteFeatureStore) -> pd.DataFrame:
    """Score and stably order one user's bounded candidate set."""

    scored = feature_store.enrich(candidates)
    scored["coarse_score"] = model.estimator.predict_proba(preprocess_features(scored))[:, 1]
    return scored.sort_values(["coarse_score", "rrf_score", "candidate_ad_id"], ascending=[False, False, True], kind="mergesort")


def iter_candidate_batches(
    path: Path, chunk_size: int, *, max_users: int, batch_users: int, batch_candidates: int
) -> Iterator[list[tuple[str, pd.DataFrame]]]:
    """Yield bounded batches made exclusively of complete user candidate sets."""

    batch: list[tuple[str, pd.DataFrame]] = []
    candidates_in_batch = 0
    users_seen = 0
    for user_id, candidates in iter_candidate_groups(path, chunk_size):
        if users_seen >= max_users:
            break
        # Never split a user at a batch boundary.  An unusually large single
        # user is still emitted as one bounded-by-source group.
        if batch and (len(batch) >= batch_users or candidates_in_batch + len(candidates) > batch_candidates):
            yield batch
            batch = []
            candidates_in_batch = 0
        batch.append((user_id, candidates))
        candidates_in_batch += len(candidates)
        users_seen += 1
    if batch:
        yield batch


def score_candidate_batch(
    user_groups: Sequence[tuple[str, pd.DataFrame]], model: CoarseRankModel, feature_store: SQLiteFeatureStore, top_k: int
) -> list[tuple[str, str, float, int]]:
    """Run one vectorised feature lookup/predict call for many complete users."""

    if not user_groups:
        return []
    boundaries: list[tuple[str, int, int]] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    for user_id, candidates in user_groups:
        frames.append(candidates)
        end = offset + len(candidates)
        boundaries.append((user_id, offset, end))
        offset = end
    # This concat is deliberately capped by inference_batch_candidates; it is
    # not a global candidate merge.
    batch = pd.concat(frames, ignore_index=True, copy=False)
    enriched = feature_store.enrich(batch)
    scores = model.estimator.predict_proba(preprocess_features(enriched))[:, 1]
    rrf_scores = pd.to_numeric(enriched["rrf_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    candidate_ids = enriched["candidate_ad_id"].astype(str).to_numpy()
    output_rows: list[tuple[str, str, float, int]] = []
    for user_id, start, end in boundaries:
        # Each group has approximately 100 rows.  Sorting this small local
        # slice gives the exact deterministic three-key Top-K ordering without
        # a costly global sort.
        order = np.lexsort((candidate_ids[start:end], -rrf_scores[start:end], -scores[start:end]))[:top_k]
        for rank, local_index in enumerate(order, start=1):
            index = start + int(local_index)
            output_rows.append((user_id, str(candidate_ids[index]), float(scores[index]), rank))
    return output_rows


def _initialise_inference_worker(
    model_blob: bytes, database_path: str, cache_size: int, top_k: int, thread_limit: int
) -> None:
    """Initialise a worker-local model, SQLite reader and one-thread backend."""

    global _WORKER_MODEL, _WORKER_STORE, _WORKER_TOP_K, _WORKER_THREAD_CONTROLLER
    os.environ["OMP_NUM_THREADS"] = str(thread_limit)
    os.environ["MKL_NUM_THREADS"] = str(thread_limit)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_limit)
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREAD_CONTROLLER = threadpool_limits(limits=thread_limit)
    except ImportError:
        # scikit-learn installs threadpoolctl, but keep inference usable in a
        # minimal environment and retain the explicit environment settings.
        pass
    _WORKER_MODEL = pickle.loads(model_blob)
    _WORKER_STORE = SQLiteFeatureStore(Path(database_path), cache_size, read_only=True)
    _WORKER_TOP_K = top_k


def _score_batch_worker(
    task: tuple[int, list[tuple[str, pd.DataFrame]]]
) -> tuple[int, list[tuple[str, str, float, int]], int, int]:
    """Process one bounded batch and return stats to the single writer."""

    if _WORKER_MODEL is None or _WORKER_STORE is None or _WORKER_TOP_K is None:
        raise RuntimeError("Coarse-rank worker was not initialised")
    sequence, user_groups = task
    rows = score_candidate_batch(user_groups, _WORKER_MODEL, _WORKER_STORE, _WORKER_TOP_K)
    missing, total = _WORKER_STORE.consume_feature_stats()
    return sequence, rows, missing, total


def _log_progress(
    *, processed_candidates: int, processed_users: int, elapsed_seconds: float, missing_features: int, total_features: int
) -> None:
    elapsed = max(elapsed_seconds, 1e-9)
    LOGGER.info(
        "processed_candidates=%s processed_users=%s elapsed_seconds=%.2f candidates_per_second=%.1f users_per_second=%.1f missing_feature_candidates=%s total_feature_candidates=%s missing_feature_rate=%.6f",
        processed_candidates,
        processed_users,
        elapsed,
        processed_candidates / elapsed,
        processed_users / elapsed,
        missing_features,
        total_features,
        missing_features / total_features if total_features else 0.0,
    )


def stream_coarse_rank_output(
    config: CoarseRankConfig,
    model: CoarseRankModel,
    feature_store: SQLiteFeatureStore,
    *,
    max_users: int | None = None,
    stats: dict[str, float] | None = None,
) -> int:
    """Micro-batch complete users, predict once per batch, and atomically write Top-K.

    ``num_workers > 1`` uses a bounded process pool with read-only SQLite
    connections.  Only the parent writes CSV and commits rows in input-batch
    sequence, so the result is deterministic and workers cannot corrupt output.
    """

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    written = 0
    processed_candidates = 0
    processed_users = 0
    missing_features = 0
    total_features = 0
    start_time = time.monotonic()
    user_limit = config.max_users if max_users is None else max_users
    worker_count = config.num_workers
    LOGGER.info(
        "Coarse inference available_cpu_count=%s configured_num_workers=%s inference_batch_users=%s inference_batch_candidates=%s prefetch_batches=%s native_threads_per_worker=%s",
        os.cpu_count(), worker_count, config.inference_batch_users, config.inference_batch_candidates, config.prefetch_batches,
        1 if worker_count > 1 else "backend-default",
    )
    # Training/evaluation may have used this parent store before inference;
    # report only feature misses from the current output pass.
    feature_store.consume_feature_stats()

    # Treat the configured batch sizes as the total in-flight budget.  In a
    # process pool each task therefore receives a fraction of that budget,
    # preventing 12 workers from accidentally buffering 12 x 100k rows.
    task_batch_users = config.inference_batch_users
    task_batch_candidates = config.inference_batch_candidates
    if worker_count > 1:
        task_batch_users = max(1, (task_batch_users + worker_count - 1) // worker_count)
        task_batch_candidates = max(1, (task_batch_candidates + worker_count - 1) // worker_count)

    def write_rows(writer: csv.writer, rows: Sequence[tuple[str, str, float, int]]) -> None:
        nonlocal written, processed_candidates, processed_users, missing_features, total_features
        writer.writerows(rows)
        written += len(rows)

    executor: ProcessPoolExecutor | None = None
    if worker_count > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialise_inference_worker,
                initargs=(
                    pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
                    str(feature_store.database_path),
                    config.feature_cache_size,
                    config.top_k,
                    1,
                ),
            )
        except (OSError, PermissionError, NotImplementedError) as error:
            # Some constrained runtimes prohibit POSIX semaphores.  This is a
            # process-pool availability fallback only; task/worker exceptions
            # from an active pool still propagate to the caller.
            LOGGER.warning("Multi-process inference unavailable (%s); falling back to one worker", error)
            worker_count = 1
            task_batch_users = config.inference_batch_users
            task_batch_candidates = config.inference_batch_candidates
    batches = iter_candidate_batches(
        config.input_path,
        config.chunk_size,
        max_users=user_limit,
        batch_users=task_batch_users,
        batch_candidates=task_batch_candidates,
    )
    LOGGER.info("Coarse inference task_batch_users=%s task_batch_candidates=%s active_workers=%s", task_batch_users, task_batch_candidates, worker_count)
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)
        if worker_count == 1:
            for user_groups in batches:
                rows = score_candidate_batch(user_groups, model, feature_store, config.top_k)
                write_rows(writer, rows)
                processed_users += len(user_groups)
                processed_candidates += sum(len(candidates) for _, candidates in user_groups)
                missing, total = feature_store.consume_feature_stats()
                missing_features += missing
                total_features += total
                _log_progress(
                    processed_candidates=processed_candidates,
                    processed_users=processed_users,
                    elapsed_seconds=time.monotonic() - start_time,
                    missing_features=missing_features,
                    total_features=total_features,
                )
        else:
            # One active task per worker plus a small producer look-ahead is a
            # bounded queue with backpressure, not an unbounded future list.
            max_pending = worker_count + config.prefetch_batches
            pending: dict[Future[tuple[int, list[tuple[str, str, float, int]], int, int]], tuple[int, int]] = {}
            completed: dict[int, tuple[list[tuple[str, str, float, int]], int, int, int, int]] = {}
            next_sequence = 0
            next_to_write = 0
            batches_exhausted = False
            assert executor is not None
            with executor:
                while pending or not batches_exhausted:
                    while not batches_exhausted and len(pending) < max_pending:
                        try:
                            user_groups = next(batches)
                        except StopIteration:
                            batches_exhausted = True
                            break
                        candidate_count = sum(len(candidates) for _, candidates in user_groups)
                        future = executor.submit(_score_batch_worker, (next_sequence, user_groups))
                        pending[future] = (len(user_groups), candidate_count)
                        next_sequence += 1
                    if not pending:
                        continue
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        user_count, candidate_count = pending.pop(future)
                        sequence, rows, missing, total = future.result()  # propagates worker errors
                        completed[sequence] = (rows, missing, total, user_count, candidate_count)
                    while next_to_write in completed:
                        rows, missing, total, user_count, candidate_count = completed.pop(next_to_write)
                        write_rows(writer, rows)
                        processed_users += user_count
                        processed_candidates += candidate_count
                        missing_features += missing
                        total_features += total
                        _log_progress(
                            processed_candidates=processed_candidates,
                            processed_users=processed_users,
                            elapsed_seconds=time.monotonic() - start_time,
                            missing_features=missing_features,
                            total_features=total_features,
                        )
                        next_to_write += 1
    temporary.replace(config.output_path)
    _log_progress(
        processed_candidates=processed_candidates,
        processed_users=processed_users,
        elapsed_seconds=time.monotonic() - start_time,
        missing_features=missing_features,
        total_features=total_features,
    )
    if stats is not None:
        elapsed = max(time.monotonic() - start_time, 1e-9)
        stats.update(
            {
                "processed_candidates": float(processed_candidates),
                "processed_users": float(processed_users),
                "elapsed_seconds": elapsed,
                "candidates_per_second": processed_candidates / elapsed,
                "users_per_second": processed_users / elapsed,
                "missing_feature_candidates": float(missing_features),
                "total_feature_candidates": float(total_features),
            }
        )
    LOGGER.info("Wrote %s coarse-rank rows to %s", written, config.output_path)
    return written


def resolve_inference_backend(config: CoarseRankConfig) -> str:
    """Select the supported backend and make every fallback explicit.

    The persisted baseline is sklearn HistGBDT.  XGBoost/LightGBM checkpoints
    have a different serialization contract and are intentionally not silently
    substituted for it.  A future accelerated experiment must use its own
    ``model_path`` and explicitly implement/train that backend.
    """

    if config.model_type == "hist_gbdt":
        if config.enable_gpu_if_available:
            LOGGER.info("HistGBDT backend is CPU-only; retaining the existing CPU checkpoint semantics")
        LOGGER.info("Coarse rank backend: hist_gbdt; device: cpu")
        return "hist_gbdt"
    if config.model_type in {"xgboost", "xgboost_gpu", "lightgbm", "lightgbm_gpu"}:
        LOGGER.warning(
            "Requested backend %s is not available in this HistGBDT checkpoint pipeline; falling back to HistGBDT CPU. "
            "Use a distinct model_path when adding a trained accelerated backend.",
            config.model_type,
        )
        return "hist_gbdt"
    raise ValueError("coarse_rank.model_type must be hist_gbdt, xgboost[_gpu], or lightgbm[_gpu]")


def benchmark_coarse_rank(
    config: CoarseRankConfig,
    model: CoarseRankModel,
    feature_store: SQLiteFeatureStore,
    *,
    worker_counts: Sequence[int] = (1, 4, 8, 12),
    max_users: int = 10_000,
) -> dict[int, dict[str, float]]:
    """Benchmark bounded inference into disposable files, never the formal output."""

    results: dict[int, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="coarse-rank-benchmark-") as directory:
        for workers in worker_counts:
            if workers <= 0:
                raise ValueError("benchmark worker counts must be positive")
            output_path = Path(directory) / f"workers-{workers}.csv"
            benchmark_config = replace(config, output_path=output_path, num_workers=workers)
            run_stats: dict[str, float] = {}
            stream_coarse_rank_output(benchmark_config, model, feature_store, max_users=max_users, stats=run_stats)
            results[workers] = run_stats
            LOGGER.info(
                "benchmark workers=%s candidates_per_second=%.1f users_per_second=%.1f elapsed_seconds=%.2f",
                workers,
                run_stats["candidates_per_second"],
                run_stats["users_per_second"],
                run_stats["elapsed_seconds"],
            )
    return results


def run_coarse_rank(config: CoarseRankConfig) -> dict[str, float]:
    """Train/load a checkpoint, evaluate it, then stream the final Top-K output."""

    resolve_inference_backend(config)
    database_path = config.model_path.with_suffix(config.model_path.suffix + ".features.sqlite")
    feature_store = build_interaction_feature_index(config, database_path)
    try:
        metrics: dict[str, float] = {}
        if config.train:
            LOGGER.warning("Candidate files lack request timestamps; coarse-rank time split is an offline baseline, not a strict causal split.")
            data = collect_training_data(config, feature_store)
            model, metrics, validation_groups = train_coarse_ranker(data, config)
            save_model(model, config.model_path)
            metrics["positive_retention_at_k"] = evaluate_positive_retention(config, model, feature_store, validation_groups)
        else:
            model = load_model(config.model_path)
            LOGGER.info("Loaded existing coarse-rank model from %s; training skipped", config.model_path)
        stream_coarse_rank_output(config, model, feature_store)
        return metrics
    finally:
        feature_store.close()
        if database_path.exists():
            database_path.unlink()


def _iter_interaction_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    if path.is_dir():
        yield from iter_csv_parts(path, chunk_size)
    elif path.is_file():
        yield from pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    else:
        raise FileNotFoundError(f"Interaction path does not exist: {path}")


def _validate_config(config: CoarseRankConfig) -> None:
    if config.max_train_rows <= 0 or config.max_users <= 0 or config.top_k <= 0 or config.chunk_size <= 0:
        raise ValueError("max_train_rows, max_users, top_k, and chunk_size must be positive")
    if config.negatives_per_positive < 0 or config.conversion_sample_weight < 1.0 or config.feature_cache_size <= 0:
        raise ValueError("negative sampling, conversion weight, and feature cache settings are invalid")
    if config.num_workers <= 0 or config.inference_batch_users <= 0 or config.inference_batch_candidates <= 0 or config.prefetch_batches <= 0:
        raise ValueError("inference worker and batch settings must be positive")
    if config.model_type not in {"hist_gbdt", "xgboost", "xgboost_gpu", "lightgbm", "lightgbm_gpu"}:
        raise ValueError("coarse_rank.model_type must be hist_gbdt, xgboost[_gpu], or lightgbm[_gpu]")


def _empty_examples() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32), -1


def _empty_ad_features() -> tuple[Any, ...]:
    return (None,) * (len(NUMERIC_AD_FEATURE_COLUMNS) + len(CATEGORICAL_AD_FEATURE_COLUMNS))


def _normalise_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _safe_int(value: object) -> int | None:
    result = _safe_float(value)
    return None if result is None else int(result)


def _safe_category(value: object) -> str | None:
    return _normalise_id(value)


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}\x1f{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _batches(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run memory-safe coarse ranking.")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    run_coarse_rank(parse_coarse_rank_config(load_yaml_config(config_path), config_path))


if __name__ == "__main__":
    main()
