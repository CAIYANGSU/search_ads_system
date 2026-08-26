"""Streaming diagnosis for coarse-rank training-sample coverage.

This module deliberately shares the production click-positive and negative
sampling helpers from :mod:`coarse_rank`.  It never fits a model, scores
candidates, or writes coarse-ranking output.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_ads_system.ranking.coarse_rank import (
    CoarseRankConfig,
    SQLiteFeatureStore,
    _stable_hash,
    build_interaction_feature_index,
    build_labeled_candidate_rows,
    iter_candidate_batches,
    split_train_validation,
)

LOGGER = logging.getLogger(__name__)


class HyperLogLog:
    """Small deterministic cardinality estimator for 50M-row diagnostics.

    Exact distinct candidate/product counts would require retaining millions
    of IDs in RAM or building a large temporary distinct table.  This 16 KiB
    sketch keeps the diagnostic memory-bounded; its output is explicitly
    labelled as an estimate.
    """

    def __init__(self, precision: int = 14) -> None:
        self.precision = precision
        self.registers = np.zeros(1 << precision, dtype=np.uint8)

    def add(self, value: str) -> None:
        self.add_many((value,))

    def add_many(self, values: list[str] | tuple[str, ...]) -> None:
        """Vectorised deterministic hashing for a bounded candidate batch."""

        if not values:
            return
        hashes = pd.util.hash_pandas_object(pd.Series(values, dtype="string"), index=False).to_numpy(dtype=np.uint64)
        indices = hashes & np.uint64((1 << self.precision) - 1)
        remainder = hashes >> np.uint64(self.precision)
        ranks = np.empty(len(hashes), dtype=np.uint8)
        nonzero = remainder != 0
        ranks[~nonzero] = 64 - self.precision + 1
        ranks[nonzero] = (64 - self.precision - np.floor(np.log2(remainder[nonzero])).astype(np.int16)).astype(np.uint8)
        np.maximum.at(self.registers, indices.astype(np.intp), ranks)

    def estimate(self) -> int:
        buckets = len(self.registers)
        alpha = 0.7213 / (1.0 + 1.079 / buckets)
        raw = alpha * buckets * buckets / float(np.exp2(-self.registers.astype(np.int16)).sum())
        zero_buckets = int((self.registers == 0).sum())
        if raw <= 2.5 * buckets and zero_buckets:
            raw = buckets * math.log(buckets / zero_buckets)
        return int(round(raw))


@dataclass(frozen=True)
class _SelectedSample:
    labels: np.ndarray
    timestamps: np.ndarray
    group_keys: np.ndarray


def diagnose_coarse_rank_samples(config: CoarseRankConfig) -> dict[str, Any]:
    """Build an interaction index and stream a no-training candidate diagnosis."""

    raw_id_check = _candidate_id_consistency_sample(config.input_path, config.chunk_size)
    database_path = config.model_path.with_suffix(config.model_path.suffix + ".diagnostics.sqlite")
    feature_store = build_interaction_feature_index(config, database_path)
    try:
        report = _diagnose_candidates(config, feature_store)
        report["interaction_side"] = feature_store.interaction_summary()
        report["id_consistency"]["product_id_sample"] = [
            str(row[0]) for row in feature_store.connection.execute("SELECT candidate_ad_id FROM ad_features ORDER BY candidate_ad_id LIMIT 5")
        ]
        report["id_consistency"].update(raw_id_check)
        LOGGER.info("COARSE_RANK_SAMPLE_DIAGNOSIS\n%s", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return report
    finally:
        feature_store.close()
        if database_path.exists():
            database_path.unlink()


def _diagnose_candidates(config: CoarseRankConfig, feature_store: SQLiteFeatureStore) -> dict[str, Any]:
    candidate_ads = HyperLogLog()
    overlapping_ads = HyperLogLog()
    total_candidate_rows = 0
    selected_candidate_users = 0
    overlapping_users = 0
    pair_matches = 0
    positive_users = 0
    positives_per_user: list[int] = []
    exclusion_reasons: Counter[str] = Counter()
    candidate_samples: list[str] = []
    sample_label_parts: list[np.ndarray] = []
    sample_timestamp_parts: list[np.ndarray] = []
    sample_group_key_parts: list[np.ndarray] = []
    sampled_rows = 0
    sampled_positives = 0
    started = time.monotonic()

    for user_groups in iter_candidate_batches(
        config.input_path,
        config.chunk_size,
        max_users=config.max_users,
        batch_users=config.inference_batch_users,
        batch_candidates=config.inference_batch_candidates,
    ):
        user_ids = [user_id for user_id, _ in user_groups]
        interaction_users = feature_store.interaction_users(user_ids)
        candidate_ids = [str(candidate_id) for _, candidates in user_groups for candidate_id in candidates["candidate_ad_id"]]
        known_products = feature_store.known_ad_ids(candidate_ids)
        matches_by_group = feature_store.interactions_for_groups(user_groups)

        for group_index, (user_id, candidates) in enumerate(user_groups):
            row_count = len(candidates)
            total_candidate_rows += row_count
            selected_candidate_users += 1
            if user_id in interaction_users:
                overlapping_users += 1
            candidate_id_values = candidates["candidate_ad_id"].astype(str).tolist()
            candidate_ads.add_many(candidate_id_values)
            overlapping_ads.add_many([candidate_id for candidate_id in candidate_id_values if candidate_id in known_products])
            if len(candidate_samples) < 5:
                candidate_samples.extend(candidate_id_values[: 5 - len(candidate_samples)])

            interactions = matches_by_group[group_index]
            positive_count = sum(candidate_id in interactions for candidate_id in candidate_id_values)
            pair_matches += positive_count
            if positive_count:
                positive_users += 1
                positives_per_user.append(positive_count)
            _count_exclusions(
                exclusion_reasons,
                user_in_interactions=user_id in interaction_users,
                candidate_ids=candidate_id_values,
                known_products=known_products,
                interactions=interactions,
            )
            if positive_count and positive_count == row_count:
                exclusion_reasons["no_negative_available"] += 1

            # Same helper as coarse-rank training: positives are real click
            # pairs and negatives use the deterministic per-user sampler.
            selected, timestamp = build_labeled_candidate_rows(
                candidates,
                user_id=user_id,
                interactions=interactions,
                negatives_per_positive=config.negatives_per_positive,
                random_seed=config.random_seed,
            )
            if selected.empty:
                continue
            selected_labels = selected["_interaction"].notna().to_numpy(dtype=np.int8)
            if sampled_rows + len(selected_labels) > config.max_train_rows:
                exclusion_reasons["excluded_by_sampling"] += int(selected_labels.sum())
                continue
            sample_label_parts.append(selected_labels)
            sample_timestamp_parts.append(np.full(len(selected_labels), timestamp, dtype=np.int64))
            sample_group_key_parts.append(np.full(len(selected_labels), _stable_hash(user_id, config.random_seed), dtype=np.uint64))
            sampled_rows += len(selected_labels)
            sampled_positives += int(selected_labels.sum())

        elapsed = max(time.monotonic() - started, 1e-9)
        LOGGER.info(
            "processed_candidate_rows=%s processed_users=%s elapsed_seconds=%.2f rows_per_second=%.1f",
            total_candidate_rows,
            selected_candidate_users,
            elapsed,
            total_candidate_rows / elapsed,
        )

    selected_sample = _combine_selected_samples(sample_label_parts, sample_timestamp_parts, sample_group_key_parts)
    time_split = _time_split_summary(selected_sample, config.random_seed)
    positive_distribution = _positive_distribution(selected_candidate_users, positives_per_user)
    unique_candidate_ads = candidate_ads.estimate()
    overlapping_products = overlapping_ads.estimate()
    return {
        "candidate_side": {
            "total_candidate_rows": total_candidate_rows,
            "selected_candidate_users": selected_candidate_users,
            "unique_candidate_ads": unique_candidate_ads,
            "unique_candidate_ads_method": "HyperLogLog estimate (bounded memory)",
            "average_candidates_per_user": _safe_divide(total_candidate_rows, selected_candidate_users),
        },
        "overlap": {
            "overlapping_users": overlapping_users,
            "user_overlap_rate": _safe_divide(overlapping_users, selected_candidate_users),
            "overlapping_products": overlapping_products,
            "product_overlap_rate": _safe_divide(overlapping_products, unique_candidate_ads),
            "product_overlap_method": "HyperLogLog estimates (bounded memory)",
        },
        "positive_funnel": {
            "candidate_interaction_pair_matches_before_split": pair_matches,
            "positive_pairs_before_time_split": pair_matches,
            "positive_pairs_after_time_split": "N/A: current training split partitions sampled rows; it does not discard positives",
            "positive_pairs_after_dedup": "N/A: current coarse-rank training does not include a candidate-pair dedup stage",
            "final_positive_samples": sampled_positives,
            "users_with_at_least_one_positive": positive_users,
            "positive_user_rate": _safe_divide(positive_users, selected_candidate_users),
            "average_positives_per_positive_user": _safe_divide(pair_matches, positive_users),
            "candidate_positive_rate": _safe_divide(pair_matches, total_candidate_rows),
        },
        "time_split": time_split,
        "per_user_positive_distribution": positive_distribution,
        "negative_sampling": {
            "negatives_per_positive": config.negatives_per_positive,
            "expected_training_rows": sampled_positives * (1 + config.negatives_per_positive),
            "actual_training_rows": sampled_rows,
            "ratio_is_exact": sampled_rows == sampled_positives * (1 + config.negatives_per_positive),
            "note": "A false ratio means a positive user had fewer sampled negatives than requested, or a group was excluded by max_train_rows.",
        },
        "exclusion_reasons": {
            "user_not_in_interactions": exclusion_reasons["user_not_in_interactions"],
            "product_not_in_interactions": exclusion_reasons["product_not_in_interactions"],
            "user_and_product_exist_but_pair_not_interacted": exclusion_reasons["user_and_product_exist_but_pair_not_interacted"],
            "excluded_by_time_split": time_split["validation_positive_pairs"],
            "excluded_by_dedup": "N/A: no production dedup stage",
            "excluded_by_sampling": exclusion_reasons["excluded_by_sampling"],
            "no_negative_available": exclusion_reasons["no_negative_available"],
        },
        "id_consistency": {
            "join_normalization": "candidate_ad_id, product_id, and user_id use coarse_rank._normalise_id (string conversion + trim; case-sensitive)",
            "candidate_ad_id_sample": candidate_samples,
            "normalized_id_note": "The streamed diagnosis sees normalized IDs, exactly as coarse-rank training does.",
        },
        "diagnosis_note": "Candidate files have no request timestamp. The 80/20 timestamp split is an offline baseline, not strict causal evaluation.",
    }


def _count_exclusions(
    counters: Counter[str],
    *,
    user_in_interactions: bool,
    candidate_ids: list[str],
    known_products: set[str],
    interactions: dict[str, tuple[int | None, int]],
) -> None:
    """Count mutually exclusive non-positive reasons without retaining rows."""

    if not user_in_interactions:
        counters["user_not_in_interactions"] += len(candidate_ids)
        return
    for candidate_id in candidate_ids:
        if candidate_id not in known_products:
            counters["product_not_in_interactions"] += 1
        elif candidate_id not in interactions:
            counters["user_and_product_exist_but_pair_not_interacted"] += 1


def _combine_selected_samples(
    labels: list[np.ndarray], timestamps: list[np.ndarray], group_keys: list[np.ndarray]
) -> _SelectedSample:
    if not labels:
        return _SelectedSample(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64))
    return _SelectedSample(np.concatenate(labels), np.concatenate(timestamps), np.concatenate(group_keys))


def _time_split_summary(sample: _SelectedSample, random_seed: int) -> dict[str, Any]:
    if not len(sample.labels):
        return {
            "split_method": "N/A: no selected training samples",
            "train_time_range": "N/A",
            "validation_time_range": "N/A",
            "train_positive_pairs": 0,
            "validation_positive_pairs": 0,
        }
    validation_mask = split_train_validation(sample.timestamps, sample.group_keys, random_seed)
    return {
        "split_method": "timestamp 80/20 when timestamps vary; deterministic fallback otherwise",
        "train_time_range": _time_range(sample.timestamps[~validation_mask]),
        "validation_time_range": _time_range(sample.timestamps[validation_mask]),
        "train_positive_pairs": int(sample.labels[~validation_mask].sum()),
        "validation_positive_pairs": int(sample.labels[validation_mask].sum()),
    }


def _positive_distribution(total_users: int, positives_per_user: list[int]) -> dict[str, Any]:
    values = np.asarray(positives_per_user, dtype=np.int64)
    return {
        "users_with_0_positive": total_users - len(values),
        "users_with_1_positive": int((values == 1).sum()),
        "users_with_2_to_5_positives": int(((values >= 2) & (values <= 5)).sum()),
        "users_with_more_than_5_positives": int((values > 5).sum()),
        "p50_positives_per_positive_user": _percentile_or_na(values, 50),
        "p90_positives_per_positive_user": _percentile_or_na(values, 90),
        "p99_positives_per_positive_user": _percentile_or_na(values, 99),
    }


def _percentile_or_na(values: np.ndarray, percentile: int) -> float | str:
    return "N/A" if not len(values) else float(np.percentile(values, percentile))


def _time_range(values: np.ndarray) -> dict[str, int | None]:
    valid = values[values >= 0]
    return {"min": None if not len(valid) else int(valid.min()), "max": None if not len(valid) else int(valid.max())}


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _candidate_id_consistency_sample(path: Path, chunk_size: int) -> dict[str, Any]:
    """Inspect only the first CSV chunk for raw identifier hazards."""

    try:
        chunk = next(pd.read_csv(path, usecols=["candidate_ad_id"], chunksize=min(chunk_size, 10_000), low_memory=False))
    except ValueError as error:
        raise ValueError(f"Fused candidates at {path} are missing candidate_ad_id: {error}") from error
    raw_values = chunk["candidate_ad_id"]
    strings = raw_values.astype("string")
    non_null = strings.dropna()
    return {
        "candidate_ad_id_raw_sample": [str(value) for value in non_null.iloc[:5]],
        "raw_candidate_id_inspection_scope": f"first {len(chunk)} candidate rows only",
        "raw_candidate_id_null_count": int(strings.isna().sum()),
        "raw_candidate_id_whitespace_count": int((non_null.str.strip() != non_null).sum()),
        "raw_candidate_id_numeric_string_count": int(non_null.str.fullmatch(r"[+-]?\d+(?:\.0+)?").sum()),
        "case_mismatch_check": "N/A: production joins are deliberately case-sensitive opaque-ID joins; no case-folding is applied",
        "accidental_truncation_check": "N/A: cannot be inferred safely without an external ID-length contract",
    }
