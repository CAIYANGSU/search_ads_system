"""Strict-temporal ID-only versus content-item Two-Tower ablation.

This module deliberately has no user-history feature.  Both variants consume
the same streaming positives, deterministic sampled negatives, user tower,
loss, catalogue, FAISS retrieval, and evaluation cohort; only the item tower
changes.  Product observations and seen-item histories are kept in a temporary
SQLite database so CSV chunks are never concatenated into a full dataframe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import tempfile
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from search_ads_system.recall.faiss_index import (
    build_faiss_index,
    load_faiss_index,
    load_faiss_index_metadata,
    save_faiss_index,
    search_faiss_index,
)
from search_ads_system.recall.two_tower_recall import OUTPUT_COLUMNS, _input_csv_files, _select_device

LOGGER = logging.getLogger(__name__)
PAD_TOKEN = "__PAD__"
OOV_TOKEN = "__OOV__"
PAD_INDEX = 0
OOV_INDEX = 1
SCHEMA_VERSION = "content_item_two_tower_v1"
ITEM_CATEGORICAL_FEATURES = (
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "partner_id",
    "product_country",
    "product_age_group",
    "product_gender",
)
DEFAULT_ENABLED_FEATURES = (
    "product_id",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "partner_id",
    "product_price",
)
ALL_ITEM_FEATURES = ("product_id",) + ITEM_CATEGORICAL_FEATURES + ("product_price",)
EVALUATION_CUTOFFS = (10, 20, 50, 100, 200)


@dataclass(frozen=True)
class ContentItemTwoTowerConfig:
    input_path: Path
    future_path: Path
    output_dir: Path
    split_timestamp: int
    product_catalog_path: Path | None = None
    product_catalog_as_of_timestamp: int | None = None
    enabled_features: tuple[str, ...] = DEFAULT_ENABLED_FEATURES
    embedding_dim: int = 32
    feature_embedding_dim: int = 16
    price_embedding_dim: int = 4
    hidden_dims: tuple[int, ...] = (128, 64)
    batch_size: int = 4096
    epochs: int = 3
    learning_rate: float = 1e-3
    negative_samples: int = 5
    click_weight: float = 1.0
    conversion_weight: float = 3.0
    max_train_rows: int | None = None
    top_k: int = 200
    exclude_seen_items: bool = True
    retrieval_oversample_ratio: float = 2.0
    search_batch_size: int = 10_000
    inference_batch_size: int = 4096
    input_chunk_size: int = 200_000
    log_every_rows: int = 200_000
    history_cache_users: int = 20_000
    seed: int = 2026
    device: str = "auto"
    faiss_index_type: str = "hnsw"
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    train: bool = True

    @property
    def catalog_mode(self) -> str:
        return "external_point_in_time" if self.product_catalog_path is not None else "past_only"


@dataclass(frozen=True)
class CatalogFeatures:
    product_ids: np.ndarray
    product_id_indices: np.ndarray
    categorical_indices: Mapping[str, np.ndarray]
    normalized_log_prices: np.ndarray
    warm_mask: np.ndarray
    price_mean: float
    price_std: float
    fingerprint: str


@dataclass
class PreparedContentItemData:
    config: ContentItemTwoTowerConfig
    state_path: Path
    user_ids: np.ndarray
    user_to_index: dict[str, int]
    product_vocab: tuple[str, ...]
    product_to_vocab_index: dict[str, int]
    categorical_vocabs: dict[str, tuple[str, ...]]
    categorical_to_index: dict[str, dict[str, int]]
    catalog: CatalogFeatures
    product_to_catalog_position: dict[str, int]
    training_product_positions: np.ndarray
    training_rows: int
    metadata: dict[str, Any]


class StreamingSampledNegativeDataset(IterableDataset[tuple[int, int, np.ndarray, float]]):
    """Re-read Past chunks and yield deterministic sampled-softmax examples."""

    def __init__(self, data: PreparedContentItemData) -> None:
        super().__init__()
        self.data = data

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray, float]]:
        lookup = HistoryLookup(self.data, self.data.config.history_cache_users)
        pool = self.data.training_product_positions
        emitted = 0
        try:
            for chunk in _iter_training_chunks(self.data.config):
                for user, product, label in chunk[["user_id", "product_id", "conversion_label"]].itertuples(
                    index=False, name=None
                ):
                    user_index = self.data.user_to_index[str(user)]
                    positive = self.data.product_to_catalog_position[str(product)]
                    history = lookup.for_user(user_index)
                    if len(history.intersection(pool)) >= len(pool):
                        continue
                    rng = np.random.default_rng(self.data.config.seed + emitted)
                    negatives: list[int] = []
                    attempts = 0
                    while len(negatives) < self.data.config.negative_samples and attempts < self.data.config.negative_samples * 30:
                        candidate = int(pool[int(rng.integers(len(pool)))])
                        if candidate not in history:
                            negatives.append(candidate)
                        attempts += 1
                    if len(negatives) < self.data.config.negative_samples:
                        available = pool[np.fromiter((int(item) not in history for item in pool), dtype=bool, count=len(pool))]
                        if not len(available):
                            continue
                        negatives.extend(
                            rng.choice(
                                available,
                                self.data.config.negative_samples - len(negatives),
                                replace=len(available) < self.data.config.negative_samples - len(negatives),
                            ).astype(np.int64).tolist()
                        )
                    weight = self.data.config.conversion_weight if int(label) == 1 else self.data.config.click_weight
                    yield user_index, positive, np.asarray(negatives, dtype=np.int64), float(weight)
                    emitted += 1
        finally:
            lookup.close()


class ContentItemTwoTowerModel(nn.Module):
    """ID-only user tower and selectable ID-only/content-aware item tower."""

    def __init__(
        self,
        data: PreparedContentItemData,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in {"id_only", "content_item"}:
            raise ValueError("variant must be 'id_only' or 'content_item'")
        self.variant = variant
        self.config = data.config
        self.user_embedding = nn.Embedding(len(data.user_ids) + 2, data.config.embedding_dim, padding_idx=PAD_INDEX)
        self.user_tower = _mlp(data.config.embedding_dim, data.config.hidden_dims, data.config.embedding_dim)

        item_features = ("product_id",) if variant == "id_only" else data.config.enabled_features
        self.item_features = tuple(item_features)
        self.product_id_embedding: nn.Embedding | None = None
        self.categorical_embeddings = nn.ModuleDict()
        self.price_projection: nn.Linear | None = None
        input_dim = 0
        if "product_id" in self.item_features:
            self.product_id_embedding = nn.Embedding(
                len(data.product_vocab), data.config.feature_embedding_dim, padding_idx=PAD_INDEX
            )
            input_dim += data.config.feature_embedding_dim
        for feature in ITEM_CATEGORICAL_FEATURES:
            if feature in self.item_features:
                self.categorical_embeddings[feature] = nn.Embedding(
                    len(data.categorical_vocabs[feature]),
                    data.config.feature_embedding_dim,
                    padding_idx=PAD_INDEX,
                )
                input_dim += data.config.feature_embedding_dim
        if "product_price" in self.item_features:
            self.price_projection = nn.Linear(1, data.config.price_embedding_dim)
            input_dim += data.config.price_embedding_dim
        if input_dim <= 0:
            raise ValueError("At least one item feature must be enabled")
        self.item_tower = _mlp(input_dim, data.config.hidden_dims, data.config.embedding_dim)

    def encode_users(self, user_indices: Tensor) -> Tensor:
        vectors = self.user_tower(self.user_embedding(user_indices))
        _require_finite(vectors, "user tower pre-normalization output")
        vectors = F.normalize(vectors, dim=-1)
        _require_finite(vectors, "user tower normalized output")
        return vectors

    def encode_items(
        self,
        product_id_indices: Tensor,
        categorical_indices: Mapping[str, Tensor],
        normalized_log_prices: Tensor,
    ) -> Tensor:
        parts: list[Tensor] = []
        if self.product_id_embedding is not None:
            parts.append(self.product_id_embedding(product_id_indices))
        for feature, embedding in self.categorical_embeddings.items():
            parts.append(embedding(categorical_indices[feature]))
        if self.price_projection is not None:
            _require_finite(normalized_log_prices, "normalized log product_price")
            parts.append(self.price_projection(normalized_log_prices.unsqueeze(-1)))
        fused = torch.cat(parts, dim=-1)
        _require_finite(fused, "item feature fusion input")
        vectors = self.item_tower(fused)
        _require_finite(vectors, "item tower pre-normalization output")
        vectors = F.normalize(vectors, dim=-1)
        _require_finite(vectors, "item tower normalized output")
        return vectors


class HistoryLookup:
    """Bounded LRU over a disk-backed Past-only user/product history."""

    def __init__(self, data: PreparedContentItemData, capacity: int) -> None:
        self.connection = sqlite3.connect(data.state_path)
        self.product_to_position = data.product_to_catalog_position
        self.index_to_user = {index + 2: value for index, value in enumerate(data.user_ids)}
        self.capacity = max(1, capacity)
        self.cache: OrderedDict[int, set[int]] = OrderedDict()

    def for_user(self, user_index: int) -> set[int]:
        if user_index in self.cache:
            value = self.cache.pop(user_index)
            self.cache[user_index] = value
            return value
        products = {
            self.product_to_position[str(row[0])]
            for row in self.connection.execute(
                "SELECT product_id FROM histories WHERE user_id = ?", (self.index_to_user[int(user_index)],)
            )
            if str(row[0]) in self.product_to_position
        }
        self.cache[user_index] = products
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return products

    def close(self) -> None:
        self.connection.close()


def prepare_content_item_data(config: ContentItemTwoTowerConfig, state_path: Path) -> PreparedContentItemData:
    """Build vocabularies, latest-before-cutoff catalogue, and Past histories."""

    _validate_config(config)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.unlink(missing_ok=True)
    connection = sqlite3.connect(state_path)
    _create_state_tables(connection)
    user_to_index: dict[str, int] = {}
    product_to_vocab_index: dict[str, int] = {}
    categorical_values = {
        feature: set() for feature in ITEM_CATEGORICAL_FEATURES if feature in config.enabled_features
    }
    price_issues = defaultdict(int)
    processed = 0
    sequence = 0
    started = time.monotonic()
    next_log = config.log_every_rows
    try:
        for chunk in _iter_training_chunks(config):
            users = chunk["user_id"].astype(str).tolist()
            products = chunk["product_id"].astype(str).tolist()
            for value in users:
                user_to_index.setdefault(value, 0)
            for value in products:
                if value not in product_to_vocab_index:
                    product_to_vocab_index[value] = len(product_to_vocab_index) + 2
            connection.executemany(
                "INSERT OR IGNORE INTO histories VALUES (?, ?)",
                ((user, product) for user, product in zip(users, products, strict=True)),
            )
            connection.executemany(
                "INSERT INTO catalog_products VALUES (?, 1) ON CONFLICT(product_id) DO UPDATE SET warm = 1",
                ((product,) for product in products),
            )
            _upsert_catalog_observations(
                connection,
                chunk,
                config,
                sequence_start=sequence,
                categorical_values=categorical_values,
                price_issues=price_issues,
                collect_vocab=True,
            )
            processed += len(chunk)
            sequence += len(chunk)
            connection.commit()
            if processed >= next_log:
                elapsed = time.monotonic() - started
                LOGGER.info(
                    "catalogue/vocab pass processed_rows=%s rows_per_sec=%.1f rss_mb=%.1f peak_rss_mb=%.1f",
                    processed, processed / max(elapsed, 1e-9), _rss_mb(), _peak_rss_mb(),
                )
                while next_log <= processed:
                    next_log += config.log_every_rows
        if not processed:
            raise ValueError("Content-item Two-Tower requires at least one valid Past interaction")

        sorted_users = sorted(user_to_index)
        user_to_index = {value: index + 2 for index, value in enumerate(sorted_users)}

        if config.product_catalog_path is not None:
            external_rows = 0
            for chunk in _iter_external_catalog_chunks(config):
                products = chunk["product_id"].astype(str).tolist()
                connection.executemany(
                    "INSERT OR IGNORE INTO catalog_products VALUES (?, 0)", ((product,) for product in products)
                )
                _upsert_catalog_observations(
                    connection,
                    chunk,
                    config,
                    sequence_start=sequence,
                    categorical_values=categorical_values,
                    price_issues=price_issues,
                    collect_vocab=False,
                )
                external_rows += len(chunk)
                sequence += len(chunk)
                connection.commit()
            LOGGER.info("Point-in-time external catalogue rows accepted=%s", external_rows)
        else:
            LOGGER.warning("Cold-item retrieval unavailable: catalogue contains Past-seen items only.")

        categorical_vocabs = {
            feature: (PAD_TOKEN, OOV_TOKEN, *sorted(values))
            for feature, values in categorical_values.items()
        }
        categorical_to_index = {
            feature: {value: index for index, value in enumerate(vocab)}
            for feature, vocab in categorical_vocabs.items()
        }
        product_vocab = (PAD_TOKEN, OOV_TOKEN, *[item for item, _ in sorted(product_to_vocab_index.items(), key=lambda row: row[1])])
        catalog, product_to_position = _materialize_catalog(
            connection,
            product_to_vocab_index,
            categorical_to_index,
        )
        training_positions = np.flatnonzero(catalog.warm_mask).astype(np.int64)
        if not len(training_positions):
            raise ValueError("Past training data produced an empty searchable catalogue")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "catalog_mode": config.catalog_mode,
            "split_timestamp": config.split_timestamp,
            "training_rows": processed,
            "training_users": len(user_to_index),
            "warm_products": int(catalog.warm_mask.sum()),
            "searchable_products": len(catalog.product_ids),
            "searchable_cold_products": int((~catalog.warm_mask).sum()),
            "price_invalid_counts": dict(price_issues),
            "price_normalization": {
                "mean_log1p_price": catalog.price_mean,
                "std_log1p_price": catalog.price_std,
                "fit_from": "latest-before-split warm Past products only",
            },
            "history_contract": "Past interactions are used only for negative exclusion and optional retrieval filtering; user tower has no history input.",
        }
        LOGGER.info(
            "Prepared streaming training state rows=%s users=%s warm_products=%s searchable_products=%s rss_mb=%.1f peak_rss_mb=%.1f",
            processed,
            len(user_to_index),
            len(training_positions),
            len(catalog.product_ids),
            _rss_mb(),
            _peak_rss_mb(),
        )
        return PreparedContentItemData(
            config=config,
            state_path=state_path,
            user_ids=np.asarray(sorted_users, dtype=str),
            user_to_index=user_to_index,
            product_vocab=product_vocab,
            product_to_vocab_index={value: index for index, value in enumerate(product_vocab)},
            categorical_vocabs=categorical_vocabs,
            categorical_to_index=categorical_to_index,
            catalog=catalog,
            product_to_catalog_position=product_to_position,
            training_product_positions=training_positions,
            training_rows=processed,
            metadata=metadata,
        )
    finally:
        connection.close()


def train_variant(
    data: PreparedContentItemData,
    variant: str,
    device: torch.device,
) -> tuple[ContentItemTwoTowerModel, list[float], dict[str, float]]:
    """Train one fair-ablation variant with the shared streaming objective."""

    _set_seed(data.config.seed)
    model = ContentItemTwoTowerModel(data, variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=data.config.learning_rate)
    loader = DataLoader(StreamingSampledNegativeDataset(data), batch_size=data.config.batch_size, num_workers=0)
    losses: list[float] = []
    started = time.monotonic()
    batches = 0
    rows = 0
    next_log = data.config.log_every_rows
    model.train()
    for epoch in range(1, data.config.epochs + 1):
        total_loss = 0.0
        epoch_rows = 0
        for users, positives, negatives, weights in loader:
            users = users.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            weights = weights.to(device)
            user_vectors = model.encode_users(users)
            positive_vectors = _encode_catalog_positions(model, data.catalog, positives, device)
            negative_vectors = _encode_catalog_positions(model, data.catalog, negatives, device)
            positive_logits = (user_vectors * positive_vectors).sum(-1, keepdim=True)
            negative_logits = torch.einsum("bd,bnd->bn", user_vectors, negative_vectors)
            logits = torch.cat((positive_logits, negative_logits), dim=1)
            _require_finite(logits, f"{variant} sampled-softmax logits")
            per_example = F.cross_entropy(
                logits,
                torch.zeros(len(users), dtype=torch.long, device=device),
                reduction="none",
            )
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1.0)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{variant} produced non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(users)
            total_loss += float(loss.detach()) * count
            epoch_rows += count
            rows += count
            batches += 1
            if rows >= next_log:
                elapsed = time.monotonic() - started
                LOGGER.info(
                    "%s training rows=%s batches=%s rows_per_sec=%.1f rss_mb=%.1f peak_rss_mb=%.1f",
                    variant, rows, batches, rows / max(elapsed, 1e-9), _rss_mb(), _peak_rss_mb(),
                )
                next_log += data.config.log_every_rows
        epoch_loss = total_loss / max(epoch_rows, 1)
        if not math.isfinite(epoch_loss):
            raise FloatingPointError(f"{variant} produced non-finite epoch loss")
        losses.append(epoch_loss)
        LOGGER.info("%s epoch=%s/%s sampled-softmax_loss=%.6f", variant, epoch, data.config.epochs, epoch_loss)
    elapsed = time.monotonic() - started
    return model, losses, {"train_time": elapsed, "peak_rss_mb": _peak_rss_mb()}


@torch.no_grad()
def extract_catalog_embeddings(
    model: ContentItemTwoTowerModel,
    catalog: CatalogFeatures,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    result = np.empty((len(catalog.product_ids), model.config.embedding_dim), dtype=np.float32)
    for start in range(0, len(catalog.product_ids), batch_size):
        stop = min(start + batch_size, len(catalog.product_ids))
        positions = torch.arange(start, stop, dtype=torch.long, device=device)
        result[start:stop] = _encode_catalog_positions(model, catalog, positions, device).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Catalog embedding export produced NaN or infinity")
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Catalog embeddings are not L2-normalized")
    return result


def save_variant_checkpoint(
    model: ContentItemTwoTowerModel,
    data: PreparedContentItemData,
    path: Path,
    losses: Sequence[float],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "variant": model.variant,
        "state_dict": model.state_dict(),
        "model_config": _model_config_payload(data.config, model.variant),
        "user_ids": data.user_ids,
        "product_vocab": list(data.product_vocab),
        "categorical_vocabs": {key: list(value) for key, value in data.categorical_vocabs.items()},
        "catalog_fingerprint": data.catalog.fingerprint,
        "catalog_product_ids": data.catalog.product_ids,
        "metadata": data.metadata | {"losses": list(losses)},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return _file_sha256(path)


def load_variant_checkpoint(
    data: PreparedContentItemData,
    variant: str,
    path: Path,
    device: torch.device,
) -> tuple[ContentItemTwoTowerModel, Mapping[str, Any], str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("variant") != variant:
        raise ValueError("Content-item checkpoint schema/variant mismatch")
    expected_config = _model_config_payload(data.config, variant)
    if checkpoint.get("model_config") != expected_config:
        raise ValueError("Content-item checkpoint model configuration mismatch; retraining is required")
    if list(checkpoint.get("product_vocab", [])) != list(data.product_vocab):
        raise ValueError("Content-item checkpoint product vocabulary mismatch; retraining is required")
    expected_categorical = {key: list(value) for key, value in data.categorical_vocabs.items()}
    if checkpoint.get("categorical_vocabs") != expected_categorical:
        raise ValueError("Content-item checkpoint categorical vocabulary mismatch; retraining is required")
    if checkpoint.get("catalog_fingerprint") != data.catalog.fingerprint:
        raise ValueError("Content-item checkpoint catalogue mismatch; retraining is required")
    model = ContentItemTwoTowerModel(data, variant).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint, _file_sha256(path)


def load_or_build_fingerprinted_index(
    model: ContentItemTwoTowerModel,
    data: PreparedContentItemData,
    checkpoint_path: Path,
    checkpoint_fingerprint: str,
    index_path: Path,
    device: torch.device,
) -> tuple[Any, float, bool, list[str]]:
    expected = _expected_index_metadata(model, data, checkpoint_fingerprint)
    reasons: list[str] = []
    if index_path.is_file():
        try:
            actual = load_faiss_index_metadata(index_path)
            reasons = fingerprint_mismatch_reasons(expected, actual)
            if not reasons:
                index, product_ids = load_faiss_index(index_path)
                if np.array_equal(product_ids, data.catalog.product_ids):
                    if hasattr(index, "hnsw"):
                        index.hnsw.efSearch = data.config.ef_search
                    LOGGER.info("Reusing fingerprint-matched FAISS index %s", index_path)
                    return index, 0.0, False, []
                reasons.append("product_ids")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            reasons = [f"unreadable metadata: {error}"]
        LOGGER.warning("Rebuilding FAISS index because metadata changed: %s", ", ".join(reasons))
    else:
        reasons = ["index missing"]
    embeddings = extract_catalog_embeddings(model, data.catalog, data.config.inference_batch_size, device)
    started = time.monotonic()
    index = build_faiss_index(
        embeddings,
        data.config.faiss_index_type,
        hnsw_m=data.config.hnsw_m,
        ef_construction=data.config.ef_construction,
        ef_search=data.config.ef_search,
    )
    build_time = time.monotonic() - started
    save_faiss_index(index, data.catalog.product_ids, index_path, metadata=expected)
    LOGGER.info(
        "Built FAISS index variant=%s products=%s seconds=%.2f rss_mb=%.1f peak_rss_mb=%.1f checkpoint=%s",
        model.variant, index.ntotal, build_time, _rss_mb(), _peak_rss_mb(), checkpoint_path,
    )
    return index, build_time, True, reasons


@torch.no_grad()
def stream_content_item_candidates(
    model: ContentItemTwoTowerModel,
    data: PreparedContentItemData,
    index: Any,
    output_path: Path,
    device: torch.device,
) -> dict[str, float | int]:
    """Batch FAISS search and immediately stream the current batch to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    lookup = HistoryLookup(data, data.config.history_cache_users)
    processed = written = batches = 0
    search_seconds = 0.0
    started = time.monotonic()
    next_log = data.config.log_every_rows
    model.eval()
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            for start in range(0, len(data.user_ids), data.config.search_batch_size):
                stop = min(start + data.config.search_batch_size, len(data.user_ids))
                user_indices = torch.arange(start + 2, stop + 2, dtype=torch.long, device=device)
                vectors = model.encode_users(user_indices).cpu().numpy().astype(np.float32, copy=False)
                histories = [lookup.for_user(index) if data.config.exclude_seen_items else set() for index in range(start + 2, stop + 2)]
                search_k = min(
                    int(index.ntotal),
                    max(data.config.top_k, int(math.ceil(data.config.top_k * data.config.retrieval_oversample_ratio))),
                )
                while True:
                    search_started = time.monotonic()
                    scores, positions = search_faiss_index(index, vectors, search_k)
                    search_seconds += time.monotonic() - search_started
                    enough = all(
                        sum(int(position) >= 0 and int(position) not in histories[row] for position in positions[row])
                        >= min(data.config.top_k, int(index.ntotal) - len(histories[row]))
                        for row in range(len(histories))
                    )
                    if enough or search_k >= int(index.ntotal):
                        break
                    search_k = min(int(index.ntotal), max(search_k + 1, search_k * 2))
                for relative, user_id in enumerate(data.user_ids[start:stop]):
                    rank = 0
                    seen = histories[relative]
                    for score, position in zip(scores[relative], positions[relative], strict=True):
                        item = int(position)
                        if item < 0 or item in seen:
                            continue
                        rank += 1
                        writer.writerow((str(user_id), str(data.catalog.product_ids[item]), float(score), rank))
                        written += 1
                        if rank >= data.config.top_k:
                            break
                processed = stop
                batches += 1
                if processed >= next_log or processed == len(data.user_ids):
                    elapsed = time.monotonic() - started
                    LOGGER.info(
                        "%s retrieval users=%s batches=%s candidate_rows=%s users_per_sec=%.1f faiss_search_seconds=%.2f rss_mb=%.1f peak_rss_mb=%.1f",
                        model.variant, processed, batches, written, processed / max(elapsed, 1e-9), search_seconds, _rss_mb(), _peak_rss_mb(),
                    )
                    next_log += data.config.log_every_rows
        temporary.replace(output_path)
    finally:
        lookup.close()
    elapsed = time.monotonic() - started
    return {
        "processed_users": processed,
        "candidate_rows_written": written,
        "retrieval_time": elapsed,
        "faiss_search_time": search_seconds,
        "peak_rss_mb": _peak_rss_mb(),
    }


def evaluate_content_item_candidates(
    candidate_path: Path,
    future_path: Path,
    data: PreparedContentItemData,
) -> dict[str, Any]:
    """Evaluate overall, searchable warm, and searchable cold temporal recall."""

    positives = _future_positives(future_path, data.config.input_chunk_size)
    warm = set(data.catalog.product_ids[data.catalog.warm_mask].tolist())
    searchable = set(data.catalog.product_ids.tolist())
    cold = searchable - warm
    totals = {
        "overall": positives,
        "warm": {user: values & warm for user, values in positives.items() if values & warm},
        "cold": {user: values & cold for user, values in positives.items() if values & cold},
    }
    recall_sums = {label: {cutoff: 0.0 for cutoff in EVALUATION_CUTOFFS} for label in totals}
    hit_sums = {label: {cutoff: 0 for cutoff in EVALUATION_CUTOFFS} for label in totals}
    unique_items: set[str] = set()
    for user, candidates in _candidate_groups(candidate_path, data.config.input_chunk_size):
        unique_items.update(item for item, _ in candidates)
        ranks = {item: rank for item, rank in candidates}
        for label, targets in totals.items():
            target = targets.get(user)
            if not target:
                continue
            for cutoff in EVALUATION_CUTOFFS:
                matched = sum(ranks.get(item, 10**12) <= cutoff for item in target)
                recall_sums[label][cutoff] += matched / len(target)
                hit_sums[label][cutoff] += int(matched > 0)
    result: dict[str, Any] = {
        "users_evaluated": len(positives),
        "searchable_catalog_items": len(searchable),
        "unique_recalled_items": len(unique_items),
        "catalog_coverage": len(unique_items & searchable) / len(searchable) if searchable else 0.0,
        "catalog_mode": data.config.catalog_mode,
    }
    for cutoff in EVALUATION_CUTOFFS:
        result[f"recall@{cutoff}"] = recall_sums["overall"][cutoff] / len(positives) if positives else 0.0
        result[f"hit_rate@{cutoff}"] = hit_sums["overall"][cutoff] / len(positives) if positives else 0.0
        warm_users = len(totals["warm"])
        result[f"warm_recall@{cutoff}"] = recall_sums["warm"][cutoff] / warm_users if warm_users else None
        cold_users = len(totals["cold"])
        result[f"cold_recall@{cutoff}"] = (
            recall_sums["cold"][cutoff] / cold_users
            if data.config.catalog_mode == "external_point_in_time" and cold_users
            else None
        )
    result["warm_positive_users"] = len(totals["warm"])
    result["searchable_cold_positive_users"] = len(totals["cold"])
    result["cold_retrieval_available"] = data.config.catalog_mode == "external_point_in_time"
    return result


def run_content_item_ablation(config: ContentItemTwoTowerConfig) -> dict[str, Any]:
    """Run the strict two-row ID-only/content-item temporal experiment."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(config.device)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="content-item-state-", dir=config.output_dir) as temporary:
        data = prepare_content_item_data(config, Path(temporary) / "state.sqlite")
        for variant in ("id_only", "content_item"):
            checkpoint_path = config.output_dir / "models" / f"two_tower_{variant}.pt"
            candidate_path = config.output_dir / "recall_candidates" / f"two_tower_{variant}_topk.csv"
            index_path = config.output_dir / "recall_candidates" / f"faiss_{variant}_product_index"
            if config.train or not checkpoint_path.is_file():
                model, losses, train_stats = train_variant(data, variant, device)
                checkpoint_fingerprint = save_variant_checkpoint(model, data, checkpoint_path, losses)
            else:
                model, checkpoint, checkpoint_fingerprint = load_variant_checkpoint(
                    data, variant, checkpoint_path, device
                )
                losses = list(checkpoint.get("metadata", {}).get("losses", []))
                train_stats = {"train_time": 0.0, "peak_rss_mb": _peak_rss_mb()}
            index, faiss_build_time, rebuilt, rebuild_reasons = load_or_build_fingerprinted_index(
                model, data, checkpoint_path, checkpoint_fingerprint, index_path, device
            )
            retrieval = stream_content_item_candidates(model, data, index, candidate_path, device)
            metrics = evaluate_content_item_candidates(candidate_path, config.future_path, data)
            row = {
                "model": variant,
                "catalog_mode": config.catalog_mode,
                "recall@50": metrics["recall@50"],
                "recall@100": metrics["recall@100"],
                "recall@200": metrics["recall@200"],
                "hit_rate@100": metrics["hit_rate@100"],
                "warm_recall@100": metrics["warm_recall@100"],
                "cold_recall@100": metrics["cold_recall@100"],
                "unique_recalled_items": metrics["unique_recalled_items"],
                "catalog_coverage": metrics["catalog_coverage"],
                "train_time": train_stats["train_time"],
                "retrieval_time": retrieval["retrieval_time"],
                "peak_rss_mb": max(float(train_stats["peak_rss_mb"]), float(retrieval["peak_rss_mb"])),
            }
            rows.append(row)
            details[variant] = {
                "losses": losses,
                "metrics": metrics,
                "checkpoint": str(checkpoint_path),
                "candidates": str(candidate_path),
                "faiss_build_time": faiss_build_time,
                "faiss_rebuilt": rebuilt,
                "faiss_rebuild_reasons": rebuild_reasons,
                "retrieval": retrieval,
            }
        metrics_dir = config.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(metrics_dir / "content_item_ablation.csv", index=False)
        report = {
            "schema_version": SCHEMA_VERSION,
            "comparison_contract": "Same Past positives, sampled negatives, ID-only user tower, loss, catalogue, FAISS settings, and Future-A cohort; only the item tower differs.",
            "data": data.metadata,
            "models": details,
            "ablation_rows": rows,
            "future_b_read_for_model_selection": False,
        }
        (metrics_dir / "content_item_ablation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report


def fingerprint_mismatch_reasons(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Return every semantic fingerprint field that prevents index reuse."""

    keys = (
        "checkpoint_fingerprint",
        "model_config_fingerprint",
        "item_vocab_fingerprint",
        "catalogue_fingerprint",
        "embedding_dim",
        "normalized",
    )
    return [key for key in keys if actual.get(key) != expected.get(key)]


def preprocess_prices(values: pd.Series) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Explicitly sanitize raw prices and return log1p(max(price, 0))."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    missing = np.isnan(numeric)
    positive_inf = np.isposinf(numeric)
    negative_inf = np.isneginf(numeric)
    negative = np.isfinite(numeric) & (numeric < 0)
    valid_observation = ~(missing | positive_inf | negative_inf)
    safe = numeric.copy()
    safe[missing | positive_inf | negative_inf] = 0.0
    safe[negative] = 0.0
    transformed = np.log1p(safe)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("product_price preprocessing produced NaN or infinity")
    return transformed.astype(np.float32), {
        "nan_or_non_numeric": int(missing.sum()),
        "positive_inf": int(positive_inf.sum()),
        "negative_inf": int(negative_inf.sum()),
        "negative": int(negative.sum()),
    }, valid_observation


def _create_state_tables(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE catalog_products (product_id TEXT PRIMARY KEY, warm INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE catalog_values (product_id TEXT NOT NULL, feature TEXT NOT NULL, value TEXT NOT NULL, event_ts INTEGER NOT NULL, sequence INTEGER NOT NULL, PRIMARY KEY(product_id, feature))"
    )
    connection.execute(
        "CREATE TABLE histories (user_id TEXT NOT NULL, product_id TEXT NOT NULL, PRIMARY KEY(user_id, product_id)) WITHOUT ROWID"
    )
    connection.execute("CREATE INDEX histories_user ON histories(user_id)")


def _upsert_catalog_observations(
    connection: sqlite3.Connection,
    chunk: pd.DataFrame,
    config: ContentItemTwoTowerConfig,
    *,
    sequence_start: int,
    categorical_values: dict[str, set[str]],
    price_issues: defaultdict[str, int],
    collect_vocab: bool,
) -> None:
    products = chunk["product_id"].astype(str).to_numpy()
    timestamps = pd.to_numeric(chunk["click_timestamp"], errors="raise").to_numpy(dtype=np.int64)
    sequences = np.arange(sequence_start, sequence_start + len(chunk), dtype=np.int64)
    statement = (
        "INSERT INTO catalog_values VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(product_id, feature) DO UPDATE SET value=excluded.value, event_ts=excluded.event_ts, sequence=excluded.sequence "
        "WHERE excluded.event_ts > catalog_values.event_ts OR "
        "(excluded.event_ts = catalog_values.event_ts AND excluded.sequence > catalog_values.sequence)"
    )
    for feature in categorical_values:
        if feature not in chunk:
            continue
        values = chunk[feature].astype("string")
        valid = values.notna() & values.str.strip().ne("")
        clean = values.fillna("").astype(str).str.strip().to_numpy()
        if collect_vocab:
            categorical_values[feature].update(clean[valid.to_numpy()].tolist())
        connection.executemany(
            statement,
            (
                (products[index], feature, clean[index], int(timestamps[index]), int(sequences[index]))
                for index in np.flatnonzero(valid.to_numpy())
            ),
        )
    if "product_price" in config.enabled_features:
        raw_prices = chunk.get("product_price", pd.Series(np.nan, index=chunk.index))
        transformed, issues, valid_price = preprocess_prices(raw_prices)
        for key, value in issues.items():
            price_issues[key] += value
        if any(issues.values()):
            LOGGER.warning("Explicitly sanitized product_price values: %s", issues)
        connection.executemany(
            statement,
            (
                (products[index], "product_price", repr(float(transformed[index])), int(timestamps[index]), int(sequences[index]))
                for index in np.flatnonzero(valid_price)
            ),
        )


def _materialize_catalog(
    connection: sqlite3.Connection,
    product_to_vocab_index: Mapping[str, int],
    categorical_to_index: Mapping[str, Mapping[str, int]],
) -> tuple[CatalogFeatures, dict[str, int]]:
    categorical_features = tuple(categorical_to_index)
    clauses = [
        f"MAX(CASE WHEN v.feature='{feature}' THEN v.value END) AS '{feature}'"
        for feature in categorical_features + ("product_price",)
    ]
    query = (
        "SELECT p.product_id, p.warm, " + ", ".join(clauses)
        + " FROM catalog_products p LEFT JOIN catalog_values v ON p.product_id=v.product_id "
        "GROUP BY p.product_id, p.warm ORDER BY p.product_id"
    )
    rows = connection.execute(query).fetchall()
    columns = ["product_id", "warm", *categorical_features, "product_price"]
    frame = pd.DataFrame(rows, columns=columns)
    product_ids = frame["product_id"].astype(str).to_numpy()
    warm_mask = frame["warm"].astype(bool).to_numpy()
    product_indices = np.asarray(
        [product_to_vocab_index.get(value, OOV_INDEX) for value in product_ids], dtype=np.int64
    )
    categorical_indices = {
        feature: np.asarray(
            [
                categorical_to_index[feature].get(str(value), OOV_INDEX)
                if pd.notna(value) and str(value).strip()
                else OOV_INDEX
                for value in frame[feature]
            ],
            dtype=np.int64,
        )
        for feature in categorical_features
    }
    log_prices = pd.to_numeric(frame["product_price"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    warm_prices = log_prices[warm_mask]
    mean = float(warm_prices.mean()) if len(warm_prices) else 0.0
    std = float(warm_prices.std()) if len(warm_prices) else 1.0
    std = max(std, 1e-6)
    normalized = ((log_prices - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Normalized product_price catalogue contains NaN or infinity")
    fingerprint = _catalog_fingerprint(product_ids, product_indices, categorical_indices, normalized, warm_mask)
    catalog = CatalogFeatures(
        product_ids=product_ids,
        product_id_indices=product_indices,
        categorical_indices=categorical_indices,
        normalized_log_prices=normalized,
        warm_mask=warm_mask,
        price_mean=mean,
        price_std=std,
        fingerprint=fingerprint,
    )
    return catalog, {value: index for index, value in enumerate(product_ids)}


def _iter_training_chunks(config: ContentItemTwoTowerConfig) -> Iterator[pd.DataFrame]:
    required = {"user_id", "product_id", "conversion_label", "click_timestamp"}
    remaining = config.max_train_rows
    for path in _input_csv_files(config.input_path):
        header = set(pd.read_csv(path, nrows=0).columns)
        if missing := required - header:
            raise ValueError(f"Past interaction input {path} is missing required columns: {sorted(missing)}")
        active_categorical = set(config.enabled_features).intersection(ITEM_CATEGORICAL_FEATURES)
        usecols = list(required | ({"product_price"} if "product_price" in config.enabled_features and "product_price" in header else set()) | (active_categorical & header))
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.input_chunk_size, low_memory=False):
            chunk["user_id"] = chunk["user_id"].astype("string").str.strip()
            chunk["product_id"] = chunk["product_id"].astype("string").str.strip()
            timestamps = pd.to_numeric(chunk["click_timestamp"], errors="coerce")
            valid = (
                chunk["user_id"].notna()
                & chunk["user_id"].ne("")
                & chunk["product_id"].notna()
                & chunk["product_id"].ne("")
                & timestamps.notna()
                & timestamps.le(config.split_timestamp)
            )
            chunk = chunk.loc[valid].copy()
            if chunk.empty:
                continue
            labels = pd.to_numeric(chunk["conversion_label"], errors="raise")
            if not labels.isin([0, 1]).all():
                raise ValueError("conversion_label must be binary")
            chunk["conversion_label"] = labels.astype(np.int8)
            chunk["click_timestamp"] = pd.to_numeric(chunk["click_timestamp"], errors="raise").astype(np.int64)
            if remaining is not None:
                chunk = chunk.iloc[:remaining].copy()
                remaining -= len(chunk)
            if not chunk.empty:
                yield chunk
            if remaining is not None and remaining <= 0:
                return


def _iter_external_catalog_chunks(config: ContentItemTwoTowerConfig) -> Iterator[pd.DataFrame]:
    assert config.product_catalog_path is not None
    normalized_parts = {part.lower().replace("-", "_") for part in config.product_catalog_path.parts}
    if normalized_parts.intersection({"future", "future_a", "future_b"}):
        raise ValueError("External product catalogue cannot be sourced from Future/Future-A/Future-B")
    if config.product_catalog_as_of_timestamp is None:
        raise ValueError("External product catalogue requires product_catalog_as_of_timestamp")
    if config.product_catalog_as_of_timestamp > config.split_timestamp:
        raise ValueError("External product catalogue as-of timestamp exceeds the temporal split timestamp")
    for path in _input_csv_files(config.product_catalog_path):
        header = set(pd.read_csv(path, nrows=0).columns)
        if "product_id" not in header:
            raise ValueError(f"External catalogue {path} is missing product_id")
        usecols = [
            "product_id",
            *(feature for feature in ITEM_CATEGORICAL_FEATURES if feature in config.enabled_features and feature in header),
        ]
        if "product_price" in config.enabled_features and "product_price" in header:
            usecols.append("product_price")
        if "click_timestamp" in header:
            usecols.append("click_timestamp")
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.input_chunk_size, low_memory=False):
            chunk["product_id"] = chunk["product_id"].astype("string").str.strip()
            valid = chunk["product_id"].notna() & chunk["product_id"].ne("")
            if "click_timestamp" in chunk:
                timestamps = pd.to_numeric(chunk["click_timestamp"], errors="coerce")
                valid &= timestamps.notna() & timestamps.le(config.split_timestamp)
                chunk["click_timestamp"] = timestamps
            else:
                chunk["click_timestamp"] = int(config.product_catalog_as_of_timestamp)
            chunk = chunk.loc[valid].copy()
            if not chunk.empty:
                chunk["click_timestamp"] = pd.to_numeric(chunk["click_timestamp"], errors="raise").astype(np.int64)
                yield chunk


def _encode_catalog_positions(
    model: ContentItemTwoTowerModel,
    catalog: CatalogFeatures,
    positions: Tensor,
    device: torch.device,
) -> Tensor:
    numpy_positions = positions.detach().cpu().numpy()
    product_ids = torch.as_tensor(catalog.product_id_indices[numpy_positions], dtype=torch.long, device=device)
    categories = {
        feature: torch.as_tensor(values[numpy_positions], dtype=torch.long, device=device)
        for feature, values in catalog.categorical_indices.items()
        if feature in model.categorical_embeddings
    }
    prices = torch.as_tensor(catalog.normalized_log_prices[numpy_positions], dtype=torch.float32, device=device)
    return model.encode_items(product_ids, categories, prices)


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(current, int(hidden)), nn.ReLU()))
        current = int(hidden)
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _require_finite(values: Tensor, context: str) -> None:
    if not bool(torch.isfinite(values).all()):
        count = int((~torch.isfinite(values)).sum().detach().cpu())
        raise FloatingPointError(f"{context} contains {count} non-finite values; shape={tuple(values.shape)}")


def _future_positives(path: Path, chunk_size: int) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for file in _input_csv_files(path):
        for chunk in pd.read_csv(file, usecols=["user_id", "product_id"], chunksize=chunk_size, low_memory=False):
            for user, product in chunk.itertuples(index=False, name=None):
                if pd.isna(user) or pd.isna(product):
                    continue
                user_id, product_id = str(user).strip(), str(product).strip()
                if user_id and product_id:
                    result[user_id].add(product_id)
    return result


def _candidate_groups(path: Path, chunk_size: int) -> Iterator[tuple[str, list[tuple[str, int]]]]:
    current: str | None = None
    rows: list[tuple[str, int]] = []
    completed: set[str] = set()
    for chunk in pd.read_csv(
        path, usecols=["user_id", "candidate_ad_id", "rank"], chunksize=chunk_size, low_memory=False
    ):
        for user, item, rank in chunk.itertuples(index=False, name=None):
            user_id, item_id = str(user).strip(), str(item).strip()
            if current is not None and user_id != current:
                if current in completed:
                    raise ValueError("Candidate rows must keep each user contiguous")
                completed.add(current)
                yield current, rows
                rows = []
            current = user_id
            rows.append((item_id, int(rank)))
    if current is not None:
        yield current, rows


def _expected_index_metadata(
    model: ContentItemTwoTowerModel,
    data: PreparedContentItemData,
    checkpoint_fingerprint: str,
) -> dict[str, Any]:
    return {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "model_config_fingerprint": _sha_json(_model_config_payload(data.config, model.variant)),
        "item_vocab_fingerprint": _sha_json(list(data.product_vocab)),
        "catalogue_fingerprint": data.catalog.fingerprint,
        "embedding_dim": data.config.embedding_dim,
        "normalized": True,
    }


def _model_config_payload(config: ContentItemTwoTowerConfig, variant: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "embedding_dim": config.embedding_dim,
        "feature_embedding_dim": config.feature_embedding_dim,
        "price_embedding_dim": config.price_embedding_dim,
        "hidden_dims": list(config.hidden_dims),
        "enabled_features": list(("product_id",) if variant == "id_only" else config.enabled_features),
    }


def _catalog_fingerprint(
    product_ids: np.ndarray,
    product_indices: np.ndarray,
    categories: Mapping[str, np.ndarray],
    prices: np.ndarray,
    warm_mask: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for product_id in product_ids:
        digest.update(str(product_id).encode())
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(product_indices).tobytes())
    for feature in sorted(categories):
        digest.update(feature.encode())
        digest.update(np.ascontiguousarray(categories[feature]).tobytes())
    digest.update(np.ascontiguousarray(prices).tobytes())
    digest.update(np.ascontiguousarray(warm_mask).tobytes())
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except ImportError:  # pragma: no cover - dependency fallback
        return _peak_rss_mb()


def _peak_rss_mb() -> float:
    import resource
    import sys

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: ContentItemTwoTowerConfig) -> None:
    unknown = set(config.enabled_features) - set(ALL_ITEM_FEATURES)
    if unknown:
        raise ValueError(f"Unknown item features: {sorted(unknown)}")
    if not config.enabled_features:
        raise ValueError("enabled_features cannot be empty")
    positive = (
        config.embedding_dim,
        config.feature_embedding_dim,
        config.price_embedding_dim,
        config.batch_size,
        config.epochs,
        config.negative_samples,
        config.top_k,
        config.search_batch_size,
        config.inference_batch_size,
        config.input_chunk_size,
        config.log_every_rows,
    )
    if min(positive) <= 0 or any(int(value) <= 0 for value in config.hidden_dims):
        raise ValueError("Content-item Two-Tower dimensions, sizes, and counts must be positive")
    if config.learning_rate <= 0 or config.retrieval_oversample_ratio < 1.0:
        raise ValueError("learning_rate must be positive and retrieval_oversample_ratio must be >= 1")
    if config.click_weight <= 0 or config.conversion_weight <= 0:
        raise ValueError("interaction weights must be positive")
    if config.max_train_rows is not None and config.max_train_rows <= 0:
        raise ValueError("max_train_rows must be positive when set")
    if config.faiss_index_type not in {"flat", "hnsw"}:
        raise ValueError("faiss_index_type must be flat or hnsw")
    if config.product_catalog_path is not None:
        if config.product_catalog_as_of_timestamp is None:
            raise ValueError("External product catalogue requires product_catalog_as_of_timestamp")
        if config.product_catalog_as_of_timestamp > config.split_timestamp:
            raise ValueError("External product catalogue as-of timestamp exceeds split timestamp")
