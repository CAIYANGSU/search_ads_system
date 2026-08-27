"""Offline Two Tower training and FAISS candidate retrieval for product ads."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.recall.faiss_index import (
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
    search_faiss_index,
)

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ("user_id", "candidate_ad_id", "two_tower_score", "rank")


@dataclass(frozen=True)
class TwoTowerRecallConfig:
    input_path: Path
    output_path: Path
    index_path: Path
    checkpoint_path: Path
    user_id_column: str = "user_id"
    product_id_column: str = "product_id"
    conversion_label_column: str = "conversion_label"
    embedding_dim: int = 64
    batch_size: int = 4096
    epochs: int = 3
    learning_rate: float = 1e-3
    top_k: int = 200
    negative_samples: int = 5
    click_weight: float = 1.0
    conversion_weight: float = 3.0
    seed: int = 2026
    device: str = "auto"
    faiss_index_type: str = "hnsw"
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    train: bool = False
    max_users: Optional[int] = None
    search_batch_size: int = 10_000
    inference_batch_size: int = 4096
    input_chunk_size: int = 200_000
    max_train_rows: Optional[int] = None


class TwoTowerModel(nn.Module):
    """Separate user and advertisement towers with 64-dimensional outputs."""

    def __init__(self, num_users: int, num_ads: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if min(num_users, num_ads, embedding_dim) <= 0:
            raise ValueError("num_users, num_ads, and embedding_dim must be positive")
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.ad_embedding = nn.Embedding(num_ads, embedding_dim)
        self.user_tower = _tower(embedding_dim)
        self.ad_tower = _tower(embedding_dim)

    def encode_users(self, user_ids: Tensor) -> Tensor:
        return functional.normalize(self.user_tower(self.user_embedding(user_ids)), dim=-1)

    def encode_ads(self, ad_ids: Tensor) -> Tensor:
        return functional.normalize(self.ad_tower(self.ad_embedding(ad_ids)), dim=-1)

    def forward(self, user_ids: Tensor, ad_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode_users(user_ids), self.encode_ads(ad_ids)


def _tower(embedding_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, embedding_dim)
    )


class NegativeSamplingDataset(Dataset[tuple[int, int, np.ndarray, float]]):
    """Generates deterministic per-positive negatives outside each user's history."""

    def __init__(
        self, user_indices: np.ndarray, ad_indices: np.ndarray, weights: np.ndarray,
        user_histories: list[set[int]], num_ads: int, negative_samples: int, seed: int,
    ) -> None:
        self.user_indices = user_indices.astype(np.int64, copy=False)
        self.ad_indices = ad_indices.astype(np.int64, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.user_histories = user_histories
        self.num_ads = num_ads
        self.negative_samples = negative_samples
        self.seed = seed
        self.valid_positions = np.asarray(
            [i for i, user in enumerate(self.user_indices) if len(user_histories[user]) < num_ads], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.valid_positions)

    def __getitem__(self, position: int) -> tuple[int, int, np.ndarray, float]:
        source_position = int(self.valid_positions[position])
        user = int(self.user_indices[source_position])
        history = self.user_histories[user]
        rng = np.random.default_rng(self.seed + source_position)
        negatives: list[int] = []
        # Rejection sampling is fast for sparse interaction histories, which is
        # the normal ads-recall case.  The fallback preserves correctness for
        # unusually dense users.
        attempts = 0
        while len(negatives) < self.negative_samples and attempts < self.negative_samples * 20:
            candidate = int(rng.integers(self.num_ads))
            if candidate not in history:
                negatives.append(candidate)
            attempts += 1
        if len(negatives) < self.negative_samples:
            available = np.fromiter((ad for ad in range(self.num_ads) if ad not in history), dtype=np.int64)
            negatives.extend(rng.choice(available, self.negative_samples - len(negatives), replace=True).tolist())
        return user, int(self.ad_indices[source_position]), np.asarray(negatives, dtype=np.int64), float(self.weights[source_position])


def parse_two_tower_config(raw_config: Mapping[str, Any], config_path: Path) -> TwoTowerRecallConfig:
    """Parse and validate the ``recall.two_tower`` settings."""

    try:
        paths = raw_config["paths"]
        recall = raw_config["recall"]
    except KeyError as error:
        raise ValueError("Configuration must define paths and recall") from error
    if not isinstance(paths, Mapping) or not isinstance(recall, Mapping):
        raise ValueError("paths and recall configuration must be mappings")
    options = recall.get("two_tower", {})
    if not isinstance(options, Mapping):
        raise ValueError("recall.two_tower configuration must be a mapping")
    root = config_path.parent.resolve()
    output_root = resolve_path(str(paths["outputs_dir"]), root)
    raw_weights = options.get("interaction_weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("recall.two_tower.interaction_weights must be a mapping")
    raw_max_users = options.get("max_users")
    config = TwoTowerRecallConfig(
        input_path=resolve_path(str(options.get("input_path", paths["unified_data"])), root),
        output_path=resolve_path(str(options.get("output_path", "outputs/recall_candidates/two_tower_topk.csv")), root),
        index_path=resolve_path(str(options.get("index_path", "outputs/recall_candidates/faiss_ad_index")), root),
        checkpoint_path=resolve_path(str(options.get("checkpoint_path", "outputs/recall_candidates/two_tower_checkpoint.pt")), root),
        user_id_column=str(options.get("user_id_column", "user_id")),
        product_id_column=str(options.get("product_id_column", "product_id")),
        conversion_label_column=str(options.get("conversion_label_column", "conversion_label")),
        embedding_dim=int(options.get("embedding_dim", 64)), batch_size=int(options.get("batch_size", 4096)),
        epochs=int(options.get("epochs", 3)), learning_rate=float(options.get("learning_rate", 1e-3)),
        top_k=int(options.get("top_k", recall.get("top_k", 200))),
        negative_samples=int(options.get("negative_samples", 5)),
        click_weight=float(raw_weights.get("click", options.get("click_weight", 1.0))),
        conversion_weight=float(raw_weights.get("conversion", options.get("conversion_weight", 3.0))),
        seed=int(options.get("seed", raw_config.get("project", {}).get("seed", 2026))),
        device=str(options.get("device", "auto")),
        faiss_index_type=str(_faiss_options(options).get("index_type", "hnsw")),
        hnsw_m=int(_faiss_options(options).get("hnsw_m", 32)),
        ef_construction=int(_faiss_options(options).get("ef_construction", 200)),
        ef_search=int(_faiss_options(options).get("ef_search", 64)),
        # ``train`` is the public inference/training switch.  The fallback
        # retains support for configs written before this option existed.
        train=bool(options.get("train", not bool(options.get("reuse_checkpoint", True)))),
        max_users=None if raw_max_users is None else int(raw_max_users),
        search_batch_size=int(options.get("search_batch_size", 10_000)),
        inference_batch_size=int(options.get("inference_batch_size", 4096)),
        input_chunk_size=int(options.get("input_chunk_size", 200_000)),
    )
    for path in (config.output_path, config.index_path, config.checkpoint_path):
        try:
            path.relative_to(output_root)
        except ValueError as error:
            raise ValueError("Two Tower output paths must be within paths.outputs_dir") from error
    _validate_config(config)
    return config


def load_interactions(config: TwoTowerRecallConfig) -> pd.DataFrame:
    """Load Criteo click/conversion interactions from a CSV file or part directory."""

    columns = [config.user_id_column, config.product_id_column, config.conversion_label_column]
    files = _input_csv_files(config.input_path)
    frames: list[pd.DataFrame] = []
    for path in files:
        LOGGER.info("Reading interactions from %s", path)
        try:
            frames.extend(pd.read_csv(path, usecols=columns, chunksize=config.input_chunk_size, low_memory=False))
        except ValueError as error:
            raise ValueError(f"Interaction input {path} is missing a required column: {error}") from error
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def prepare_training_data(
    interactions: pd.DataFrame, config: TwoTowerRecallConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[set[int]], dict[str, int]]:
    """Encode interaction samples, assign click/conversion weights, and build histories."""

    required = {config.user_id_column, config.product_id_column, config.conversion_label_column}
    if missing := required - set(interactions.columns):
        raise ValueError(f"Interaction data is missing required columns: {sorted(missing)}")
    data = interactions.loc[:, [config.user_id_column, config.product_id_column, config.conversion_label_column]].copy()
    data.columns = ["user_id", "product_id", "conversion_label"]
    data["user_id"] = data["user_id"].astype("string").str.strip()
    data["product_id"] = data["product_id"].astype("string").str.strip()
    valid = data[["user_id", "product_id"]].notna().all(axis=1) & data[["user_id", "product_id"]].ne("").all(axis=1)
    data = data.loc[valid].copy()
    if config.max_train_rows is not None:
        data = data.iloc[: config.max_train_rows].copy()
    labels = pd.to_numeric(data["conversion_label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("conversion_label must be binary")
    data["weight"] = np.where(
        labels.to_numpy(dtype=np.int8) == 1, config.conversion_weight, config.click_weight
    )
    user_codes, users = pd.factorize(data["user_id"], sort=True)
    ad_codes, ads = pd.factorize(data["product_id"], sort=True)
    histories = [set() for _ in range(len(users))]
    for user, ad in zip(user_codes, ad_codes, strict=True):
        histories[int(user)].add(int(ad))
    trainable = np.fromiter((len(histories[user]) < len(ads) for user in user_codes), dtype=bool)
    trainable_positive_count = int(trainable.sum())
    stats = {
        "positive_samples": trainable_positive_count,
        "negative_samples": trainable_positive_count * config.negative_samples,
        "total_samples": trainable_positive_count * (1 + config.negative_samples),
        "click_positive_samples": int((labels.to_numpy(dtype=np.int8)[trainable] == 0).sum()),
        "conversion_positive_samples": int((labels.to_numpy(dtype=np.int8)[trainable] == 1).sum()),
        "excluded_no_negative_available": int((~trainable).sum()),
    }
    return user_codes, ad_codes, data["weight"].to_numpy(), users.astype(str).to_numpy(), ads.astype(str).to_numpy(), histories, stats


def train_two_tower(
    model: TwoTowerModel, dataset: NegativeSamplingDataset, config: TwoTowerRecallConfig, device: torch.device,
) -> list[float]:
    """Train with weighted sampled softmax (InfoNCE over one positive plus sampled negatives)."""

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(device)
    model.train()
    losses: list[float] = []
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        total_count = 0
        for users, positives, negatives, weights in loader:
            users, positives, negatives, weights = (value.to(device) for value in (users, positives, negatives, weights))
            user_vectors = model.encode_users(users)
            positive_vectors = model.encode_ads(positives)
            negative_vectors = model.encode_ads(negatives)
            positive_logits = (user_vectors * positive_vectors).sum(dim=1, keepdim=True)
            negative_logits = torch.einsum("bd,bnd->bn", user_vectors, negative_vectors)
            per_example = functional.cross_entropy(torch.cat([positive_logits, negative_logits], dim=1), torch.zeros(len(users), dtype=torch.long, device=device), reduction="none")
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(users)
            total_count += len(users)
        epoch_loss = total_loss / max(total_count, 1)
        losses.append(epoch_loss)
        LOGGER.info("Two Tower epoch %s/%s sampled-softmax loss=%.6f", epoch, config.epochs, epoch_loss)
    return losses


@torch.no_grad()
def extract_embeddings(model: TwoTowerModel, count: int, tower: str, batch_size: int, device: torch.device) -> np.ndarray:
    """Encode every user or advertisement in batches and return CPU float32 vectors."""

    if tower not in {"user", "ad"}:
        raise ValueError("tower must be 'user' or 'ad'")
    model.eval()
    values: list[np.ndarray] = []
    for start in range(0, count, batch_size):
        ids = torch.arange(start, min(start + batch_size, count), device=device)
        encoded = model.encode_users(ids) if tower == "user" else model.encode_ads(ids)
        values.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(values, axis=0) if values else np.empty((0, model.user_embedding.embedding_dim), dtype=np.float32)


def generate_two_tower_candidates(
    user_ids: np.ndarray, user_embeddings: np.ndarray, product_ids: np.ndarray, index: Any,
    histories: list[set[int]], top_k: int, query_batch_size: int = 4096,
) -> pd.DataFrame:
    """Retrieve nearest unseen ads for every user, ranked by cosine score."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    started_at = time.monotonic()
    rows: list[tuple[str, str, float, int]] = []
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be greater than zero")
    for start in range(0, len(user_ids), query_batch_size):
        stop = min(start + query_batch_size, len(user_ids))
        # A batch only needs enough extra results to cover its largest history:
        # this guarantees top_k unseen ads without materialising every ad for
        # every query user.
        search_k = min(index.ntotal, top_k + max((len(history) for history in histories[start:stop]), default=0))
        scores, positions = search_faiss_index(index, user_embeddings[start:stop], search_k)
        for relative_position, user_id in enumerate(user_ids[start:stop]):
            user_position = start + relative_position
            rank = 0
            for score, ad_position in zip(scores[relative_position], positions[relative_position], strict=True):
                if ad_position < 0 or int(ad_position) in histories[user_position]:
                    continue
                rank += 1
                rows.append((str(user_id), str(product_ids[int(ad_position)]), float(score), rank))
                if rank == top_k:
                    break
    elapsed_seconds = time.monotonic() - started_at
    LOGGER.info(
        "FAISS search benchmark: search users=%s total search time=%.2f seconds users/sec=%.2f",
        len(user_ids),
        elapsed_seconds,
        len(user_ids) / elapsed_seconds if elapsed_seconds else 0.0,
    )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).astype(
        {"user_id": "string", "candidate_ad_id": "string", "two_tower_score": "float64", "rank": "int64"}
    )


def save_checkpoint(model: TwoTowerModel, config: TwoTowerRecallConfig, user_ids: np.ndarray, product_ids: np.ndarray) -> None:
    """Save model parameters and ID vocabularies for reproducible offline inference."""

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "embedding_dim": config.embedding_dim, "user_ids": user_ids, "product_ids": product_ids}, config.checkpoint_path)
    LOGGER.info("Saved Two Tower checkpoint to %s", config.checkpoint_path)


def load_checkpoint(config: TwoTowerRecallConfig, user_ids: np.ndarray, product_ids: np.ndarray, device: torch.device) -> TwoTowerModel:
    """Load a trained tower and verify it matches the current data vocabularies."""

    checkpoint = _read_checkpoint(config.checkpoint_path, device)
    checkpoint_users = np.asarray(checkpoint.get("user_ids", []), dtype=str)
    checkpoint_products = np.asarray(checkpoint.get("product_ids", []), dtype=str)
    if not np.array_equal(checkpoint_users, user_ids) or not np.array_equal(checkpoint_products, product_ids):
        raise ValueError(
            "Checkpoint vocabularies do not match the current interactions; set train=true to retrain"
        )
    embedding_dim = int(checkpoint.get("embedding_dim", config.embedding_dim))
    if embedding_dim != config.embedding_dim:
        raise ValueError("Checkpoint embedding_dim does not match recall.two_tower.embedding_dim")
    model = TwoTowerModel(len(user_ids), len(product_ids), embedding_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    LOGGER.info("Loaded existing Two Tower checkpoint from %s; training skipped", config.checkpoint_path)
    return model


def load_checkpoint_for_inference(
    config: TwoTowerRecallConfig, device: torch.device,
) -> tuple[TwoTowerModel, np.ndarray, np.ndarray]:
    """Load a checkpoint and its ID vocabularies without reading training data."""

    checkpoint = _read_checkpoint(config.checkpoint_path, device)
    user_ids = np.asarray(checkpoint.get("user_ids", []), dtype=str)
    product_ids = np.asarray(checkpoint.get("product_ids", []), dtype=str)
    embedding_dim = int(checkpoint.get("embedding_dim", config.embedding_dim))
    if not len(user_ids) or not len(product_ids):
        raise ValueError("Two Tower checkpoint is missing user_ids or product_ids")
    if embedding_dim != config.embedding_dim:
        raise ValueError("Checkpoint embedding_dim does not match recall.two_tower.embedding_dim")
    model = TwoTowerModel(len(user_ids), len(product_ids), embedding_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    LOGGER.info("Loaded Two Tower checkpoint for inference from %s", config.checkpoint_path)
    return model, user_ids, product_ids


def load_or_build_faiss_index(
    model: TwoTowerModel, product_ids: np.ndarray, config: TwoTowerRecallConfig, device: torch.device,
) -> tuple[Any, np.ndarray]:
    """Load the persisted ANN index, rebuilding it only when it is absent."""

    if config.index_path.is_file():
        index, indexed_product_ids = load_faiss_index(config.index_path)
        if not np.array_equal(indexed_product_ids, product_ids):
            raise ValueError("FAISS index product IDs do not match the loaded Two Tower checkpoint")
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = config.ef_search
        LOGGER.info("Loaded existing FAISS index from %s; index construction skipped", config.index_path)
        return index, indexed_product_ids

    LOGGER.info("FAISS index does not exist at %s; building it from checkpoint ad embeddings", config.index_path)
    ad_embeddings = extract_embeddings(model, len(product_ids), "ad", config.inference_batch_size, device)
    index_started_at = time.monotonic()
    index = build_faiss_index(
        ad_embeddings,
        config.faiss_index_type,
        hnsw_m=config.hnsw_m,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
    )
    index_build_seconds = time.monotonic() - index_started_at
    save_faiss_index(index, product_ids, config.index_path)
    LOGGER.info(
        "FAISS index build benchmark: index_type=%s ads=%s index build time=%.2f seconds path=%s",
        config.faiss_index_type, index.ntotal, index_build_seconds, config.index_path,
    )
    return index, product_ids


@torch.no_grad()
def stream_two_tower_candidates(
    model: TwoTowerModel,
    user_ids: np.ndarray,
    product_ids: np.ndarray,
    index: Any,
    output_path: Path,
    *,
    top_k: int,
    max_users: Optional[int],
    search_batch_size: int,
    device: torch.device,
) -> int:
    """Retrieve and write one user batch at a time without global result state.

    The checkpoint provides the full user vocabulary.  Each batch produces its
    user embeddings, calls FAISS, and immediately writes CSV rows, so memory
    is bounded by ``search_batch_size * top_k`` rather than all users.
    """

    if top_k <= 0 or search_batch_size <= 0:
        raise ValueError("top_k and search_batch_size must be greater than zero")
    total_users = len(user_ids)
    selected_user_ids = user_ids if max_users is None else user_ids[:max_users]
    LOGGER.info("Selected users=%s/%s", len(selected_user_ids), total_users)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    started_at = time.monotonic()
    processed_users = 0
    written_rows = 0
    model.eval()
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)
        for start in range(0, len(selected_user_ids), search_batch_size):
            stop = min(start + search_batch_size, len(selected_user_ids))
            user_indices = torch.arange(start, stop, device=device)
            user_embeddings = model.encode_users(user_indices).cpu().numpy().astype(np.float32, copy=False)
            scores, positions = search_faiss_index(index, user_embeddings, top_k)
            for relative_position, user_id in enumerate(selected_user_ids[start:stop]):
                for rank, (score, ad_position) in enumerate(
                    zip(scores[relative_position], positions[relative_position], strict=True), start=1
                ):
                    if ad_position < 0:
                        continue
                    writer.writerow((str(user_id), str(product_ids[int(ad_position)]), float(score), rank))
                    written_rows += 1
            processed_users = stop
            elapsed_seconds = time.monotonic() - started_at
            LOGGER.info(
                "Memory-friendly FAISS retrieval benchmark: processed_users=%s elapsed_time=%.2f seconds "
                "users_per_second=%.2f",
                processed_users,
                elapsed_seconds,
                processed_users / elapsed_seconds if elapsed_seconds else 0.0,
            )
    temporary_path.replace(output_path)
    LOGGER.info("Wrote %s streamed Two Tower candidates to %s", written_rows, output_path)
    return written_rows


def run_two_tower_recall(config: TwoTowerRecallConfig) -> pd.DataFrame:
    """Train when requested, then run memory-bounded offline FAISS recall."""

    _set_seed(config.seed)
    device = _select_device(config.device)
    LOGGER.info("Two Tower inference device: %s", device)
    if not config.train and config.checkpoint_path.is_file():
        # The inference path intentionally avoids loading the interaction data.
        # User and product vocabularies are checkpointed with the model.
        model, user_ids, product_ids = load_checkpoint_for_inference(config, device)
    else:
        if not config.train:
            LOGGER.info("No existing checkpoint at %s; training Two Tower model", config.checkpoint_path)
        else:
            LOGGER.info("Two Tower training forced by recall.two_tower.train=true")
        interactions = load_interactions(config)
        user_codes, ad_codes, weights, user_ids, product_ids, histories, stats = prepare_training_data(
            interactions, config
        )
        LOGGER.info("Training sample statistics: %s", stats)
        if not len(user_ids) or not len(product_ids):
            raise ValueError("No valid interactions available for Two Tower training")
        dataset = NegativeSamplingDataset(
            user_codes, ad_codes, weights, histories, len(product_ids), config.negative_samples, config.seed
        )
        if not len(dataset):
            raise ValueError("Cannot sample negatives: every user has interacted with every advertisement")
        model = TwoTowerModel(len(user_ids), len(product_ids), config.embedding_dim)
        train_two_tower(model, dataset, config, device)
        save_checkpoint(model, config, user_ids, product_ids)
    index, indexed_product_ids = load_or_build_faiss_index(model, product_ids, config, device)
    stream_two_tower_candidates(
        model, user_ids, indexed_product_ids, index, config.output_path,
        top_k=config.top_k, max_users=config.max_users,
        search_batch_size=config.search_batch_size, device=device,
    )
    # Streaming intentionally leaves no full candidate dataframe resident.
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype(
        {"user_id": "string", "candidate_ad_id": "string", "two_tower_score": "float64", "rank": "int64"}
    )


def write_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    if tuple(candidates.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Candidate output columns must be {OUTPUT_COLUMNS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    candidates.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    LOGGER.info("Wrote %s Two Tower candidates to %s", len(candidates), output_path)


def _input_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Two Tower interaction input does not exist: {path}")
    files = sorted(path.glob("part-*.csv")) or sorted(item for item in path.glob("*.csv") if not item.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No CSV interaction files found in: {path}")
    return files


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _validate_config(config: TwoTowerRecallConfig) -> None:
    if min(config.embedding_dim, config.batch_size, config.epochs, config.top_k, config.negative_samples, config.search_batch_size, config.inference_batch_size, config.input_chunk_size) <= 0:
        raise ValueError("Two Tower numeric configuration values must be greater than zero")
    if config.learning_rate <= 0:
        raise ValueError("recall.two_tower.learning_rate must be greater than zero")
    if config.click_weight <= 0 or config.conversion_weight <= 0:
        raise ValueError("recall.two_tower interaction weights must be greater than zero")
    if config.faiss_index_type.lower() not in {"flat", "hnsw"}:
        raise ValueError("recall.two_tower.faiss.index_type must be 'flat' or 'hnsw'")
    if min(config.hnsw_m, config.ef_construction, config.ef_search) <= 0:
        raise ValueError("recall.two_tower.faiss HNSW values must be greater than zero")
    if config.max_users is not None and config.max_users <= 0:
        raise ValueError("recall.two_tower.max_users must be greater than zero when set")
    if config.max_train_rows is not None and config.max_train_rows <= 0:
        raise ValueError("recall.two_tower.max_train_rows must be greater than zero when set")


def _read_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Unable to load Two Tower checkpoint: {path}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Two Tower checkpoint must be a mapping")
    return checkpoint


def checkpoint_parameter_counts(path: Path) -> dict[str, int]:
    """Read ID-only checkpoint capacity without constructing an inference run."""
    checkpoint = _read_checkpoint(path, torch.device("cpu"))
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, Mapping):
        raise ValueError("Two Tower checkpoint is missing state_dict")
    total = sum(int(value.numel()) for value in state.values() if isinstance(value, Tensor))
    embedding = sum(int(value.numel()) for name, value in state.items() if "embedding" in name and isinstance(value, Tensor))
    return {"total": total, "embedding": embedding, "dense": total - embedding}


def _faiss_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read FAISS options, accepting legacy index_type for a smooth upgrade."""

    faiss_options = options.get("faiss", {})
    if not isinstance(faiss_options, Mapping):
        raise ValueError("recall.two_tower.faiss configuration must be a mapping")
    return {"index_type": options.get("index_type", "hnsw"), **faiss_options}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate product-ad candidates with Two Tower + FAISS.")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    run_two_tower_recall(parse_two_tower_config(load_yaml_config(config_path), config_path))


if __name__ == "__main__":
    main()
