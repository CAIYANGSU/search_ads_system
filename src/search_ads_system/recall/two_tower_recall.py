"""Offline Two Tower training and FAISS candidate retrieval for product ads."""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.recall.faiss_index import (
    build_faiss_index,
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
    reuse_checkpoint: bool = True
    inference_batch_size: int = 4096
    input_chunk_size: int = 200_000


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
        reuse_checkpoint=bool(options.get("reuse_checkpoint", True)),
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

    try:
        checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Unable to load Two Tower checkpoint: {config.checkpoint_path}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Two Tower checkpoint must be a mapping")
    checkpoint_users = np.asarray(checkpoint.get("user_ids", []), dtype=str)
    checkpoint_products = np.asarray(checkpoint.get("product_ids", []), dtype=str)
    if not np.array_equal(checkpoint_users, user_ids) or not np.array_equal(checkpoint_products, product_ids):
        raise ValueError(
            "Checkpoint vocabularies do not match the current interactions; set reuse_checkpoint=false to retrain"
        )
    embedding_dim = int(checkpoint.get("embedding_dim", config.embedding_dim))
    if embedding_dim != config.embedding_dim:
        raise ValueError("Checkpoint embedding_dim does not match recall.two_tower.embedding_dim")
    model = TwoTowerModel(len(user_ids), len(product_ids), embedding_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    LOGGER.info("Loaded existing Two Tower checkpoint from %s; training skipped", config.checkpoint_path)
    return model


def run_two_tower_recall(config: TwoTowerRecallConfig) -> pd.DataFrame:
    """Execute data preparation, training, index creation, and offline recall."""

    _set_seed(config.seed)
    device = _select_device(config.device)
    LOGGER.info("Two Tower training device: %s", device)
    interactions = load_interactions(config)
    user_codes, ad_codes, weights, user_ids, product_ids, histories, stats = prepare_training_data(interactions, config)
    LOGGER.info("Training sample statistics: %s", stats)
    if not len(user_ids) or not len(product_ids):
        raise ValueError("No valid interactions available for Two Tower training")
    if config.reuse_checkpoint and config.checkpoint_path.is_file():
        model = load_checkpoint(config, user_ids, product_ids, device)
    else:
        dataset = NegativeSamplingDataset(
            user_codes, ad_codes, weights, histories, len(product_ids), config.negative_samples, config.seed
        )
        if not len(dataset):
            raise ValueError("Cannot sample negatives: every user has interacted with every advertisement")
        model = TwoTowerModel(len(user_ids), len(product_ids), config.embedding_dim)
        train_two_tower(model, dataset, config, device)
        save_checkpoint(model, config, user_ids, product_ids)
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
        config.faiss_index_type,
        index.ntotal,
        index_build_seconds,
        config.index_path,
    )
    user_embeddings = extract_embeddings(model, len(user_ids), "user", config.inference_batch_size, device)
    candidates = generate_two_tower_candidates(
        user_ids, user_embeddings, product_ids, index, histories, config.top_k, config.inference_batch_size
    )
    write_candidates(candidates, config.output_path)
    return candidates


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
    if min(config.embedding_dim, config.batch_size, config.epochs, config.top_k, config.negative_samples, config.inference_batch_size, config.input_chunk_size) <= 0:
        raise ValueError("Two Tower numeric configuration values must be greater than zero")
    if config.learning_rate <= 0:
        raise ValueError("recall.two_tower.learning_rate must be greater than zero")
    if config.click_weight <= 0 or config.conversion_weight <= 0:
        raise ValueError("recall.two_tower interaction weights must be greater than zero")
    if config.faiss_index_type.lower() not in {"flat", "hnsw"}:
        raise ValueError("recall.two_tower.faiss.index_type must be 'flat' or 'hnsw'")
    if min(config.hnsw_m, config.ef_construction, config.ef_search) <= 0:
        raise ValueError("recall.two_tower.faiss HNSW values must be greater than zero")


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
