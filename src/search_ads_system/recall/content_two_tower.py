"""Past-only, content-aware Two-Tower retrieval for temporal cold start.

An optional product catalogue is a point-in-time input, not a Future-A/B
label source.  Without one, the catalogue is built from Past and the run
explicitly reports that it cannot index products absent from Past.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from search_ads_system.recall.faiss_index import build_faiss_index, load_faiss_index, save_faiss_index, search_faiss_index
from search_ads_system.recall.two_tower_recall import OUTPUT_COLUMNS, _input_csv_files, _select_device

CONTENT_SCHEMA_VERSION = "content_two_tower_v1"
CONTENT_COLUMNS = ("product_brand", "product_category_1", "product_category_2", "product_category_3", "product_gender", "product_age_group", "product_country")
MISSING_TOKEN = "__UNKNOWN__"


@dataclass(frozen=True)
class ContentTwoTowerConfig:
    input_path: Path
    output_path: Path
    index_path: Path
    checkpoint_path: Path
    product_catalog_path: Path | None = None
    catalog_as_of_timestamp: int | None = None
    variant: str = "content"  # content | content_no_product_id
    embedding_dim: int = 16
    hidden_dim: int = 64
    categorical_buckets: int = 4096
    product_id_buckets: int = 65537
    batch_size: int = 4096
    epochs: int = 3
    learning_rate: float = 1e-3
    top_k: int = 100
    negative_samples: int = 5
    click_weight: float = 1.0
    conversion_weight: float = 3.0
    max_history_items: int = 100
    max_train_rows: int | None = None
    seed: int = 2026
    device: str = "auto"
    faiss_index_type: str = "hnsw"
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    train: bool = True
    search_batch_size: int = 10_000
    inference_batch_size: int = 4096
    input_chunk_size: int = 200_000

    @property
    def use_product_id(self) -> bool:
        return self.variant == "content"


@dataclass
class ContentTrainingData:
    user_indices: np.ndarray
    product_indices: np.ndarray
    weights: np.ndarray
    user_ids: np.ndarray
    product_ids: np.ndarray
    training_product_indices: np.ndarray
    histories: list[set[int]]
    product_category_indices: np.ndarray
    product_id_indices: np.ndarray
    normalized_prices: np.ndarray
    history_indices: np.ndarray
    history_mask: np.ndarray
    user_dense_stats: np.ndarray
    metadata: dict[str, Any]


class ContentNegativeSamplingDataset(Dataset[tuple[int, int, np.ndarray, float]]):
    """Same sampled-softmax semantics as the ID-only baseline; negatives are Past pool samples."""
    def __init__(self, data: ContentTrainingData, config: ContentTwoTowerConfig) -> None:
        self.users, self.products, self.weights = data.user_indices, data.product_indices, data.weights
        self.histories, self.pool, self.count, self.seed = data.histories, data.training_product_indices, config.negative_samples, config.seed
        self.valid = np.asarray([index for index, user in enumerate(self.users) if len(self.histories[int(user)] & set(self.pool.tolist())) < len(self.pool)], dtype=np.int64)

    def __len__(self) -> int: return len(self.valid)
    def __getitem__(self, position: int) -> tuple[int, int, np.ndarray, float]:
        row = int(self.valid[position]); history = self.histories[int(self.users[row])]; rng = np.random.default_rng(self.seed + row)
        candidates = self.pool[np.fromiter((item not in history for item in self.pool), dtype=bool, count=len(self.pool))]
        negatives = rng.choice(candidates, self.count, replace=len(candidates) < self.count).astype(np.int64)
        return int(self.users[row]), int(self.products[row]), negatives, float(self.weights[row])


class ContentTwoTowerModel(nn.Module):
    """Product content tower plus Past-only pooled product history in user tower."""
    def __init__(self, data: ContentTrainingData, config: ContentTwoTowerConfig) -> None:
        super().__init__()
        self.embedding_dim, self.variant = config.embedding_dim, config.variant
        self.user_embedding = nn.Embedding(len(data.user_ids), config.embedding_dim)
        self.product_id_embedding = nn.Embedding(config.product_id_buckets, config.embedding_dim) if config.use_product_id else None
        self.content_embeddings = nn.ModuleList([nn.Embedding(config.categorical_buckets, config.embedding_dim) for _ in CONTENT_COLUMNS])
        product_inputs = len(CONTENT_COLUMNS) * config.embedding_dim + 1 + (config.embedding_dim if config.use_product_id else 0)
        self.product_tower = _mlp(product_inputs, config.hidden_dim, config.embedding_dim)
        self.user_tower = _mlp(config.embedding_dim * 2 + 2, config.hidden_dim, config.embedding_dim)
        self.register_buffer("product_category_indices", torch.as_tensor(data.product_category_indices, dtype=torch.long))
        self.register_buffer("product_id_indices", torch.as_tensor(data.product_id_indices, dtype=torch.long))
        self.register_buffer("normalized_prices", torch.as_tensor(data.normalized_prices, dtype=torch.float32))
        self.register_buffer("history_indices", torch.as_tensor(data.history_indices, dtype=torch.long))
        self.register_buffer("history_mask", torch.as_tensor(data.history_mask, dtype=torch.bool))
        self.register_buffer("user_dense_stats", torch.as_tensor(data.user_dense_stats, dtype=torch.float32))

    def encode_products(self, positions: Tensor) -> Tensor:
        categories = self.product_category_indices[positions]
        parts = [embedding(categories[..., index]) for index, embedding in enumerate(self.content_embeddings)]
        if self.product_id_embedding is not None:
            parts.insert(0, self.product_id_embedding(self.product_id_indices[positions]))
        parts.append(self.normalized_prices[positions].unsqueeze(-1))
        return F.normalize(self.product_tower(torch.cat(parts, dim=-1)), dim=-1)

    def encode_users(self, user_indices: Tensor) -> Tensor:
        history = self.history_indices[user_indices]
        mask = self.history_mask[user_indices]
        # Padding index is masked after encoding, so a real product at position
        # zero is never confused with missing history.
        history_vectors = self.encode_products(history)
        pooled = (history_vectors * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        features = torch.cat((self.user_embedding(user_indices), pooled, self.user_dense_stats[user_indices]), dim=-1)
        return F.normalize(self.user_tower(features), dim=-1)

    def forward(self, users: Tensor, products: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode_users(users), self.encode_products(products)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


def prepare_content_training_data(config: ContentTwoTowerConfig) -> ContentTrainingData:
    """Construct all training inputs from Past and an optional safe catalogue."""
    if config.product_catalog_path is not None and any(part in {"future_a", "future_b", "future"} for part in config.product_catalog_path.parts):
        raise ValueError("Content catalogue must be a point-in-time source, never a Future/Future-A/Future-B split")
    if config.product_catalog_path is not None and config.catalog_as_of_timestamp is None:
        raise ValueError("External content catalogue requires declared catalog_as_of_timestamp at or before the Past cutoff")
    past = _read_interactions(config.input_path, require_label=True, chunk_size=config.input_chunk_size)
    if config.max_train_rows is not None:
        past = past.iloc[:config.max_train_rows].copy()
    if past.empty: raise ValueError("Content Two Tower requires at least one valid Past interaction")
    catalogue_source = config.product_catalog_path or config.input_path
    catalogue = _read_interactions(catalogue_source, require_label=False, require_user=False, chunk_size=config.input_chunk_size)
    # Past is last so its observed metadata wins for products it knows; external
    # catalogue rows remain available for content-representable cold products.
    catalogue = pd.concat((catalogue, past.drop(columns=["conversion_label"])), ignore_index=True).drop_duplicates("product_id", keep="last")
    catalogue = catalogue.loc[catalogue.product_id.notna() & catalogue.product_id.ne("")].reset_index(drop=True)
    product_ids = catalogue.product_id.astype(str).to_numpy(); product_to_index = {value: index for index, value in enumerate(product_ids)}
    past = past.loc[past.product_id.isin(product_to_index)].copy()
    user_codes, user_values = pd.factorize(past.user_id, sort=True)
    product_indices = past.product_id.map(product_to_index).to_numpy(dtype=np.int64)
    labels = pd.to_numeric(past.conversion_label, errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(labels, (0, 1)).all(): raise ValueError("conversion_label must be binary")
    weights = np.where(labels == 1, config.conversion_weight, config.click_weight).astype(np.float32)
    histories = [set() for _ in range(len(user_values))]
    for user, product in zip(user_codes, product_indices, strict=True): histories[int(user)].add(int(product))
    pool = np.unique(product_indices)
    category_indices = np.column_stack([_hash_series(catalogue[column], config.categorical_buckets) for column in CONTENT_COLUMNS]).astype(np.int64)
    product_id_indices = _hash_series(catalogue.product_id, config.product_id_buckets).astype(np.int64)
    log_price = np.log1p(pd.to_numeric(catalogue.get("product_price"), errors="coerce").clip(lower=0).fillna(0).to_numpy(dtype=np.float32))
    train_prices = log_price[pool]; mean, std = float(train_prices.mean()) if len(train_prices) else 0.0, float(train_prices.std()) if len(train_prices) else 1.0
    normalized_prices = ((log_price - mean) / max(std, 1e-6)).astype(np.float32)
    history_indices, history_mask = _history_matrix(histories, config.max_history_items)
    counts = np.asarray([len(history) for history in histories], dtype=np.float32)
    average_price = np.asarray([normalized_prices[list(history)].mean() if history else 0.0 for history in histories], dtype=np.float32)
    user_dense = np.column_stack((np.log1p(counts), average_price)).astype(np.float32)
    trainable = np.fromiter((len(history & set(pool.tolist())) < len(pool) for history in histories), dtype=bool)
    sample_valid = trainable[user_codes]
    metadata = {"schema_version": CONTENT_SCHEMA_VERSION, "variant": config.variant, "catalogue_source": str(catalogue_source), "catalogue_from_past_only": config.product_catalog_path is None, "catalog_as_of_timestamp": config.catalog_as_of_timestamp, "catalogue_products": len(product_ids), "past_training_products": len(pool), "content_representable_cold_products": int(len(product_ids) - len(pool)), "price_normalization": {"mean_log1p_price": mean, "std_log1p_price": std, "fit_from": "Past training products only"}, "user_history": "Past-only clicked-product content mean pooling plus log click count and mean normalized log price", "training_contract": "Past observed interactions as positives; sampled negatives are Past-pool negatives, not exposure negatives."}
    return ContentTrainingData(user_codes[sample_valid].astype(np.int64), product_indices[sample_valid], weights[sample_valid], user_values.astype(str).to_numpy(), product_ids, pool, histories, category_indices, product_id_indices, normalized_prices, history_indices, history_mask, user_dense, metadata)


def train_content_two_tower(model: ContentTwoTowerModel, dataset: ContentNegativeSamplingDataset, config: ContentTwoTowerConfig, device: torch.device) -> list[float]:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate); model.to(device); model.train(); losses=[]
    for epoch in range(config.epochs):
        total = count = 0
        for users, positives, negatives, weights in loader:
            users, positives, negatives, weights = (value.to(device) for value in (users, positives, negatives, weights))
            user_vectors = model.encode_users(users); positive_vectors = model.encode_products(positives); negative_vectors = model.encode_products(negatives)
            logits = torch.cat(((user_vectors * positive_vectors).sum(-1, keepdim=True), torch.einsum("bd,bnd->bn", user_vectors, negative_vectors)), 1)
            loss = (F.cross_entropy(logits, torch.zeros(len(users), dtype=torch.long, device=device), reduction="none") * weights).sum() / weights.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(users); count += len(users)
        losses.append(total / max(count, 1))
    return losses


@torch.no_grad()
def extract_content_product_embeddings(model: ContentTwoTowerModel, count: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); values=[]
    for start in range(0, count, batch_size): values.append(model.encode_products(torch.arange(start, min(start + batch_size, count), device=device)).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(values) if values else np.empty((0, model.embedding_dim), dtype=np.float32)


def save_content_checkpoint(model: ContentTwoTowerModel, data: ContentTrainingData, config: ContentTwoTowerConfig, losses: list[float]) -> None:
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": CONTENT_SCHEMA_VERSION, "variant": config.variant, "embedding_dim": config.embedding_dim, "hidden_dim": config.hidden_dim, "categorical_buckets": config.categorical_buckets, "product_id_buckets": config.product_id_buckets, "state_dict": model.state_dict(), "user_ids": data.user_ids, "product_ids": data.product_ids, "training_product_ids": data.product_ids[data.training_product_indices], "metadata": data.metadata | {"losses": losses, "parameter_counts": parameter_counts(model)}}, config.checkpoint_path)


def load_content_checkpoint(config: ContentTwoTowerConfig, device: torch.device) -> tuple[ContentTwoTowerModel, ContentTrainingData, Mapping[str, Any]]:
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != CONTENT_SCHEMA_VERSION or checkpoint.get("variant") != config.variant:
        raise ValueError("Content checkpoint schema/variant mismatch; it cannot be loaded as ID-only or another content ablation")
    # Recreate buffers from Past/catalogue and require an exact vocabulary match.
    data = prepare_content_training_data(config)
    if not np.array_equal(np.asarray(checkpoint["user_ids"], dtype=str), data.user_ids) or not np.array_equal(np.asarray(checkpoint["product_ids"], dtype=str), data.product_ids):
        raise ValueError("Content checkpoint vocabularies do not match Past/catalogue inputs; retrain is required")
    model = ContentTwoTowerModel(data, config); model.load_state_dict(checkpoint["state_dict"]); model.to(device); model.eval()
    return model, data, checkpoint


def run_content_two_tower_recall(config: ContentTwoTowerConfig) -> dict[str, Any]:
    _set_seed(config.seed); device = _select_device(config.device)
    if config.train or not config.checkpoint_path.is_file():
        data = prepare_content_training_data(config); dataset = ContentNegativeSamplingDataset(data, config)
        if not len(dataset): raise ValueError("Cannot draw Past-pool negatives for content Two Tower")
        model = ContentTwoTowerModel(data, config); losses = train_content_two_tower(model, dataset, config, device); save_content_checkpoint(model, data, config, losses)
    else:
        model, data, checkpoint = load_content_checkpoint(config, device); losses = list(checkpoint.get("metadata", {}).get("losses", []))
    embeddings = extract_content_product_embeddings(model, len(data.product_ids), config.inference_batch_size, device)
    if config.index_path.is_file():
        index, indexed_ids = load_faiss_index(config.index_path)
        if not np.array_equal(indexed_ids, data.product_ids): raise ValueError("Content FAISS index product IDs do not match checkpoint catalogue")
    else:
        index = build_faiss_index(embeddings, config.faiss_index_type, hnsw_m=config.hnsw_m, ef_construction=config.ef_construction, ef_search=config.ef_search); save_faiss_index(index, data.product_ids, config.index_path)
    rows = _stream_content_candidates(model, data, index, config, device)
    return {"rows": rows, "parameter_counts": parameter_counts(model), "metadata": data.metadata, "losses": losses, "index_products": int(index.ntotal)}


@torch.no_grad()
def _stream_content_candidates(model: ContentTwoTowerModel, data: ContentTrainingData, index: Any, config: ContentTwoTowerConfig, device: torch.device) -> int:
    config.output_path.parent.mkdir(parents=True, exist_ok=True); temp=config.output_path.with_suffix(config.output_path.suffix + ".tmp"); written=0; model.eval()
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(OUTPUT_COLUMNS)
        for start in range(0, len(data.user_ids), config.search_batch_size):
            stop=min(start+config.search_batch_size,len(data.user_ids)); users=torch.arange(start,stop,device=device); vectors=model.encode_users(users).cpu().numpy().astype(np.float32,copy=False)
            maximum_history=max((len(data.histories[index]) for index in range(start,stop)),default=0); scores, positions=search_faiss_index(index,vectors,min(index.ntotal,config.top_k+maximum_history))
            for relative,user_id in enumerate(data.user_ids[start:stop]):
                rank=0; history=data.histories[start+relative]
                for score,position in zip(scores[relative],positions[relative],strict=True):
                    if position < 0 or int(position) in history: continue
                    rank += 1; writer.writerow((str(user_id),str(data.product_ids[int(position)]),float(score),rank)); written += 1
                    if rank == config.top_k: break
    temp.replace(config.output_path); return written


def parameter_counts(model: nn.Module) -> dict[str, int]:
    embedding = sum(parameter.numel() for name, parameter in model.named_parameters() if "embedding" in name)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total": total, "embedding": embedding, "dense": total - embedding}


def content_checkpoint_parameter_counts(path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, Mapping):
        raise ValueError("Content Two Tower checkpoint is missing state_dict")
    total = sum(int(value.numel()) for value in state.values() if isinstance(value, Tensor))
    embedding = sum(int(value.numel()) for name, value in state.items() if "embedding" in name and isinstance(value, Tensor))
    return {"total": total, "embedding": embedding, "dense": total - embedding}


def _read_interactions(path: Path, *, require_label: bool, chunk_size: int, require_user: bool = True) -> pd.DataFrame:
    required = (["user_id"] if require_user else []) + ["product_id"] + (["conversion_label"] if require_label else [])
    frames=[]
    for file in _input_csv_files(path):
        header=set(pd.read_csv(file,nrows=0).columns); missing=set(required)-header
        if missing: raise ValueError(f"{file} is missing required columns: {sorted(missing)}")
        columns=list(dict.fromkeys(required + [column for column in ("product_price",) + CONTENT_COLUMNS if column in header]))
        for chunk in pd.read_csv(file,usecols=columns,chunksize=chunk_size,low_memory=False): frames.append(chunk)
    result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=required)
    for column in CONTENT_COLUMNS:
        values = result[column] if column in result else pd.Series(MISSING_TOKEN, index=result.index)
        result[column]=values.astype("string").fillna(MISSING_TOKEN).replace("",MISSING_TOKEN)
    result["product_price"]=pd.to_numeric(result["product_price"] if "product_price" in result else pd.Series(0.0,index=result.index),errors="coerce")
    if "user_id" not in result: result["user_id"] = MISSING_TOKEN
    result["user_id"]=result.user_id.astype("string").str.strip(); result["product_id"]=result.product_id.astype("string").str.strip()
    valid=result.product_id.notna() & result.product_id.ne("")
    if require_user: valid &= result.user_id.notna() & result.user_id.ne("")
    return result.loc[valid].copy()


def _hash_series(values: pd.Series, buckets: int) -> np.ndarray:
    if buckets < 2: raise ValueError("hash bucket count must be at least 2")
    return np.asarray([0 if pd.isna(value) or str(value).strip() in {"", MISSING_TOKEN} else 1 + _stable_hash(str(value)) % (buckets - 1) for value in values],dtype=np.int64)


def _stable_hash(value: str) -> int: return int.from_bytes(hashlib.blake2b(value.encode(),digest_size=8).digest(),"big")


def _history_matrix(histories: list[set[int]], maximum: int) -> tuple[np.ndarray,np.ndarray]:
    width=max(1,maximum); values=np.zeros((len(histories),width),dtype=np.int64); mask=np.zeros_like(values,dtype=bool)
    for user,history in enumerate(histories):
        selected=sorted(history)[:width]; values[user,:len(selected)]=selected; mask[user,:len(selected)]=True
    return values,mask


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
