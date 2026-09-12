"""Small, auditable strict-temporal Two-Tower recall baseline.

This module intentionally does not import the historical content Two-Tower
implementations.  It reads only a Past directory and a Future-A directory;
Future-A is used exclusively for the fixed evaluation cohort and labels.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

PAD_INDEX, OOV_INDEX = 0, 1
ITEM_FIELDS = ("product_brand", "product_category_1", "product_category_2", "product_category_3", "partner_id")
USER_FIELDS = ("device_type", "audience_id", "hour", "day_of_week")
CUT_OFFS = (50, 100, 200)


@dataclass(frozen=True)
class SimpleTwoTowerV2Config:
    past_path: Path
    future_a_path: Path
    output_dir: Path
    max_users: int = 100_000
    seed: int = 42
    embedding_dim: int = 64
    feature_embedding_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 128)
    batch_size: int = 16_384
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    negative_samples: int = 5
    click_weight: float = 1.0
    conversion_weight: float = 3.0
    temperature: float = 0.07
    top_k: int = 200
    exclude_seen_items: bool = True
    retrieval_oversample_ratio: float = 2.0
    retrieval_batch_size: int = 20_000
    inference_batch_size: int = 32_768
    use_bf16: bool = True
    use_torch_compile: bool = False
    use_fused_adamw: bool = True
    device: str = "auto"
    max_train_rows: int | None = None
    variants: tuple[str, ...] = ("v0_id_only", "v1_item_content", "v2_user_context_stats", "v3_in_batch_negatives")


@dataclass
class PreparedData:
    users: np.ndarray
    products: np.ndarray
    train_user: np.ndarray
    train_item: np.ndarray
    train_context: np.ndarray
    train_weight: np.ndarray
    eval_context: np.ndarray
    user_stats: np.ndarray
    item_features: np.ndarray
    item_price: np.ndarray
    histories: list[set[int]]
    truth: dict[str, set[str]]
    warm_truth: dict[str, set[str]]
    vocabs: dict[str, dict[str, int]]
    metadata: dict[str, Any]


def _parts(path: Path) -> list[Path]:
    files = sorted(path.glob("part-*.csv")) if path.is_dir() else [path]
    if not files: raise FileNotFoundError(f"No part-*.csv files at {path}")
    if any("future_b" in part.lower() for part in path.parts): raise ValueError("Future-B is prohibited for simple_two_tower_v2")
    return files


def _clean(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _id(value: object) -> str | None:
    if value is None or pd.isna(value): return None
    result = str(value).strip()
    return result or None


def _read_selected(path: Path, users: set[str] | None, columns: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for part in _parts(path):
        for chunk in pd.read_csv(part, usecols=columns, chunksize=200_000, low_memory=False):
            chunk["user_id"] = _clean(chunk["user_id"])
            chunk["product_id"] = _clean(chunk["product_id"])
            chunk = chunk[(chunk.user_id != "") & (chunk.product_id != "")]
            if users is not None: chunk = chunk[chunk.user_id.isin(users)]
            if len(chunk): frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or [])


def _eligible_users(past_path: Path, future_path: Path, max_users: int, seed: int) -> np.ndarray:
    def collect(path: Path) -> set[str]:
        values: set[str] = set()
        for part in _parts(path):
            for chunk in pd.read_csv(part, usecols=["user_id", "product_id"], chunksize=200_000, low_memory=False):
                valid = (_clean(chunk.user_id) != "") & (_clean(chunk.product_id) != "")
                values.update(_clean(chunk.loc[valid, "user_id"]).tolist())
        return values
    overlap = sorted(collect(past_path) & collect(future_path))
    if not overlap: raise RuntimeError("No users occur in both Past and Future-A")
    rng = np.random.default_rng(seed)
    chosen = np.asarray(overlap if len(overlap) <= max_users else rng.choice(overlap, max_users, replace=False), dtype=str)
    return np.sort(chosen)


def _vocab(values: Iterable[object]) -> dict[str, int]:
    # Missing is intentionally OOV-safe (index 1); index 0 is reserved padding.
    known = sorted({value for raw in values if (value := _id(raw)) is not None})
    return {value: index + 2 for index, value in enumerate(known)}


def _encode(values: Iterable[object], vocab: Mapping[str, int]) -> np.ndarray:
    return np.asarray([vocab.get(_id(value) or "", OOV_INDEX) for value in values], dtype=np.int64)


def prepare_data(config: SimpleTwoTowerV2Config) -> PreparedData:
    """Encode compact in-memory arrays from Past only; Future-A supplies truth only."""
    chosen = _eligible_users(config.past_path, config.future_a_path, config.max_users, config.seed)
    cohort = set(chosen.tolist())
    required = ["user_id", "product_id", "click_timestamp", "conversion_label", "product_price", *ITEM_FIELDS, "device_type", "audience_id"]
    past = _read_selected(config.past_path, cohort, required)
    # Catalogue must include every product observed in Past, not just the cohort.
    catalogue = _read_selected(config.past_path, None, ["user_id", "product_id", "product_price", *ITEM_FIELDS])
    future = _read_selected(config.future_a_path, cohort, ["user_id", "product_id", "click_timestamp"])
    if past.empty or catalogue.empty or future.empty: raise RuntimeError("Past, catalogue, and Future-A must all contain valid interactions")
    if config.max_train_rows is not None: past = past.iloc[:config.max_train_rows].copy()
    past["click_timestamp"] = pd.to_numeric(past.click_timestamp, errors="coerce").fillna(0).astype("int64")
    future["click_timestamp"] = pd.to_numeric(future.click_timestamp, errors="coerce").fillna(0).astype("int64")
    if int(past.click_timestamp.max()) >= int(future.click_timestamp.min()):
        raise RuntimeError("Temporal contract failed: max(Past timestamp) must be < min(Future-A timestamp)")
    users = chosen
    user_to_i = {value: index for index, value in enumerate(users)}
    catalogue = catalogue.drop_duplicates("product_id", keep="last").reset_index(drop=True)
    products = catalogue.product_id.astype(str).to_numpy()
    product_to_i = {value: index for index, value in enumerate(products)}
    past = past[past.product_id.isin(product_to_i)].copy()
    if past.empty: raise RuntimeError("No Past training rows map to the Past product catalogue")
    truth = {user: set(group.product_id.astype(str)) for user, group in future.groupby("user_id", sort=False)}
    warm_truth = {user: values & set(products) for user, values in truth.items()}
    total_truth, warm_total = sum(map(len, truth.values())), sum(map(len, warm_truth.values()))
    overlap = set(products) & set().union(*truth.values())
    if not overlap or not warm_total:
        examples = {"catalogue": products[:3].tolist(), "future": sorted(set().union(*truth.values()))[:3]}
        raise RuntimeError(f"Past catalogue and Future-A raw product IDs have zero overlap: {examples}")
    user_vocab = _vocab(users)
    vocabs = {"user_id": user_vocab, "product_id": _vocab(products)}
    for field in ITEM_FIELDS + ("device_type", "audience_id"):
        source = catalogue[field] if field in ITEM_FIELDS else past[field]
        vocabs[field] = _vocab(source)
    for field in ("hour", "day_of_week"):
        vocabs[field] = _vocab(map(str, range(24 if field == "hour" else 7)))
    item_features = np.column_stack([_encode(catalogue[field], vocabs[field]) for field in ITEM_FIELDS]).astype(np.int64)
    raw_price = pd.to_numeric(catalogue.product_price, errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
    log_price = np.log1p(raw_price)
    mean, std = float(log_price.mean()), max(float(log_price.std()), 1e-6)
    item_price = ((log_price - mean) / std).astype(np.float32)
    if not np.isfinite(item_price).all(): raise ValueError("Past-fitted normalized item prices are not finite")
    past["hour"] = (past.click_timestamp // 3600 % 24).astype(str); past["day_of_week"] = ((past.click_timestamp // 86400 + 4) % 7).astype(str)
    train_context = np.column_stack([_encode(past[field], vocabs[field]) for field in ("device_type", "audience_id", "hour", "day_of_week")])
    train_user = past.user_id.map(user_to_i).to_numpy(dtype=np.int64); train_item = past.product_id.map(product_to_i).to_numpy(dtype=np.int64)
    labels = pd.to_numeric(past.conversion_label, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    train_weight = np.where(labels == 1, config.conversion_weight, config.click_weight).astype(np.float32)
    histories = [set() for _ in users]
    for user, item in zip(train_user, train_item, strict=True): histories[int(user)].add(int(item))
    # A user who has seen the complete warm catalogue has no valid sampled
    # negative.  Exclude only those rows instead of fabricating a false one.
    valid_negative = np.asarray([len(histories[int(user)]) < len(products) for user in train_user], dtype=bool)
    train_user, train_item = train_user[valid_negative], train_item[valid_negative]
    train_context, train_weight = train_context[valid_negative], train_weight[valid_negative]
    if not len(train_user): raise RuntimeError("No training rows have an available Past-catalogue negative")
    grouped = past.sort_values("click_timestamp").groupby("user_id", sort=False)
    count = grouped.size().reindex(users, fill_value=0).to_numpy(dtype=np.float32)
    conversions = grouped.conversion_label.apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).reindex(users, fill_value=0).to_numpy(dtype=np.float32)
    global_cvr = float(labels.mean()); smooth = (conversions + 20 * global_cvr) / (count + 20)
    product_price = dict(zip(products, item_price, strict=True))
    average_price = grouped.product_id.apply(lambda values: np.mean([product_price[str(v)] for v in values])).reindex(users, fill_value=0).to_numpy(dtype=np.float32)
    max_ts = int(past.click_timestamp.max()); last_ts = grouped.click_timestamp.max().reindex(users, fill_value=max_ts).to_numpy(dtype=np.float32)
    recency = np.log1p(np.maximum(max_ts - last_ts, 0))
    stats_raw = np.column_stack((count, conversions, smooth, average_price, np.log1p(count), recency)).astype(np.float32)
    # All aggregate normalizers are fit on Past users only.
    stats = (stats_raw - stats_raw.mean(0)) / np.maximum(stats_raw.std(0), 1e-6)
    last = past.sort_values("click_timestamp").drop_duplicates("user_id", keep="last").set_index("user_id")
    eval_context = np.column_stack([_encode(last.reindex(users)[field], vocabs[field]) for field in ("device_type", "audience_id", "hour", "day_of_week")])
    metadata = {"evaluation_users": len(users), "catalogue_products": len(products), "total_future_truth_items": total_truth, "searchable_future_truth_items": warm_total, "non_searchable_future_truth_items": total_truth - warm_total, "searchable_truth_ratio": warm_total / total_truth, "users_with_searchable_truth": sum(bool(v) for v in warm_truth.values()), "price_normalization": {"transform": "log1p(max(price, 0))", "mean": mean, "std": std, "fit": "Past catalogue only"}, "user_aggregates": {"source": "Past only", "definitions": ["interaction_count", "conversion_count", "(conversion_count + 20*global_cvr)/(interaction_count + 20)", "mean_normalized_log_price", "log1p(interaction_count)", "log1p(max_past_timestamp-last_past_timestamp)"]}, "cohort_seed": config.seed}
    return PreparedData(users, products, train_user, train_item, train_context, train_weight, eval_context, stats.astype(np.float32), item_features, item_price, histories, truth, warm_truth, vocabs, metadata)


def _mlp(inputs: int, hidden: tuple[int, ...], output: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.GELU()]; inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, output))


class SimpleTwoTowerV2Model(nn.Module):
    def __init__(self, data: PreparedData, config: SimpleTwoTowerV2Config, variant: str) -> None:
        super().__init__(); self.variant = variant; self.output_dim = config.embedding_dim
        self.user_id = nn.Embedding(len(data.vocabs["user_id"]) + 2, config.feature_embedding_dim, padding_idx=PAD_INDEX)
        self.item_id = nn.Embedding(len(data.vocabs["product_id"]) + 2, config.feature_embedding_dim, padding_idx=PAD_INDEX)
        self.item_embeddings = nn.ModuleList([nn.Embedding(len(data.vocabs[field]) + 2, config.feature_embedding_dim, padding_idx=PAD_INDEX) for field in ITEM_FIELDS])
        self.context_embeddings = nn.ModuleList([nn.Embedding(len(data.vocabs[field]) + 2, config.feature_embedding_dim, padding_idx=PAD_INDEX) for field in USER_FIELDS])
        item_size = config.feature_embedding_dim if variant == "v0_id_only" else config.feature_embedding_dim * (1 + len(ITEM_FIELDS)) + 1
        user_size = config.feature_embedding_dim if variant in ("v0_id_only", "v1_item_content") else config.feature_embedding_dim * (1 + len(USER_FIELDS)) + 6
        self.item_tower, self.user_tower = _mlp(item_size, config.hidden_dims, config.embedding_dim), _mlp(user_size, config.hidden_dims, config.embedding_dim)
        self.register_buffer("item_ids", torch.as_tensor(_encode(data.products, data.vocabs["product_id"])))
        self.register_buffer("item_features", torch.as_tensor(data.item_features)); self.register_buffer("item_price", torch.as_tensor(data.item_price))
        self.register_buffer("user_ids", torch.as_tensor(_encode(data.users, data.vocabs["user_id"]))); self.register_buffer("user_stats", torch.as_tensor(data.user_stats))

    def encode_items(self, positions: Tensor) -> Tensor:
        parts = [self.item_id(self.item_ids[positions])]
        if self.variant != "v0_id_only": parts += [emb(self.item_features[positions, i]) for i, emb in enumerate(self.item_embeddings)] + [self.item_price[positions].unsqueeze(1)]
        result = self.item_tower(torch.cat(parts, 1)); return F.normalize(result, dim=1)

    def encode_users(self, positions: Tensor, context: Tensor) -> Tensor:
        parts = [self.user_id(self.user_ids[positions])]
        if self.variant not in ("v0_id_only", "v1_item_content"):
            parts += [emb(context[:, i]) for i, emb in enumerate(self.context_embeddings)] + [self.user_stats[positions]]
        result = self.user_tower(torch.cat(parts, 1)); return F.normalize(result, dim=1)


def duplicate_aware_inbatch_loss(logits: Tensor, positive_items: Tensor, weights: Tensor) -> Tensor:
    """Multi-positive InfoNCE: duplicate positives never become false negatives."""
    mask = positive_items[:, None].eq(positive_items[None, :])
    numerator = torch.logsumexp(logits.masked_fill(~mask, float("-inf")), dim=1)
    per_example = torch.logsumexp(logits, dim=1) - numerator
    return (per_example * weights).sum() / weights.sum().clamp_min(1)


def _optimizer(model: nn.Module, config: SimpleTwoTowerV2Config) -> torch.optim.Optimizer:
    args = dict(lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.use_fused_adamw and torch.cuda.is_available():
        try: return torch.optim.AdamW(model.parameters(), fused=True, **args)
        except (TypeError, RuntimeError): pass
    return torch.optim.AdamW(model.parameters(), **args)


def _device_array(values: np.ndarray, device: torch.device) -> Tensor:
    """Pinned async copies for the custom contiguous-array batch path."""
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    return tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tensor.to(device)


def _sample_past_negatives(users: np.ndarray, data: PreparedData, count: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized draws with bounded, set-based rejection of known Past pairs."""
    result = rng.integers(0, len(data.products), size=(len(users), count), dtype=np.int64)
    for _ in range(3):
        bad = np.zeros_like(result, dtype=bool)
        for row, user in enumerate(users):
            history = data.histories[int(user)]
            if history: bad[row] = np.fromiter((int(item) in history for item in result[row]), dtype=bool, count=count)
        if not bad.any(): return result
        result[bad] = rng.integers(0, len(data.products), size=int(bad.sum()), dtype=np.int64)
    # Extremely heavy-history users are uncommon; finish safely rather than
    # silently training on their known positives as negatives.
    for row in np.flatnonzero(bad.any(1)):
        available = np.setdiff1d(np.arange(len(data.products)), np.fromiter(data.histories[int(users[row])], dtype=np.int64), assume_unique=False)
        if len(available): result[row, bad[row]] = rng.choice(available, size=int(bad[row].sum()), replace=len(available) < int(bad[row].sum()))
    return result


def train_model(model: SimpleTwoTowerV2Model, data: PreparedData, config: SimpleTwoTowerV2Config, device: torch.device) -> dict[str, float]:
    rng = np.random.default_rng(config.seed); model.to(device); model.train(); optimizer = _optimizer(model, config)
    users, items, context, weights = data.train_user, data.train_item, data.train_context, data.train_weight
    start, samples, batches, loss_sum = time.perf_counter(), 0, 0, 0.0
    bf16 = config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    for _ in range(config.epochs):
        for offset in range(0, len(users), config.batch_size):
            order = rng.permutation(len(users)) if offset == 0 else order
            indices = order[offset:offset + config.batch_size]
            u = _device_array(users[indices], device); p = _device_array(items[indices], device)
            c = _device_array(context[indices], device); w = _device_array(weights[indices], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                uv, pv = model.encode_users(u, c), model.encode_items(p)
                if model.variant == "v3_in_batch_negatives": loss = duplicate_aware_inbatch_loss(uv @ pv.T / config.temperature, p, w)
                else:
                    negatives = _sample_past_negatives(users[indices], data, config.negative_samples, rng)
                    nv = model.encode_items(_device_array(negatives.reshape(-1), device)).reshape(len(indices), config.negative_samples, -1)
                    logits = torch.cat(((uv * pv).sum(1, keepdim=True), torch.einsum("bd,bnd->bn", uv, nv)), 1) / config.temperature
                    loss = (F.cross_entropy(logits, torch.zeros(len(indices), dtype=torch.long, device=device), reduction="none") * w).sum() / w.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            samples += len(indices); batches += 1; loss_sum += float(loss.detach())
    elapsed = time.perf_counter() - start
    return {"training_wall_seconds": elapsed, "training_rows_per_second": samples / max(elapsed, 1e-9), "training_batches_per_second": batches / max(elapsed, 1e-9), "train_loss": loss_sum / max(batches, 1), "bf16_active": bf16}


@torch.no_grad()
def _embeddings(model: SimpleTwoTowerV2Model, count: int, batch_size: int, device: torch.device, *, users: bool, context: np.ndarray | None = None) -> np.ndarray:
    model.eval(); chunks = []
    for start in range(0, count, batch_size):
        positions = torch.arange(start, min(start + batch_size, count), device=device)
        value = model.encode_users(positions, torch.as_tensor(context[start:start + len(positions)], device=device)) if users else model.encode_items(positions)
        chunks.append(value.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _search(index: Any, queries: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    return index.search(queries, count)


def _build_index(embeddings: np.ndarray) -> Any:
    try:
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1]); index.add(embeddings); return index
    except ImportError:
        class NumpyIP:
            def __init__(self, matrix: np.ndarray): self.matrix, self.ntotal = matrix, len(matrix)
            def search(self, q: np.ndarray, k: int):
                scores = q @ self.matrix.T; pos = np.argpartition(-scores, kth=min(k - 1, scores.shape[1] - 1), axis=1)[:, :k]; order = np.take_along_axis(scores, pos, axis=1).argsort(1)[:, ::-1]; return np.take_along_axis(scores, np.take_along_axis(pos, order, 1), 1), np.take_along_axis(pos, order, 1)
        return NumpyIP(embeddings)


def write_candidates(model: SimpleTwoTowerV2Model, data: PreparedData, config: SimpleTwoTowerV2Config, device: torch.device, variant: str) -> dict[str, float]:
    output = config.output_dir / "candidates" / f"{variant}_top{config.top_k}.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    index_start = time.perf_counter(); item_vectors = _embeddings(model, len(data.products), config.inference_batch_size, device, users=False); index = _build_index(item_vectors); index_seconds = time.perf_counter() - index_start
    retrieve_start, latencies, retained, short = time.perf_counter(), [], 0, 0
    requested = min(len(data.products), max(config.top_k, int(math.ceil(config.top_k * config.retrieval_oversample_ratio))))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["user_id", "candidate_ad_id", "two_tower_score", "rank", "model_variant"])
        for start in range(0, len(data.users), config.retrieval_batch_size):
            t0 = time.perf_counter(); stop = min(start + config.retrieval_batch_size, len(data.users)); vectors = _embeddings(model, stop - start, config.inference_batch_size, device, users=True, context=data.eval_context[start:stop])
            scores, positions = _search(index, vectors, requested)
            for row, user_index in enumerate(range(start, stop)):
                rank = 0
                for score, position in zip(scores[row], positions[row], strict=True):
                    if config.exclude_seen_items and int(position) in data.histories[user_index]: continue
                    rank += 1; writer.writerow([data.users[user_index], data.products[int(position)], float(score), rank, variant])
                    if rank == config.top_k: break
                retained += rank; short += int(rank < config.top_k)
            latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - retrieve_start
    return {"candidate_path": str(output), "index_build_seconds": index_seconds, "retrieval_wall_seconds": elapsed, "retrieval_users_per_second": len(data.users) / max(elapsed, 1e-9), "faiss_search_seconds_included": elapsed, "retrieval_batch_p50_seconds": float(np.percentile(latencies, 50)), "retrieval_batch_p95_seconds": float(np.percentile(latencies, 95)), "average_retained_candidates": retained / len(data.users), "users_below_top_k": short, "unique_catalogue_items": len(data.products)}


def evaluate_candidates(path: Path, truth: Mapping[str, set[str]], warm_truth: Mapping[str, set[str]], catalogue: set[str]) -> dict[str, float]:
    sums = defaultdict(float); ndcg = mrr = 0.0; recalled_at_100: set[str] = set(); seen_users: set[str] = set(); previous: str | None = None; last_rank = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user, item, rank = row["user_id"], row["candidate_ad_id"], int(row["rank"])
            if user != previous: previous, last_rank = user, 0
            if rank != last_rank + 1: raise AssertionError("Candidate rows must be contiguous user groups with ranks starting at 1")
            last_rank = rank; seen_users.add(user)
            if rank <= 100: recalled_at_100.add(item)
            target, warm = truth.get(user, set()), warm_truth.get(user, set())
            for cutoff in CUT_OFFS:
                if rank <= cutoff and item in target: sums[f"overall_recall@{cutoff}"] += 1 / len(target)
                if rank <= cutoff and item in warm: sums[f"warm_recall@{cutoff}"] += 1 / len(warm)
    # Re-read grouped only for rank-sensitive MRR/NDCG, keeping the normal path streaming and bounded.
    per_user: dict[str, list[str]] = defaultdict(list)
    for chunk in pd.read_csv(path, usecols=["user_id", "candidate_ad_id"], chunksize=200_000):
        for user, item in chunk.itertuples(index=False, name=None): per_user[str(user)].append(str(item))
    for user, target in truth.items():
        candidates = per_user.get(user, [])[:100]; gains = [int(item in target) for item in candidates]
        first = next((i + 1 for i, gain in enumerate(gains) if gain), None)
        mrr += 0.0 if first is None else 1 / first
        dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains)); ideal = sum(1 / math.log2(i + 2) for i in range(min(len(target), 100))); ndcg += dcg / ideal if ideal else 0.0
    users, warm_users = len(truth), sum(bool(values) for values in warm_truth.values())
    result = {key: value / (warm_users if key.startswith("warm_") else users) for key, value in sums.items()}
    result.update({f"hit_rate@{cutoff}": sum(1 for user, target in truth.items() if set(per_user.get(user, [])[:cutoff]) & target) / users for cutoff in CUT_OFFS})
    result.update({"ndcg@100": ndcg / users, "mrr@100": mrr / users, "unique_recalled_items": len(recalled_at_100), "catalog_coverage@100": len(recalled_at_100) / max(len(catalogue), 1), "candidate_users": len(seen_users)})
    return result


def popularity_metrics(path: Path, truth: Mapping[str, set[str]], data: PreparedData) -> dict[str, float]:
    counts = np.bincount(data.train_item, minlength=len(data.products)); order = np.argsort(counts); groups = {"head": set(order[int(.8 * len(order)):]), "mid": set(order[int(.5 * len(order)):int(.8 * len(order))]), "tail": set(order[:int(.5 * len(order))])}
    if any(not values for values in groups.values()): raise RuntimeError("Popularity buckets must be non-empty")
    products = data.products; candidates: dict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(path, usecols=["user_id", "candidate_ad_id", "rank"], chunksize=200_000):
        for user, item, rank in chunk.itertuples(index=False, name=None):
            if int(rank) <= 100: candidates[str(user)].add(str(item))
    result = {}
    for name, positions in groups.items():
        valid = {products[i] for i in positions}; denominator = sum(len(values & valid) for values in truth.values())
        if not denominator: raise RuntimeError(f"Popularity bucket {name} has no Future-A truth")
        numerator = sum(len(candidates[user] & values & valid) for user, values in truth.items()); result[f"{name}_recall@100"] = numerator / denominator
    return result


def run_ablation(config: SimpleTwoTowerV2Config) -> pd.DataFrame:
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")
    data = prepare_data(config); config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(config.output_dir / "evaluation_users.txt", data.users, fmt="%s")
    rows: list[dict[str, Any]] = []
    for variant in config.variants:
        if variant not in {"v0_id_only", "v1_item_content", "v2_user_context_stats", "v3_in_batch_negatives"}: raise ValueError(f"Unknown variant: {variant}")
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        model: Any = SimpleTwoTowerV2Model(data, config, variant)
        if config.use_torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        train = train_model(model, data, config, device); retrieval = write_candidates(model, data, config, device, variant)
        metrics = evaluate_candidates(Path(retrieval["candidate_path"]), data.truth, data.warm_truth, set(data.products))
        metrics.update(popularity_metrics(Path(retrieval["candidate_path"]), data.truth, data)); gpu = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        try:
            import psutil
            host_mb = psutil.Process(os.getpid()).memory_info().rss / 2**20
        except ImportError: host_mb = float("nan")
        rows.append({"model": variant, **data.metadata, **train, **retrieval, **metrics, "peak_gpu_mb": gpu, "peak_host_ram_mb": host_mb})
    frame = pd.DataFrame(rows); metrics_path = config.output_dir / "metrics"; metrics_path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_path / "simple_two_tower_v2_ablation.csv", index=False)
    (metrics_path / "simple_two_tower_v2_metadata.json").write_text(json.dumps({"config": asdict(config) | {"past_path": str(config.past_path), "future_a_path": str(config.future_a_path), "output_dir": str(config.output_dir)}, "data": data.metadata, "contract": "Past-only fitting/training/features; Future-A evaluation labels only; Future-B untouched; warm-catalogue retrieval only."}, indent=2, default=str), encoding="utf-8")
    return frame
