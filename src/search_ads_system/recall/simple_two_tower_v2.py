"""Small, auditable strict-temporal Two-Tower recall baseline.

This module intentionally does not import the historical content Two-Tower
implementations.  It reads only a Past directory and a Future-A directory;
Future-A is used exclusively for the fixed evaluation cohort and labels.
"""
from __future__ import annotations

import csv
import hashlib
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
    # V4 is deliberately separate from ``negative_samples`` so changing its
    # composition can never alter the historical V0--V2 sampled objective.
    random_negative_count: int = 3
    hard_negative_count: int = 2
    hard_negative_retrieval_topn: int = 200
    hard_negative_rank_start: int = 10
    hard_negative_strategy: str = "top_band"
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
    use_preprocessing_cache: bool = True
    rebuild_preprocessing_cache: bool = False
    cache_dir: Path | None = None
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


def _eligible_users(past_path: Path, future_path: Path, max_users: int, seed: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    """Scan slim ID columns once; retain Future-A's three required columns."""
    def past_users() -> set[str]:
        values: set[str] = set()
        for part in _parts(past_path):
            for chunk in pd.read_csv(part, usecols=["user_id", "product_id"], chunksize=200_000, low_memory=False):
                valid = (_clean(chunk.user_id) != "") & (_clean(chunk.product_id) != "")
                values.update(_clean(chunk.loc[valid, "user_id"]).tolist())
        return values
    timings: dict[str, float] = {}; started = time.perf_counter(); future_frames = []
    for part in _parts(future_path):
        for chunk in pd.read_csv(part, usecols=["user_id", "product_id", "click_timestamp"], chunksize=200_000, low_memory=False):
            chunk["user_id"], chunk["product_id"] = _clean(chunk.user_id), _clean(chunk.product_id)
            chunk = chunk[(chunk.user_id != "") & (chunk.product_id != "")]
            if len(chunk): future_frames.append(chunk)
    future = pd.concat(future_frames, ignore_index=True) if future_frames else pd.DataFrame(columns=["user_id", "product_id", "click_timestamp"])
    timings["future_a_loading"] = time.perf_counter() - started; _log(f"[preprocess] Future-A loaded: {len(future):,} valid rows, {timings['future_a_loading']:.1f}s")
    started = time.perf_counter(); overlap = sorted(past_users() & set(future.user_id.unique())); timings["past_eligibility_scan"] = time.perf_counter() - started
    if not overlap: raise RuntimeError("No users occur in both Past and Future-A")
    rng = np.random.default_rng(seed)
    chosen = np.asarray(overlap if len(overlap) <= max_users else rng.choice(overlap, max_users, replace=False), dtype=str)
    return np.sort(chosen), future, timings


def _vocab(values: Iterable[object]) -> dict[str, int]:
    # Missing is intentionally OOV-safe (index 1); index 0 is reserved padding.
    known = sorted({value for raw in values if (value := _id(raw)) is not None})
    return {value: index + 2 for index, value in enumerate(known)}


def _encode(values: Iterable[object], vocab: Mapping[str, int]) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return _clean(series).map(vocab).fillna(OOV_INDEX).to_numpy(dtype=np.int64)


def _log(message: str) -> None:
    print(message, flush=True)


def _stage(timings: dict[str, float], name: str, started: float) -> None:
    elapsed = time.perf_counter() - started; timings[name] = elapsed; _log(f"[preprocess] {name}: {elapsed:.1f}s")


def _read_past_once(past_path: Path, cohort: set[str], required: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One full Past pass produces both selected training rows and catalogue rows."""
    selected, catalogue_parts = [], []
    catalogue_columns = ["user_id", "product_id", "product_price", *ITEM_FIELDS]
    files = _parts(past_path); _log(f"[preprocess] scanning Past files ({len(files)} files)...")
    for file_index, part in enumerate(files, start=1):
        for chunk in pd.read_csv(part, usecols=required, chunksize=200_000, low_memory=False):
            chunk["user_id"], chunk["product_id"] = _clean(chunk.user_id), _clean(chunk.product_id)
            chunk = chunk[(chunk.user_id != "") & (chunk.product_id != "")]
            if not len(chunk): continue
            catalogue_parts.append(chunk[catalogue_columns].drop_duplicates("product_id", keep="last"))
            chosen = chunk[chunk.user_id.isin(cohort)]
            if len(chosen): selected.append(chosen)
        if file_index % 10 == 0 or file_index == len(files): _log(f"[preprocess] Past scan progress: {file_index}/{len(files)} files")
    empty_selected = pd.DataFrame(columns=required); empty_catalogue = pd.DataFrame(columns=catalogue_columns)
    return (pd.concat(selected, ignore_index=True) if selected else empty_selected, pd.concat(catalogue_parts, ignore_index=True) if catalogue_parts else empty_catalogue)


def _source_fingerprint(config: SimpleTwoTowerV2Config) -> str:
    files = _parts(config.past_path) + _parts(config.future_a_path)
    sources = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    payload = {"sources": sources, "max_users": config.max_users, "seed": config.seed, "max_train_rows": config.max_train_rows, "item_fields": ITEM_FIELDS, "user_fields": USER_FIELDS, "schema": 3}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_root(config: SimpleTwoTowerV2Config) -> Path:
    return config.cache_dir or config.output_dir / "simple_two_tower_v2_cache"


def _save_cache(data: PreparedData, config: SimpleTwoTowerV2Config, fingerprint: str) -> float:
    started = time.perf_counter(); root = _cache_root(config); root.mkdir(parents=True, exist_ok=True)
    offsets = [0]; flat: list[int] = []
    for history in data.histories: flat.extend(sorted(history)); offsets.append(len(flat))
    np.savez_compressed(root / "prepared_arrays.npz", users=np.asarray(data.users, dtype=str), products=np.asarray(data.products, dtype=str), train_user=data.train_user, train_item=data.train_item, train_context=data.train_context, train_weight=data.train_weight, eval_context=data.eval_context, user_stats=data.user_stats, item_features=data.item_features, item_price=data.item_price, history_offsets=np.asarray(offsets, dtype=np.int64), history_items=np.asarray(flat, dtype=np.int64))
    payload = {"fingerprint": fingerprint, "metadata": data.metadata, "truth": {key: sorted(value) for key, value in data.truth.items()}, "warm_truth": {key: sorted(value) for key, value in data.warm_truth.items()}, "vocabs": data.vocabs}
    (root / "prepared_metadata.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    elapsed = time.perf_counter() - started; _log(f"[cache] saved preprocessing artifacts: {root} ({elapsed:.1f}s)"); return elapsed


def _load_cache(config: SimpleTwoTowerV2Config, fingerprint: str) -> PreparedData | None:
    root = _cache_root(config); metadata_path, arrays_path = root / "prepared_metadata.json", root / "prepared_arrays.npz"
    if config.rebuild_preprocessing_cache or not config.use_preprocessing_cache or not metadata_path.is_file() or not arrays_path.is_file(): return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        _log("[cache] preprocessing fingerprint changed; rebuilding."); return None
    started = time.perf_counter()
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    offsets, flat = values.pop("history_offsets"), values.pop("history_items")
    histories = [set(flat[offsets[index]:offsets[index + 1]].tolist()) for index in range(len(offsets) - 1)]
    data = PreparedData(**values, histories=histories, truth={key: set(value) for key, value in payload["truth"].items()}, warm_truth={key: set(value) for key, value in payload["warm_truth"].items()}, vocabs={key: {str(item): int(code) for item, code in value.items()} for key, value in payload["vocabs"].items()}, metadata=payload["metadata"])
    elapsed = time.perf_counter() - started; data.metadata["preprocessing_cache_hit"] = True; data.metadata["cache_load_seconds"] = elapsed
    _log(f"[cache] loaded preprocessing artifacts: {root} ({elapsed:.1f}s)"); return data


def prepare_data(config: SimpleTwoTowerV2Config) -> PreparedData:
    """Load a validated cache or build shared Past-only arrays once for V0–V3."""
    fingerprint = _source_fingerprint(config)
    if cached := _load_cache(config, fingerprint): return cached
    return _prepare_data_uncached(config, fingerprint)


def _prepare_data_uncached(config: SimpleTwoTowerV2Config, fingerprint: str) -> PreparedData:
    """Reference preprocessing semantics, optimized to eliminate redundant scans/row loops."""
    timings: dict[str, float] = {}; started = time.perf_counter(); _log("[preprocess] scanning Future-A and building evaluation cohort...")
    chosen, future_all, scan_timings = _eligible_users(config.past_path, config.future_a_path, config.max_users, config.seed); timings.update(scan_timings)
    _stage(timings, "evaluation_cohort_construction", started)
    _log(f"[preprocess] cohort built: {len(chosen):,} users")
    cohort = set(chosen.tolist()); future = future_all[future_all.user_id.isin(cohort)].copy()
    required = ["user_id", "product_id", "click_timestamp", "conversion_label", "product_price", *ITEM_FIELDS, "device_type", "audience_id"]
    started = time.perf_counter(); past, catalogue = _read_past_once(config.past_path, cohort, required); _stage(timings, "loading_past_and_catalogue", started)
    if past.empty or catalogue.empty or future.empty: raise RuntimeError("Past, catalogue, and Future-A must all contain valid interactions")
    if config.max_train_rows is not None: past = past.iloc[:config.max_train_rows].copy()
    past["click_timestamp"] = pd.to_numeric(past.click_timestamp, errors="coerce").fillna(0).astype("int64")
    future["click_timestamp"] = pd.to_numeric(future.click_timestamp, errors="coerce").fillna(0).astype("int64")
    if int(past.click_timestamp.max()) >= int(future.click_timestamp.min()):
        raise RuntimeError("Temporal contract failed: max(Past timestamp) must be < min(Future-A timestamp)")
    users = chosen
    user_to_i = {value: index for index, value in enumerate(users)}
    started = time.perf_counter(); catalogue = catalogue.drop_duplicates("product_id", keep="last").reset_index(drop=True); _stage(timings, "past_catalogue_construction", started)
    products = catalogue.product_id.astype(str).to_numpy()
    past = past[past.product_id.isin(products)].copy()
    if past.empty: raise RuntimeError("No Past training rows map to the Past product catalogue")
    started = time.perf_counter(); truth = future.groupby("user_id", sort=False).product_id.agg(set).to_dict(); product_set = set(products); warm_truth = {user: values & product_set for user, values in truth.items()}; _stage(timings, "candidate_evaluation_preparation", started)
    total_truth, warm_total = sum(map(len, truth.values())), sum(map(len, warm_truth.values()))
    overlap = set(products) & set().union(*truth.values())
    if not overlap or not warm_total:
        examples = {"catalogue": products[:3].tolist(), "future": sorted(set().union(*truth.values()))[:3]}
        raise RuntimeError(f"Past catalogue and Future-A raw product IDs have zero overlap: {examples}")
    started = time.perf_counter(); user_vocab = _vocab(users)
    vocabs = {"user_id": user_vocab, "product_id": _vocab(products)}
    for field in ITEM_FIELDS + ("device_type", "audience_id"):
        source = catalogue[field] if field in ITEM_FIELDS else past[field]
        vocabs[field] = _vocab(source)
    for field in ("hour", "day_of_week"):
        vocabs[field] = _vocab(map(str, range(24 if field == "hour" else 7)))
    _stage(timings, "vocab_construction", started)
    started = time.perf_counter()
    item_features = np.column_stack([_encode(catalogue[field], vocabs[field]) for field in ITEM_FIELDS]).astype(np.int64)
    raw_price = pd.to_numeric(catalogue.product_price, errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
    log_price = np.log1p(raw_price)
    mean, std = float(log_price.mean()), max(float(log_price.std()), 1e-6)
    item_price = ((log_price - mean) / std).astype(np.float32)
    if not np.isfinite(item_price).all(): raise ValueError("Past-fitted normalized item prices are not finite")
    _stage(timings, "item_feature_construction", started)
    started = time.perf_counter()
    past["hour"] = (past.click_timestamp // 3600 % 24).astype(str); past["day_of_week"] = ((past.click_timestamp // 86400 + 4) % 7).astype(str)
    train_context = np.column_stack([_encode(past[field], vocabs[field]) for field in ("device_type", "audience_id", "hour", "day_of_week")])
    train_user = pd.Categorical(past.user_id, categories=users).codes.astype(np.int64); train_item = pd.Categorical(past.product_id, categories=products).codes.astype(np.int64)
    labels = pd.to_numeric(past.conversion_label, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    train_weight = np.where(labels == 1, config.conversion_weight, config.click_weight).astype(np.float32)
    _stage(timings, "training_row_encoding", started)
    started = time.perf_counter()
    history_series = pd.DataFrame({"user": train_user, "item": train_item}).groupby("user", sort=False).item.agg(set)
    histories = [history_series.get(index, set()) for index in range(len(users))]
    # A user who has seen the complete warm catalogue has no valid sampled
    # negative.  Exclude only those rows instead of fabricating a false one.
    valid_negative = np.asarray([len(histories[int(user)]) < len(products) for user in train_user], dtype=bool)
    train_user, train_item = train_user[valid_negative], train_item[valid_negative]
    train_context, train_weight = train_context[valid_negative], train_weight[valid_negative]
    if not len(train_user): raise RuntimeError("No training rows have an available Past-catalogue negative")
    _stage(timings, "past_history_construction", started)
    started = time.perf_counter(); past["_item_price"] = item_price[pd.Categorical(past.product_id, categories=products).codes]
    past["_conversion"] = pd.to_numeric(past.conversion_label, errors="coerce").fillna(0).astype(np.float32)
    grouped = past.sort_values("click_timestamp").groupby("user_id", sort=False)
    aggregates = grouped.agg(interactions=("product_id", "size"), conversions=("_conversion", "sum"), average_price=("_item_price", "mean"), last_timestamp=("click_timestamp", "max")).reindex(users)
    count = aggregates.interactions.fillna(0).to_numpy(dtype=np.float32)
    conversions = aggregates.conversions.fillna(0).to_numpy(dtype=np.float32)
    global_cvr = float(labels.mean()); smooth = (conversions + 20 * global_cvr) / (count + 20)
    average_price = aggregates.average_price.fillna(0).to_numpy(dtype=np.float32)
    max_ts = int(past.click_timestamp.max()); last_ts = aggregates.last_timestamp.fillna(max_ts).to_numpy(dtype=np.float32)
    recency = np.log1p(np.maximum(max_ts - last_ts, 0))
    stats_raw = np.column_stack((count, conversions, smooth, average_price, np.log1p(count), recency)).astype(np.float32)
    # All aggregate normalizers are fit on Past users only.
    stats = (stats_raw - stats_raw.mean(0)) / np.maximum(stats_raw.std(0), 1e-6)
    last = past.sort_values("click_timestamp").drop_duplicates("user_id", keep="last").set_index("user_id")
    eval_context = np.column_stack([_encode(last.reindex(users)[field], vocabs[field]) for field in ("device_type", "audience_id", "hour", "day_of_week")])
    _stage(timings, "user_aggregate_construction", started)
    metadata = {"evaluation_users": len(users), "catalogue_products": len(products), "total_future_truth_items": total_truth, "searchable_future_truth_items": warm_total, "non_searchable_future_truth_items": total_truth - warm_total, "searchable_truth_ratio": warm_total / total_truth, "users_with_searchable_truth": sum(bool(v) for v in warm_truth.values()), "price_normalization": {"transform": "log1p(max(price, 0))", "mean": mean, "std": std, "fit": "Past catalogue only"}, "user_aggregates": {"source": "Past only", "definitions": ["interaction_count", "conversion_count", "(conversion_count + 20*global_cvr)/(interaction_count + 20)", "mean_normalized_log_price", "log1p(interaction_count)", "log1p(max_past_timestamp-last_past_timestamp)"]}, "cohort_seed": config.seed, "preprocessing_stage_seconds": timings, "preprocessing_wall_seconds": sum(timings.values())}
    data = PreparedData(users, products, train_user, train_item, train_context, train_weight, eval_context, stats.astype(np.float32), item_features, item_price, histories, truth, warm_truth, vocabs, metadata)
    metadata["preprocessing_cache_hit"] = False
    if config.use_preprocessing_cache: metadata["cache_write_seconds"] = _save_cache(data, config, fingerprint)
    _log(f"[preprocess] complete: {metadata['preprocessing_wall_seconds']:.1f}s")
    return data


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
        # v4_hard_negatives is intentionally architecturally identical to V2.
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


def _synchronize_cuda(device: torch.device) -> None:
    """Put CUDA work on the correct side of a coarse wall-clock boundary."""
    if device.type == "cuda": torch.cuda.synchronize(device)


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


def _negative_pool(user: int, data: PreparedData, forbidden: set[int]) -> np.ndarray:
    """Return the Past catalogue entries safe for this user and batch row."""
    excluded = data.histories[int(user)] | forbidden
    return np.fromiter((item for item in range(len(data.products)) if item not in excluded), dtype=np.int64)


def sample_mixed_negatives(
    users: np.ndarray,
    positives: np.ndarray,
    hard_pools: list[np.ndarray],
    data: PreparedData,
    config: SimpleTwoTowerV2Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample V4's random + hard negatives without ever fabricating positives.

    A short hard pool is filled by additional *random* Past-catalogue samples.
    This is conservative: it preserves the requested fixed loss width while
    never promoting an unclicked candidate to a label-derived negative.
    """
    width = config.random_negative_count + config.hard_negative_count
    result = np.empty((len(users), width), dtype=np.int64)
    requested_hard = min(config.hard_negative_count, width)
    supplied_hard = 0
    examples_with_requested = 0
    duplicate_count = 0
    fallback_examples = 0
    for row, (user, positive, pool) in enumerate(zip(users, positives, hard_pools, strict=True)):
        # Mining should already enforce this; validate it again at the boundary
        # where training labels are constructed.
        candidates = np.asarray(pool, dtype=np.int64)
        candidates = candidates[(candidates != int(positive)) & ~np.isin(candidates, list(data.histories[int(user)]))]
        candidates = np.unique(candidates)
        take = min(requested_hard, len(candidates))
        hard = rng.choice(candidates, size=take, replace=False) if take else np.empty(0, dtype=np.int64)
        supplied_hard += take
        examples_with_requested += int(take == requested_hard)
        fallback_examples += int(take < requested_hard)
        # The random part includes the hard shortfall.  Exclude selected hard
        # entries where possible, so duplicates are only possible for tiny
        # catalogues which cannot provide enough distinct negatives.
        random_needed = config.random_negative_count + (requested_hard - take)
        available = _negative_pool(int(user), data, set(hard.tolist()))
        if not len(available):
            # This can occur only when all safe candidates were selected as
            # hard negatives. Reuse those safe values rather than a positive.
            available = hard
        random_part = rng.choice(available, size=random_needed, replace=len(available) < random_needed)
        values = np.concatenate((random_part, hard))
        if len(np.unique(values)) != len(values): duplicate_count += len(values) - len(np.unique(values))
        result[row] = values
    total = max(len(users), 1)
    return result, {
        "mixed_requested_hard_negatives": float(requested_hard),
        "mixed_average_hard_negatives": supplied_hard / total,
        "mixed_fraction_requested_hard": examples_with_requested / total,
        "mixed_fraction_random_fallback": fallback_examples / total,
        "mixed_duplicate_rate": duplicate_count / max(len(users) * width, 1),
    }


def train_model(
    model: SimpleTwoTowerV2Model,
    data: PreparedData,
    config: SimpleTwoTowerV2Config,
    device: torch.device,
    *,
    hard_pools: list[np.ndarray] | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(config.seed); model.to(device); model.train(); optimizer = _optimizer(model, config)
    users, items, context, weights = data.train_user, data.train_item, data.train_context, data.train_weight
    _synchronize_cuda(device)
    start, samples, batches, loss_sum = time.perf_counter(), 0, 0, 0.0
    mixed_totals: defaultdict[str, float] = defaultdict(float)
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
                    if model.variant == "v4_hard_negatives":
                        if hard_pools is None: raise ValueError("V4 requires Past-mined hard-negative pools")
                        negatives, sampled = sample_mixed_negatives(users[indices], items[indices], [hard_pools[int(i)] for i in indices], data, config, rng)
                        for key, value in sampled.items(): mixed_totals[key] += value * len(indices)
                    else:
                        negatives = _sample_past_negatives(users[indices], data, config.negative_samples, rng)
                    nv = model.encode_items(_device_array(negatives.reshape(-1), device)).reshape(len(indices), negatives.shape[1], -1)
                    logits = torch.cat(((uv * pv).sum(1, keepdim=True), torch.einsum("bd,bnd->bn", uv, nv)), 1) / config.temperature
                    loss = (F.cross_entropy(logits, torch.zeros(len(indices), dtype=torch.long, device=device), reduction="none") * w).sum() / w.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            samples += len(indices); batches += 1; loss_sum += float(loss.detach())
    _synchronize_cuda(device)
    elapsed = time.perf_counter() - start
    result = {"training_wall_seconds": elapsed, "training_rows_per_second": samples / max(elapsed, 1e-9), "training_batches_per_second": batches / max(elapsed, 1e-9), "train_loss": loss_sum / max(batches, 1), "bf16_active": bf16}
    if mixed_totals:
        result.update({key: value / max(samples, 1) for key, value in mixed_totals.items()})
    return result


@torch.no_grad()
def _embeddings(model: SimpleTwoTowerV2Model, count: int, batch_size: int, device: torch.device, *, users: bool, context: np.ndarray | None = None) -> np.ndarray:
    model.eval(); chunks = []
    for start in range(0, count, batch_size):
        positions = torch.arange(start, min(start + batch_size, count), device=device)
        value = model.encode_users(positions, torch.as_tensor(context[start:start + len(positions)], device=device)) if users else model.encode_items(positions)
        chunks.append(value.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _search(index: Any, queries: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    # FAISS's native bindings require C-contiguous float32 input; PyTorch's
    # CPU views are not guaranteed to satisfy that on every platform.
    return index.search(np.ascontiguousarray(queries, dtype=np.float32), count)


def _build_index(embeddings: np.ndarray) -> Any:
    try:
        import faiss
        # Some local macOS FAISS builds cannot initialize their default OpenMP
        # pool; one thread remains batched and avoids a native crash.
        if hasattr(faiss, "omp_set_num_threads"): faiss.omp_set_num_threads(1)
        values = np.ascontiguousarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(values.shape[1]); index.add(values); return index
    except ImportError:
        class NumpyIP:
            def __init__(self, matrix: np.ndarray): self.matrix, self.ntotal = matrix, len(matrix)
            def search(self, q: np.ndarray, k: int):
                scores = q @ self.matrix.T; pos = np.argpartition(-scores, kth=min(k - 1, scores.shape[1] - 1), axis=1)[:, :k]; order = np.take_along_axis(scores, pos, axis=1).argsort(1)[:, ::-1]; return np.take_along_axis(scores, np.take_along_axis(pos, order, 1), 1), np.take_along_axis(pos, order, 1)
        return NumpyIP(embeddings)


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _v2_teacher_fingerprint(data: PreparedData, config: SimpleTwoTowerV2Config) -> str:
    """Identity of the Past-only V2 teacher, independent of V4 settings."""
    payload = {
        "schema": 1,
        "past_catalogue": _array_fingerprint(data.products, data.item_features, data.item_price),
        "training": _array_fingerprint(data.train_user, data.train_item, data.train_context, data.train_weight),
        "vocabs": data.vocabs,
        "seed": config.seed,
        "architecture": [config.embedding_dim, config.feature_embedding_dim, config.hidden_dims],
        "optimizer": [config.epochs, config.batch_size, config.learning_rate, config.weight_decay, config.negative_samples, config.temperature, config.click_weight, config.conversion_weight, config.use_bf16],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def _load_or_train_v2_teacher(data: PreparedData, config: SimpleTwoTowerV2Config, device: torch.device) -> tuple[SimpleTwoTowerV2Model, dict[str, Any]]:
    """Return a normal V2 teacher without consulting Future-A labels or truth."""
    root = _cache_root(config) / "hard_negatives" / "teachers"; root.mkdir(parents=True, exist_ok=True)
    identity = _v2_teacher_fingerprint(data, config); checkpoint = root / f"v2_teacher_{identity}.pt"
    model = SimpleTwoTowerV2Model(data, config, "v2_user_context_stats")
    started = time.perf_counter()
    if checkpoint.is_file():
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state); model.to(device)
        return model, {"teacher_cache_hit": True, "teacher_cache_load_seconds": time.perf_counter() - started, "teacher_checkpoint": str(checkpoint), "teacher_checkpoint_sha256": _checkpoint_digest(checkpoint)}
    train = train_model(model, data, config, device)
    model.to("cpu")
    torch.save(model.state_dict(), checkpoint)
    model.to(device)
    return model, {"teacher_cache_hit": False, "teacher_training_wall_seconds": train["training_wall_seconds"], "teacher_checkpoint": str(checkpoint), "teacher_checkpoint_sha256": _checkpoint_digest(checkpoint)}


def _hard_negative_fingerprint(data: PreparedData, config: SimpleTwoTowerV2Config, teacher_sha256: str) -> str:
    payload = {
        "schema": 1, "teacher_checkpoint_sha256": teacher_sha256,
        "past_catalogue": _array_fingerprint(data.products),
        "training": _array_fingerprint(data.train_user, data.train_item, data.train_context),
        "feature_vocab": hashlib.sha256(json.dumps(data.vocabs, sort_keys=True).encode()).hexdigest(),
        "seed": config.seed, "topn": config.hard_negative_retrieval_topn,
        "rank_start": config.hard_negative_rank_start, "strategy": config.hard_negative_strategy,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_past_only_hard_negative_contract(data: PreparedData, users: np.ndarray, positives: np.ndarray, pools: list[np.ndarray]) -> None:
    """Boundary assertion: mined negatives are Past-catalogue non-positives only.

    The miner's API intentionally has no Future-A argument. This assertion
    guards the two ways a leaked label could enter its output: a non-catalogue
    item or a Past known-positive item.
    """
    catalogue_size = len(data.products)
    for user, positive, pool in zip(users, positives, pools, strict=True):
        values = np.asarray(pool, dtype=np.int64)
        if np.any(values < 0) or np.any(values >= catalogue_size): raise AssertionError("Hard-negative mining emitted a non-Past-catalogue item")
        if int(positive) in values: raise AssertionError("Current positive item was mined as a negative")
        if any(int(value) in data.histories[int(user)] for value in values): raise AssertionError("Known Past positive was mined as a negative")


@torch.no_grad()
def _mine_hard_negative_pools(model: SimpleTwoTowerV2Model, data: PreparedData, config: SimpleTwoTowerV2Config, device: torch.device) -> tuple[list[np.ndarray], dict[str, float]]:
    if config.hard_negative_strategy != "top_band": raise ValueError(f"Unsupported hard-negative strategy: {config.hard_negative_strategy}")
    if config.hard_negative_retrieval_topn < 1 or config.hard_negative_rank_start < 0: raise ValueError("Hard-negative retrieval parameters must be non-negative and non-empty")
    # Context participates in the key. We only reuse a retrieval pool when V2
    # would produce the identical user representation for that training row.
    keys = np.column_stack((data.train_user, data.train_context)).astype(np.int64, copy=False)
    groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    try:
        import faiss
    except ImportError as exc:
        # Unlike the evaluator's small local fallback, V4 must never degrade
        # into a user-by-catalogue dense score matrix during mining.
        raise RuntimeError("v4_hard_negatives requires faiss-cpu/faiss-gpu; install the repository requirements before mining") from exc
    if hasattr(faiss, "omp_set_num_threads"): faiss.omp_set_num_threads(1)
    model.eval(); item_vectors = _embeddings(model, len(data.products), config.inference_batch_size, device, users=False)
    item_vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(item_vectors.shape[1]); index.add(item_vectors)
    requested = min(len(data.products), config.hard_negative_retrieval_topn)
    group_pools: list[np.ndarray] = []; group_ranks: list[np.ndarray] = []
    rejected = 0
    for start in range(0, len(groups), config.retrieval_batch_size):
        batch = groups[start:start + config.retrieval_batch_size]
        vectors: list[np.ndarray] = []
        for offset in range(0, len(batch), config.inference_batch_size):
            piece = batch[offset:offset + config.inference_batch_size]
            positions = torch.as_tensor(piece[:, 0], device=device)
            context = torch.as_tensor(piece[:, 1:], device=device)
            vectors.append(model.encode_users(positions, context).float().cpu().numpy())
        _, positions = _search(index, np.concatenate(vectors).astype(np.float32, copy=False), requested)
        for key, retrieved in zip(batch, positions, strict=True):
            history = data.histories[int(key[0])]
            allowed = [int(item) for item in retrieved if int(item) not in history]
            rejected += len(retrieved) - len(allowed)
            # rank_start is applied after known-positive removal, so the
            # highest unlabelled-looking candidates can be conservatively skipped.
            selected = np.asarray(allowed[config.hard_negative_rank_start:], dtype=np.int64)
            original_ranks = np.asarray([rank + 1 for rank, item in enumerate(retrieved) if int(item) not in history][config.hard_negative_rank_start:], dtype=np.int64)
            group_pools.append(selected); group_ranks.append(original_ranks)
    pools = [group_pools[int(group)] for group in inverse]
    ranks = [group_ranks[int(group)] for group in inverse]
    assert_past_only_hard_negative_contract(data, data.train_user, data.train_item, pools)
    available = np.asarray([len(pool) for pool in pools], dtype=np.float64)
    all_ranks = np.concatenate([rank for rank in ranks if len(rank)]) if any(len(rank) for rank in ranks) else np.empty(0)
    pool_values = np.concatenate([pool for pool in pools if len(pool)]) if any(len(pool) for pool in pools) else np.empty(0, dtype=np.int64)
    counts = np.bincount(data.train_item, minlength=len(data.products)); order = np.argsort(counts)
    buckets = {"tail": set(order[:int(.5 * len(order))]), "mid": set(order[int(.5 * len(order)):int(.8 * len(order))]), "head": set(order[int(.8 * len(order)):])}
    diagnostics: dict[str, float] = {
        "hard_negative_unique_user_contexts": float(len(groups)), "hard_negative_average_available_per_user_context": float(available.mean()) if len(available) else 0.0,
        "hard_negative_fraction_examples_requested_available": float(np.mean(available >= config.hard_negative_count)) if len(available) else 0.0,
        "hard_negative_known_positive_rejections": float(rejected), "hard_negative_known_positive_rejection_rate": rejected / max(len(groups) * requested, 1),
        "hard_negative_average_v2_rank": float(all_ranks.mean()) if len(all_ranks) else float("nan"), "hard_negative_p50_v2_rank": float(np.percentile(all_ranks, 50)) if len(all_ranks) else float("nan"),
        "hard_negative_p95_v2_rank": float(np.percentile(all_ranks, 95)) if len(all_ranks) else float("nan"), "hard_negative_unique_items": float(len(np.unique(pool_values))),
        "hard_negative_pool_duplicate_rate": 0.0,
    }
    for name, values in buckets.items(): diagnostics[f"hard_negative_{name}_fraction"] = sum(int(item) in values for item in pool_values) / max(len(pool_values), 1)
    return pools, diagnostics


def _load_or_mine_hard_negatives(model: SimpleTwoTowerV2Model, data: PreparedData, config: SimpleTwoTowerV2Config, device: torch.device, teacher: Mapping[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Cache compact user/context pools; stale fingerprints are never reused."""
    root = _cache_root(config) / "hard_negatives"; root.mkdir(parents=True, exist_ok=True)
    fingerprint = _hard_negative_fingerprint(data, config, str(teacher["teacher_checkpoint_sha256"]))
    arrays_path, metadata_path = root / f"{fingerprint}.npz", root / f"{fingerprint}.json"
    if arrays_path.is_file() and metadata_path.is_file():
        started = time.perf_counter(); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            with np.load(arrays_path, allow_pickle=False) as values:
                keys, inverse, offsets, items = (values[name].copy() for name in ("user_context_keys", "training_group_index", "candidate_offsets", "candidate_item_indices"))
            if len(inverse) == len(data.train_user) and len(keys) == len(offsets) - 1:
                groups = [items[offsets[i]:offsets[i + 1]] for i in range(len(keys))]
                pools = [groups[int(index)] for index in inverse]
                assert_past_only_hard_negative_contract(data, data.train_user, data.train_item, pools)
                return pools, {**metadata["diagnostics"], "hard_negative_cache_hit": True, "hard_negative_cache_load_seconds": time.perf_counter() - started, "hard_negative_mining_wall_seconds": 0.0, "hard_negative_cache_path": str(arrays_path), "hard_negative_fingerprint": fingerprint}
        _log("[hard-negatives] cache metadata did not match; re-mining.")
    started = time.perf_counter(); pools, diagnostics = _mine_hard_negative_pools(model, data, config, device)
    # Store one pool per exact (user, encoded-context) key rather than one per
    # interaction. This preserves V2 semantics and is much smaller on repeat users.
    keys = np.column_stack((data.train_user, data.train_context)).astype(np.int64, copy=False)
    groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    group_pools = [pools[int(np.flatnonzero(inverse == index)[0])] for index in range(len(groups))]
    offsets = [0]; flat: list[int] = []
    for pool in group_pools: flat.extend(pool.tolist()); offsets.append(len(flat))
    np.savez_compressed(arrays_path, user_context_keys=groups, training_group_index=inverse.astype(np.int64), candidate_offsets=np.asarray(offsets, dtype=np.int64), candidate_item_indices=np.asarray(flat, dtype=np.int64))
    diagnostics = {**diagnostics, "hard_negative_cache_hit": False, "hard_negative_cache_load_seconds": 0.0, "hard_negative_mining_wall_seconds": time.perf_counter() - started, "hard_negative_cache_path": str(arrays_path), "hard_negative_fingerprint": fingerprint}
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "teacher_checkpoint_sha256": teacher["teacher_checkpoint_sha256"], "diagnostics": diagnostics}, indent=2, default=str), encoding="utf-8")
    return pools, diagnostics


def write_candidates(model: SimpleTwoTowerV2Model, data: PreparedData, config: SimpleTwoTowerV2Config, device: torch.device, variant: str) -> dict[str, float]:
    output = config.output_dir / "candidates" / f"{variant}_top{config.top_k}.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    _synchronize_cuda(device)
    index_start = time.perf_counter(); item_vectors = _embeddings(model, len(data.products), config.inference_batch_size, device, users=False); _synchronize_cuda(device); index = _build_index(item_vectors); index_seconds = time.perf_counter() - index_start
    _synchronize_cuda(device)
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
    _synchronize_cuda(device)
    elapsed = time.perf_counter() - retrieve_start
    return {"candidate_path": str(output), "index_build_seconds": index_seconds, "retrieval_wall_seconds": elapsed, "retrieval_users_per_second": len(data.users) / max(elapsed, 1e-9), "faiss_search_seconds_included": elapsed, "retrieval_batch_p50_seconds": float(np.percentile(latencies, 50)), "retrieval_batch_p95_seconds": float(np.percentile(latencies, 95)), "average_retained_candidates": retained / len(data.users), "users_below_top_k": short, "unique_catalogue_items": len(data.products)}


def evaluate_candidates(path: Path, truth: Mapping[str, set[str]], warm_truth: Mapping[str, set[str]], catalogue: set[str], *, top_k: int = 200) -> dict[str, float]:
    """Evaluate only ranks that were actually retrieved; zero-hit recall is 0.0."""
    cutoffs = tuple(cutoff for cutoff in CUT_OFFS if cutoff <= top_k)
    if not cutoffs: raise ValueError(f"top_k must be at least {min(CUT_OFFS)} to report recall")
    ranking_cutoff = min(100, top_k)
    sums = {f"overall_recall@{cutoff}": 0.0 for cutoff in cutoffs} | {f"warm_recall@{cutoff}": 0.0 for cutoff in cutoffs}
    ndcg = mrr = 0.0; recalled: set[str] = set(); seen_users: set[str] = set(); previous: str | None = None; last_rank = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user, item, rank = row["user_id"], row["candidate_ad_id"], int(row["rank"])
            if user != previous: previous, last_rank = user, 0
            if rank != last_rank + 1: raise AssertionError("Candidate rows must be contiguous user groups with ranks starting at 1")
            last_rank = rank; seen_users.add(user)
            if rank <= ranking_cutoff: recalled.add(item)
            target, warm = truth.get(user, set()), warm_truth.get(user, set())
            for cutoff in cutoffs:
                if rank <= cutoff and item in target: sums[f"overall_recall@{cutoff}"] += 1 / len(target)
                if rank <= cutoff and item in warm: sums[f"warm_recall@{cutoff}"] += 1 / len(warm)
    # Re-read grouped only for rank-sensitive MRR/NDCG, keeping the normal path streaming and bounded.
    per_user: dict[str, list[str]] = defaultdict(list)
    for chunk in pd.read_csv(path, usecols=["user_id", "candidate_ad_id"], chunksize=200_000):
        for user, item in chunk.itertuples(index=False, name=None): per_user[str(user)].append(str(item))
    for user, target in truth.items():
        candidates = per_user.get(user, [])[:ranking_cutoff]; gains = [int(item in target) for item in candidates]
        first = next((i + 1 for i, gain in enumerate(gains) if gain), None)
        mrr += 0.0 if first is None else 1 / first
        dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains)); ideal = sum(1 / math.log2(i + 2) for i in range(min(len(target), ranking_cutoff))); ndcg += dcg / ideal if ideal else 0.0
    users, warm_users = len(truth), sum(bool(values) for values in warm_truth.values())
    result = {key: value / (warm_users if key.startswith("warm_") else users) if (warm_users if key.startswith("warm_") else users) else float("nan") for key, value in sums.items()}
    result.update({f"hit_rate@{cutoff}": sum(1 for user, target in truth.items() if set(per_user.get(user, [])[:cutoff]) & target) / users if users else float("nan") for cutoff in cutoffs})
    result.update({f"ndcg@{ranking_cutoff}": ndcg / users if users else float("nan"), f"mrr@{ranking_cutoff}": mrr / users if users else float("nan"), "unique_recalled_items": len(recalled), f"catalog_coverage@{ranking_cutoff}": len(recalled) / max(len(catalogue), 1), "candidate_users": len(seen_users)})
    return result


def popularity_metrics(path: Path, truth: Mapping[str, set[str]], data: PreparedData, *, top_k: int) -> dict[str, float]:
    counts = np.bincount(data.train_item, minlength=len(data.products)); order = np.argsort(counts); groups = {"head": set(order[int(.8 * len(order)):]), "mid": set(order[int(.5 * len(order)):int(.8 * len(order))]), "tail": set(order[:int(.5 * len(order))])}
    if any(not values for values in groups.values()): raise RuntimeError("Popularity buckets must be non-empty")
    products = data.products; candidates: dict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(path, usecols=["user_id", "candidate_ad_id", "rank"], chunksize=200_000):
        for user, item, rank in chunk.itertuples(index=False, name=None):
            if int(rank) <= top_k: candidates[str(user)].add(str(item))
    result = {}
    for name, positions in groups.items():
        valid = {products[i] for i in positions}; denominator = sum(len(values & valid) for values in truth.values())
        if not denominator: raise RuntimeError(f"Popularity bucket {name} has no Future-A truth")
        numerator = sum(len(candidates[user] & values & valid) for user, values in truth.items()); result[f"{name}_recall@{top_k}"] = numerator / denominator
    return result


def run_ablation(config: SimpleTwoTowerV2Config) -> pd.DataFrame:
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")
    overall_started = time.perf_counter(); data = prepare_data(config); preprocess_seconds = time.perf_counter() - overall_started; config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(config.output_dir / "evaluation_users.txt", data.users, fmt="%s")
    rows: list[dict[str, Any]] = []
    for variant in config.variants:
        if variant not in {"v0_id_only", "v1_item_content", "v2_user_context_stats", "v3_in_batch_negatives", "v4_hard_negatives"}: raise ValueError(f"Unknown variant: {variant}")
        _log(f"[train] starting {variant}...")
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        hard_pools: list[np.ndarray] | None = None
        hard_metadata: dict[str, Any] = {}
        if variant == "v4_hard_negatives":
            # The teacher has the normal V2 sampled-negative objective.  Both
            # teacher training and mining have only PreparedData's Past arrays;
            # Future-A truth is passed solely to evaluation below.
            teacher, teacher_metadata = _load_or_train_v2_teacher(data, config, device)
            hard_pools, hard_metadata = _load_or_mine_hard_negatives(teacher, data, config, device, teacher_metadata)
            hard_metadata.update(teacher_metadata)
            torch.manual_seed(config.seed)  # deterministic V4 initialization independent of cache hit.
        model: Any = SimpleTwoTowerV2Model(data, config, variant)
        if config.use_torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        train = train_model(model, data, config, device, hard_pools=hard_pools); retrieval = write_candidates(model, data, config, device, variant)
        metrics = evaluate_candidates(Path(retrieval["candidate_path"]), data.truth, data.warm_truth, set(data.products), top_k=config.top_k)
        metrics.update(popularity_metrics(Path(retrieval["candidate_path"]), data.truth, data, top_k=config.top_k)); gpu = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        try:
            import psutil
            host_mb = psutil.Process(os.getpid()).memory_info().rss / 2**20
        except ImportError: host_mb = float("nan")
        rows.append({"model": variant, **data.metadata, **hard_metadata, **train, **retrieval, **metrics, "preprocessing_elapsed_seconds": preprocess_seconds, "end_to_end_elapsed_seconds": time.perf_counter() - overall_started, "peak_gpu_mb": gpu, "peak_host_ram_mb": host_mb})
    frame = pd.DataFrame(rows); metrics_path = config.output_dir / "metrics"; metrics_path.mkdir(parents=True, exist_ok=True)
    # A V4-only invocation must never overwrite the published V0--V3 table.
    suffix = "v4_hard_negatives" if "v4_hard_negatives" in config.variants else "ablation"
    frame.to_csv(metrics_path / f"simple_two_tower_v2_{suffix}.csv", index=False)
    metadata_name = f"simple_two_tower_v2_{suffix}_metadata.json"
    (metrics_path / metadata_name).write_text(json.dumps({"config": asdict(config) | {"past_path": str(config.past_path), "future_a_path": str(config.future_a_path), "output_dir": str(config.output_dir)}, "data": data.metadata, "contract": "Past-only fitting/training/features; hard-negative teacher and FAISS mining use Past arrays/catalogue only; Future-A is passed only to the fixed evaluator; Future-B untouched; warm-catalogue retrieval only."}, indent=2, default=str), encoding="utf-8")
    if "v4_hard_negatives" in config.variants:
        old_path = metrics_path / "simple_two_tower_v2_ablation.csv"
        baseline = pd.read_csv(old_path) if old_path.is_file() else pd.DataFrame()
        combined = pd.concat((baseline[~baseline.get("model", pd.Series(dtype=str)).eq("v4_hard_negatives")] if len(baseline) else baseline, frame), ignore_index=True, sort=False)
        # This is a new comparison artifact: the historical baseline CSV is
        # read-only, so published V0--V3 rows are never rewritten by V4.
        combined.to_csv(metrics_path / "simple_two_tower_v2_v0_to_v4_comparison.csv", index=False)
        if {"v2_user_context_stats", "v4_hard_negatives"}.issubset(set(combined.model)):
            v2, v4 = (combined.loc[combined.model.eq(name)].iloc[-1] for name in ("v2_user_context_stats", "v4_hard_negatives"))
            def delta(key: str) -> float: return float(v4[key] - v2[key])
            def finite_value(row: pd.Series, key: str) -> float:
                value = row.get(key, 0.0)
                return 0.0 if pd.isna(value) else float(value)
            mining_seconds = finite_value(v4, "hard_negative_mining_wall_seconds") + finite_value(v4, "teacher_training_wall_seconds")
            comparison = {"v4_vs_v2_warm_recall@100_absolute": delta("warm_recall@100"), "v4_vs_v2_warm_recall@100_relative": delta("warm_recall@100") / max(abs(float(v2["warm_recall@100"])), 1e-12), "v4_vs_v2_overall_recall@100": delta("overall_recall@100"), "v4_vs_v2_ndcg@100": delta("ndcg@100"), "v4_vs_v2_training_seconds": delta("training_wall_seconds"), "v4_teacher_and_mining_seconds": mining_seconds, "v4_vs_v2_training_plus_mining_seconds": delta("training_wall_seconds") + mining_seconds, "v4_vs_v2_peak_gpu_mb": delta("peak_gpu_mb")}
            (metrics_path / "simple_two_tower_v2_v4_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return frame
