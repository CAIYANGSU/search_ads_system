"""DCNv2 multi-task fine ranking for conversion probability and value.

This module ranks an offline expected-conversion-value proxy
(``pCVR * predicted_conversion_value``).  It does not claim to estimate CTR,
bid, eCPM, or online revenue because the source dataset contains clicks only.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from search_ads_system.common.config import resolve_path
from search_ads_system.ranking.dcnv2 import DCNv2MultiTask
from search_ads_system.ranking.fine_rank_dataset import (
    DEFAULT_BUCKET_SIZES, DENSE_FEATURES, SPARSE_FEATURES, FineRankDatasetSpec,
    FineRankFeatureStore, FineRankParquetDataset, build_or_reuse_cached_datasets,
    encode_feature_frame,
)
from search_ads_system.ranking.fine_rank_metrics import evaluate_fine_rank_predictions

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FineRankConfig:
    mode: str
    input_path: Path
    output_path: Path
    model_path: Path
    cache_dir: Path
    feature_source_path: Path
    train_label_path: Path
    validation_label_path: Path | None
    metrics_path: Path
    top_k: int = 20
    max_train_rows: int = 3_000_000
    chunk_size: int = 200_000
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    num_cross_layers: int = 3
    batch_size: int = 8192
    inference_batch_size: int = 32_768
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    value_loss_weight: float = 0.2
    value_log_clip_max: float = 20.0
    gradient_clip_norm: float = 5.0
    num_workers: int = 12
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    amp: bool = True
    device: str = "auto"
    train: bool = True
    early_stopping: bool = True
    patience: int = 2
    validation_fraction: float = 0.1
    random_seed: int = 2026
    oom_retries: int = 3
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES


def parse_fine_rank_config(raw_config: Mapping[str, Any], config_path: Path) -> FineRankConfig:
    paths = raw_config.get("paths")
    options = raw_config.get("fine_rank", {})
    if not isinstance(paths, Mapping) or not isinstance(options, Mapping):
        raise ValueError("Configuration must define paths and a fine_rank mapping")
    root = config_path.parent.resolve()
    mode = str(options.get("mode", "full")).lower()
    if mode not in {"full", "temporal"}:
        raise ValueError("fine_rank.mode must be full or temporal")
    seed = int(options.get("random_seed", raw_config.get("project", {}).get("seed", 2026)))
    buckets_raw = options.get("hash_buckets", {})
    if buckets_raw and not isinstance(buckets_raw, Mapping):
        raise ValueError("fine_rank.hash_buckets must be a mapping")
    bucket_sizes = tuple(int(buckets_raw.get(name, default)) for name, default in zip(SPARSE_FEATURES, DEFAULT_BUCKET_SIZES))
    if mode == "temporal":
        temporal = raw_config.get("temporal", {})
        if not isinstance(temporal, Mapping):
            raise ValueError("fine_rank temporal mode requires a temporal mapping")
        temporal_root = resolve_path(str(temporal.get("output_dir", "outputs/temporal")), root)
        input_path = temporal_root / "ranking" / "coarse_rank_topk.csv"
        output_path = temporal_root / "ranking" / "fine_rank_topk.csv"
        model_path = temporal_root / "models" / "fine_rank_dcnv2.pt"
        cache_dir = temporal_root / "ranking" / "fine_rank" / "train"
        feature_source = temporal_root / "split" / "past"
        train_labels = temporal_root / "split" / "future_a"
        validation_labels: Path | None = temporal_root / "split" / "future_b"
        metrics_path = temporal_root / "metrics" / "fine_rank_metrics.json"
    else:
        input_path = resolve_path(str(options.get("input_path", "outputs/ranking/coarse_rank_topk.csv")), root)
        output_path = resolve_path(str(options.get("output_path", "outputs/ranking/fine_rank_topk.csv")), root)
        model_path = resolve_path(str(options.get("model_path", "outputs/models/fine_rank_dcnv2.pt")), root)
        cache_dir = resolve_path(str(options.get("cache_dir", "outputs/ranking/fine_rank/train")), root)
        feature_source = resolve_path(str(options.get("interaction_path", paths.get("unified_data", "outputs/processed/criteo_unified"))), root)
        train_labels = feature_source
        validation_labels = None
        metrics_path = output_path.parent / "fine_rank_metrics.json"
    early = options.get("early_stopping", {})
    if not isinstance(early, Mapping):
        raise ValueError("fine_rank.early_stopping must be a mapping")
    config = FineRankConfig(
        mode=mode, input_path=input_path, output_path=output_path, model_path=model_path, cache_dir=cache_dir,
        feature_source_path=feature_source, train_label_path=train_labels, validation_label_path=validation_labels, metrics_path=metrics_path,
        top_k=int(options.get("top_k", 20)), max_train_rows=int(options.get("max_train_rows", 3_000_000)), chunk_size=int(options.get("chunk_size", 200_000)),
        embedding_dim=int(options.get("embedding_dim", 32)), hidden_dims=tuple(int(value) for value in options.get("hidden_dims", (256, 128, 64))), num_cross_layers=int(options.get("num_cross_layers", 3)),
        batch_size=int(options.get("batch_size", 8192)), inference_batch_size=int(options.get("inference_batch_size", options.get("batch_size", 8192) * 4)), epochs=int(options.get("epochs", 5)),
        learning_rate=float(options.get("learning_rate", 0.001)), weight_decay=float(options.get("weight_decay", 0.00001)), value_loss_weight=float(options.get("value_loss_weight", 0.2)), value_log_clip_max=float(options.get("value_log_clip_max", 20.0)), gradient_clip_norm=float(options.get("gradient_clip_norm", 5.0)),
        num_workers=int(options.get("num_workers", min(12, max(1, os.cpu_count() or 1)))), prefetch_factor=int(options.get("prefetch_factor", 4)), pin_memory=bool(options.get("pin_memory", True)), persistent_workers=bool(options.get("persistent_workers", True)),
        amp=bool(options.get("amp", True)), device=str(options.get("device", "auto")), train=bool(options.get("train", True)), early_stopping=bool(early.get("enabled", True)), patience=int(early.get("patience", 2)), validation_fraction=float(options.get("validation_fraction", 0.1)), random_seed=seed, oom_retries=int(options.get("oom_retries", 3)), bucket_sizes=bucket_sizes,
    )
    _validate_config(config)
    return config


def dataset_spec(config: FineRankConfig) -> FineRankDatasetSpec:
    return FineRankDatasetSpec(cache_dir=config.cache_dir, candidate_path=config.input_path, feature_source_path=config.feature_source_path, train_label_path=config.train_label_path, validation_label_path=config.validation_label_path, mode=config.mode, max_train_rows=config.max_train_rows, chunk_size=config.chunk_size, validation_fraction=config.validation_fraction, random_seed=config.random_seed, value_log_clip_max=config.value_log_clip_max, bucket_sizes=config.bucket_sizes)


def multitask_loss(logits: Tensor, predicted_log_value: Tensor, labels: Tensor, normalized_log_values: Tensor, value_mask: Tensor, value_loss_weight: float) -> tuple[Tensor, Tensor, Tensor]:
    """BCE plus masked Huber loss in train-split-normalized log-value space."""
    cvr_loss = F.binary_cross_entropy_with_logits(logits, labels)
    valid = value_mask > 0.5
    value_loss = F.smooth_l1_loss(predicted_log_value[valid], normalized_log_values[valid]) if valid.any() else logits.new_zeros(())
    return cvr_loss + value_loss_weight * value_loss, cvr_loss, value_loss


def build_dataset(config: FineRankConfig) -> dict[str, Any]:
    LOGGER.info("Fine-rank feature list dense=%s sparse=%s", list(DENSE_FEATURES), list(SPARSE_FEATURES))
    return build_or_reuse_cached_datasets(dataset_spec(config))


def train_fine_ranker(config: FineRankConfig, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(metadata or build_dataset(config))
    device = resolve_device(config.device)
    _log_device_config(config, device)
    actual_batch_size = config.batch_size
    for attempt in range(config.oom_retries + 1):
        try:
            return _train_once(config, metadata, device, actual_batch_size)
        except RuntimeError as error:
            if device.type != "cuda" or "out of memory" not in str(error).lower() or attempt >= config.oom_retries or actual_batch_size <= 1:
                raise
            next_batch = max(1, actual_batch_size // 2)
            LOGGER.warning("CUDA OOM at batch_size=%s; clearing cache and retrying with batch_size=%s (%s/%s)", actual_batch_size, next_batch, attempt + 1, config.oom_retries)
            torch.cuda.empty_cache()
            actual_batch_size = next_batch
    raise AssertionError("unreachable")


def _train_once(config: FineRankConfig, metadata: Mapping[str, Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    torch.manual_seed(config.random_seed)
    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    value_transform = _value_transform(metadata)
    train_loader = _loader(config.cache_dir, config, batch_size=batch_size, value_transform=value_transform, include_identifiers=False)
    has_validation = bool(metadata.get("validation_row_count", 0)) and any(dataset_spec(config).validation_dir.glob("part-*.parquet"))
    history: list[dict[str, Any]] = []
    best_metric = float("-inf"); best_epoch = 0; stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train(); totals = np.zeros(3, dtype=float); rows = 0; started = time.perf_counter(); batches = 0
        for batch in train_loader:
            dense, sparse, labels, values, masks = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits, predicted_log_value = model(dense, sparse)
                total, cvr, value = multitask_loss(logits, predicted_log_value, labels, values, masks, config.value_loss_weight)
            if not torch.isfinite(total):
                raise FloatingPointError("Fine-rank loss became non-finite before the optimizer update")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            _assert_finite_parameters(model)
            current_rows = len(labels); totals += np.asarray((float(total.detach()), float(cvr.detach()), float(value.detach()))) * current_rows; rows += current_rows; batches += 1
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {"epoch": epoch, "train_loss": float(totals[0] / rows), "cvr_loss": float(totals[1] / rows), "value_loss": float(totals[2] / rows), "rows_per_second": rows / elapsed if elapsed else 0.0, "batches_per_second": batches / elapsed if elapsed else 0.0, "elapsed_seconds": elapsed, "batch_size": batch_size, "gradient_norm_last_batch": float(gradient_norm)}
        if device.type == "cuda":
            record["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device)); torch.cuda.reset_peak_memory_stats(device)
        if has_validation:
            validation = evaluate_fine_ranker(model, config, split="validation", device=device, metadata=metadata)
            record["validation"] = validation
            monitor = validation["pcvr"]["pr_auc"]
            monitor = float(monitor) if monitor is not None else -float(validation["pcvr"]["logloss"] or 0.0)
        else:
            validation = {}; monitor = -record["train_loss"]
        history.append(record)
        LOGGER.info("Fine-rank epoch=%s loss=%.6f cvr=%.6f value=%.6f rows/s=%.1f validation_pr_auc=%s", epoch, record["train_loss"], record["cvr_loss"], record["value_loss"], record["rows_per_second"], validation.get("pcvr", {}).get("pr_auc"))
        if monitor > best_metric:
            best_metric, best_epoch, stale_epochs = monitor, epoch, 0
            save_checkpoint(model, optimizer, config, epoch, metadata, {"history": history, "best_validation": validation, "batch_size": batch_size})
        else:
            stale_epochs += 1
            if config.early_stopping and stale_epochs >= config.patience:
                LOGGER.info("Fine-rank early stopping at epoch %s (best epoch %s)", epoch, best_epoch)
                break
    return {"history": history, "best_epoch": best_epoch, "best_metric": best_metric, "checkpoint": str(config.model_path)}


def build_model(config: FineRankConfig) -> DCNv2MultiTask:
    return DCNv2MultiTask(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=config.bucket_sizes, embedding_dim=config.embedding_dim, hidden_dims=config.hidden_dims, num_cross_layers=config.num_cross_layers)


def save_checkpoint(model: DCNv2MultiTask, optimizer: AdamW, config: FineRankConfig, epoch: int, metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "config": _checkpoint_config(config), "vocab_hash_metadata": {"bucket_sizes": list(config.bucket_sizes), "hash": "blake2b"}, "feature_schema": {"dense": list(DENSE_FEATURES), "sparse": list(SPARSE_FEATURES)}, "dataset_metadata": dict(metadata), "metrics": dict(metrics)}, config.model_path)
    LOGGER.info("Saved fine-rank best checkpoint epoch=%s to %s", epoch, config.model_path)


def load_fine_ranker(config: FineRankConfig, device: torch.device | None = None) -> tuple[DCNv2MultiTask, dict[str, Any]]:
    target = device or resolve_device(config.device)
    checkpoint = torch.load(config.model_path, map_location=target, weights_only=False)
    schema = checkpoint.get("feature_schema", {})
    if schema.get("dense") != list(DENSE_FEATURES) or schema.get("sparse") != list(SPARSE_FEATURES):
        raise ValueError("Fine-rank checkpoint feature schema does not match the current leakage-safe schema")
    model = build_model(config).to(target); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model, checkpoint


def evaluate_fine_ranker(model: DCNv2MultiTask, config: FineRankConfig, *, split: str = "validation", device: torch.device | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    target = device or resolve_device(config.device)
    directory = config.cache_dir if split == "train" else dataset_spec(config).validation_dir
    if not any(directory.glob("part-*.parquet")):
        return {"pcvr": {"roc_auc": None, "pr_auc": None, "logloss": None}, "value": {"mae": None, "rmse": None, "log_value_mae": None, "positive_value_rows": 0}, "ranking": {}, "expected_value_comparison": {}, "rows": 0}
    transform = _value_transform(metadata or _load_cache_metadata(config))
    loader = _loader(directory, config, batch_size=config.inference_batch_size, value_transform=transform, include_identifiers=True, force_workers=0)
    model.eval()
    diagnostics = _PredictionDiagnostics()
    def batches() -> Iterator[dict[str, Any]]:
        with torch.no_grad():
            for batch in loader:
                dense, sparse, labels, values, masks = _to_device(batch, target)
                probability, predicted_log_value, predicted_value, _ = model.predict_with_log(dense, sparse, **_prediction_kwargs(transform))
                diagnostics.update(predicted_log_value.cpu().numpy(), predicted_value.cpu().numpy())
                yield {"label": labels.cpu().numpy(), "pcvr": probability.cpu().numpy(), "predicted_value": predicted_value.cpu().numpy(), "observed_value": batch["observed_value"].numpy(), "value_mask": masks.cpu().numpy(), "user_id": batch["user_id"], "candidate_ad_id": batch["candidate_ad_id"], "coarse_score": batch["coarse_score"].numpy()}
    result = evaluate_fine_rank_predictions(batches())
    result["prediction_diagnostics"] = diagnostics.summary()
    return result


def infer_fine_rank(config: FineRankConfig) -> dict[str, Any]:
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Fine-rank checkpoint not found: {config.model_path}")
    if not dataset_spec(config).index_path.is_file():
        build_dataset(config)
    device = resolve_device(config.device); model, checkpoint = load_fine_ranker(config, device); store = FineRankFeatureStore(dataset_spec(config).index_path)
    transform = _value_transform(checkpoint.get("dataset_metadata", {}))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    started = time.perf_counter(); written = 0; processed = 0; batches = 0; diagnostics = _PredictionDiagnostics()
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file); writer.writerow(("user_id", "candidate_ad_id", "pCVR", "predicted_conversion_value", "expected_value_score", "rank"))
            pending: list[tuple[str, pd.DataFrame]] = []; pending_rows = 0
            for user, group in _iter_candidate_groups(config.input_path, config.chunk_size):
                pending.append((user, group)); pending_rows += len(group)
                if pending_rows >= config.inference_batch_size:
                    written += _score_pending(pending, model, store, config, device, writer, transform, diagnostics); processed += pending_rows; batches += 1; pending = []; pending_rows = 0
            if pending:
                written += _score_pending(pending, model, store, config, device, writer, transform, diagnostics); processed += pending_rows; batches += 1
        temporary.replace(config.output_path)
    finally:
        store.close()
        if temporary.exists():
            temporary.unlink()
    elapsed = time.perf_counter() - started
    metrics = {"output_rows": written, "input_candidates": processed, "elapsed_seconds": elapsed, "candidates_per_second": processed / elapsed if elapsed else 0.0, "batches_per_second": batches / elapsed if elapsed else 0.0, "device": str(device), "batch_size": config.inference_batch_size, "prediction_diagnostics": diagnostics.summary()}
    LOGGER.info("Fine-rank inference wrote %s rows (%.1f candidates/s) to %s", written, metrics["candidates_per_second"], config.output_path)
    return metrics


def run_fine_rank(config: FineRankConfig, *, stage: str = "all") -> dict[str, Any]:
    if stage not in {"build_dataset", "train", "evaluate", "infer", "all"}:
        raise ValueError("stage must be build_dataset, train, evaluate, infer, or all")
    result: dict[str, Any] = {}
    if stage in {"build_dataset", "train", "all"}:
        result["dataset"] = build_dataset(config)
    if stage in {"train", "all"}:
        if config.train:
            result["training"] = train_fine_ranker(config, result.get("dataset"))
        else:
            LOGGER.info("fine_rank.train=false; loading existing checkpoint for evaluation/inference")
    if stage in {"evaluate", "all"}:
        model, _ = load_fine_ranker(config)
        result["evaluation"] = evaluate_fine_ranker(model, config)
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True); config.metrics_path.write_text(json.dumps(result["evaluation"], indent=2, sort_keys=True), encoding="utf-8")
    if stage in {"infer", "all"}:
        result["inference"] = infer_fine_rank(config)
    return result


def resolve_device(option: str) -> torch.device:
    if option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(option)
    if requested.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def _score_pending(groups: list[tuple[str, pd.DataFrame]], model: DCNv2MultiTask, store: FineRankFeatureStore, config: FineRankConfig, device: torch.device, writer: csv.writer, value_transform: Mapping[str, float], diagnostics: "_PredictionDiagnostics") -> int:
    frame = pd.concat([group for _, group in groups], ignore_index=True)
    encoded = encode_feature_frame(store.enrich(frame), bucket_sizes=config.bucket_sizes, random_seed=config.random_seed)
    dense = torch.as_tensor(encoded[[f"dense__{name}" for name in DENSE_FEATURES]].to_numpy(dtype=np.float32), device=device)
    sparse = torch.as_tensor(encoded[[f"sparse__{name}" for name in SPARSE_FEATURES]].to_numpy(dtype=np.int64), device=device)
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
        probability, predicted_log_value, value, expected = model.predict_with_log(dense, sparse, **_prediction_kwargs(value_transform))
    diagnostics.update(predicted_log_value.cpu().numpy(), value.cpu().numpy())
    encoded["pCVR"] = probability.cpu().numpy(); encoded["predicted_conversion_value"] = value.cpu().numpy(); encoded["expected_value_score"] = expected.cpu().numpy()
    offset = 0; written = 0
    for user, group in groups:
        scored = encoded.iloc[offset : offset + len(group)].copy(); offset += len(group)
        scored = scored.sort_values(["expected_value_score", "pCVR", "candidate_ad_id"], ascending=[False, False, True], kind="mergesort").iloc[: config.top_k]
        for rank, row in enumerate(scored.itertuples(index=False), start=1):
            writer.writerow((user, row.candidate_ad_id, float(row.pCVR), float(row.predicted_conversion_value), float(row.expected_value_score), rank)); written += 1
    return written


def _iter_candidate_groups(path: Path, chunk_size: int) -> Iterator[tuple[str, pd.DataFrame]]:
    current: str | None = None; rows: list[dict[str, Any]] = []; completed: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        required = {"user_id", "candidate_ad_id"}
        if not required.issubset(chunk):
            raise ValueError(f"Fine-rank input missing columns: {sorted(required - set(chunk))}")
        for row in chunk.to_dict("records"):
            user = str(row["user_id"]).strip(); candidate = str(row["candidate_ad_id"]).strip()
            if not user or not candidate or user.lower() == "nan" or candidate.lower() == "nan":
                continue
            if current is not None and user != current:
                if current in completed:
                    raise ValueError("Fine-rank input must keep each user's candidate rows contiguous")
                completed.add(current); yield current, pd.DataFrame(rows); rows = []
            current = user; row["user_id"] = user; row["candidate_ad_id"] = candidate; rows.append(row)
    if current is not None:
        if current in completed:
            raise ValueError("Fine-rank input must keep each user's candidate rows contiguous")
        yield current, pd.DataFrame(rows)


def _load_cache_metadata(config: FineRankConfig) -> dict[str, Any]:
    path = dataset_spec(config).metadata_path
    if not path.is_file():
        raise FileNotFoundError(f"Fine-rank cache metadata is required for value decoding: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _value_transform(metadata: Mapping[str, Any]) -> dict[str, float]:
    raw = metadata.get("value_transform")
    if not isinstance(raw, Mapping):
        raise ValueError("Fine-rank cache/checkpoint lacks normalized log-value transform metadata; rebuild the dataset")
    required = ("mean", "std", "prediction_log_min", "prediction_log_max")
    if any(name not in raw for name in required):
        raise ValueError("Fine-rank value transform metadata is incomplete; rebuild the dataset")
    transform = {name: float(raw[name]) for name in required}
    if not np.isfinite(list(transform.values())).all() or transform["std"] <= 0 or transform["prediction_log_max"] < transform["prediction_log_min"]:
        raise ValueError("Fine-rank value transform metadata is invalid")
    return transform


def _prediction_kwargs(transform: Mapping[str, float]) -> dict[str, float]:
    return {"value_mean": float(transform["mean"]), "value_std": float(transform["std"]), "prediction_log_min": float(transform["prediction_log_min"]), "prediction_log_max": float(transform["prediction_log_max"])}


class _PredictionDiagnostics:
    """Bounded prediction diagnostics; validation is exact at normal cache size."""

    def __init__(self, sample_limit: int = 1_000_000) -> None:
        self.sample_limit = sample_limit
        self.log_values: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.rows = 0
        self.nonfinite = 0

    def update(self, predicted_log_value: np.ndarray, predicted_value: np.ndarray) -> None:
        logs = np.asarray(predicted_log_value, dtype=np.float64)
        values = np.asarray(predicted_value, dtype=np.float64)
        self.rows += len(logs)
        self.nonfinite += int((~np.isfinite(logs)).sum() + (~np.isfinite(values)).sum())
        if self.nonfinite:
            raise FloatingPointError("Fine-rank prediction diagnostics found non-finite decoded values")
        current = sum(len(value) for value in self.values)
        remaining = self.sample_limit - current
        if remaining > 0:
            self.log_values.append(logs[:remaining].copy()); self.values.append(values[:remaining].copy())

    def summary(self) -> dict[str, Any]:
        logs = np.concatenate(self.log_values) if self.log_values else np.empty(0)
        values = np.concatenate(self.values) if self.values else np.empty(0)
        return {"rows": self.rows, "sampled_rows": len(values), "nonfinite_values": self.nonfinite, "predicted_log_value": _quantile_summary(logs), "predicted_value": _quantile_summary(values)}


def _quantile_summary(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"min": None, "median": None, "p95": None, "p99": None, "max": None}
    return {"min": float(values.min()), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def _assert_finite_parameters(model: DCNv2MultiTask) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Fine-rank optimizer produced non-finite parameter: {name}")


def _loader(directory: Path, config: FineRankConfig, *, batch_size: int, value_transform: Mapping[str, float], include_identifiers: bool, force_workers: int | None = None) -> DataLoader[Any]:
    workers = config.num_workers if force_workers is None else force_workers
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    kwargs: dict[str, Any] = {"batch_size": batch_size, "num_workers": workers, "pin_memory": config.pin_memory and resolve_device(config.device).type == "cuda"}
    if workers:
        kwargs.update(prefetch_factor=config.prefetch_factor, persistent_workers=config.persistent_workers)
    return DataLoader(FineRankParquetDataset(directory, value_transform=value_transform, include_identifiers=include_identifiers), **kwargs)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (batch["dense"].to(device, non_blocking=True), batch["sparse"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True), batch["value"].to(device, non_blocking=True), batch["value_mask"].to(device, non_blocking=True))


def _checkpoint_config(config: FineRankConfig) -> dict[str, Any]:
    result = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}


def _log_device_config(config: FineRankConfig, device: torch.device) -> None:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        LOGGER.info("Fine-rank device=%s CUDA=%s VRAM=%.1fGiB workers=%s AMP=%s", device, properties.name, properties.total_memory / 2**30, config.num_workers, config.amp)
    else:
        LOGGER.info("Fine-rank device=cpu workers=%s AMP disabled (CUDA unavailable)", config.num_workers)


def _validate_config(config: FineRankConfig) -> None:
    positive = (config.top_k, config.max_train_rows, config.chunk_size, config.embedding_dim, config.num_cross_layers, config.batch_size, config.inference_batch_size, config.epochs, config.num_workers, config.prefetch_factor)
    if any(value <= 0 for value in positive) or not 0 <= config.value_loss_weight or not 0 < config.validation_fraction < 1 or config.patience <= 0 or not np.isfinite(config.value_log_clip_max) or config.value_log_clip_max <= 0 or not np.isfinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0:
        raise ValueError("Fine-rank numeric configuration contains an invalid value")
    if len(config.bucket_sizes) != len(SPARSE_FEATURES) or any(value <= 1 for value in config.bucket_sizes):
        raise ValueError("fine_rank hash bucket configuration is invalid")
