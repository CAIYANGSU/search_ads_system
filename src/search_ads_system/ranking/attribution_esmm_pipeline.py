"""Streaming training/evaluation runner for Attribution-only ESMM baselines.

This module deliberately has no Future-B path.  Past is the only training
source and Future-A is the only development/evaluation source.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from search_ads_system.common.config import resolve_path
from search_ads_system.ranking.attribution_esmm import (
    CATEGORICAL_FEATURES,
    DENSE_FEATURES,
    AttributionESMM,
    AttributionSingleTask,
    esmm_loss,
    stable_hash_series,
    validate_feature_contract,
)

LOGGER = logging.getLogger(__name__)
MODEL_KINDS: tuple[str, ...] = ("single_ctr", "naive_cvr", "esmm")
_REQUIRED_COLUMNS = (*CATEGORICAL_FEATURES, "time_since_last_click", "click", "conversion", "click_and_conversion")


@dataclass(frozen=True)
class AttributionESMMConfig:
    enabled: bool
    seed: int
    device: str
    past_path: Path
    future_a_path: Path
    checkpoint_dir: Path
    metrics_dir: Path
    embedding_dim: int
    bucket_sizes: tuple[int, ...]
    shared_hidden_dims: tuple[int, ...]
    ctr_hidden_dims: tuple[int, ...]
    cvr_hidden_dims: tuple[int, ...]
    batch_size: int
    inference_batch_size: int
    io_chunk_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    lambda_ctcvr: float
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    mixed_precision: bool
    max_train_rows: int | None
    max_validation_rows: int | None
    early_stopping_patience: int
    sanity_max_train_rows: int
    sanity_max_validation_rows: int
    sanity_epochs: int


@dataclass(frozen=True)
class NumericNormalization:
    mean: float
    std: float
    valid_count: int
    missing_count: int


def parse_attribution_esmm_config(raw_config: Mapping[str, Any], config_path: Path) -> AttributionESMMConfig:
    """Parse an isolated ESMM configuration; Future-B is intentionally absent."""

    options = raw_config.get("attribution_esmm", {})
    preprocessing = raw_config.get("attribution_preprocessing", {})
    if not isinstance(options, Mapping) or not isinstance(preprocessing, Mapping):
        raise ValueError("Configuration must define attribution_esmm and attribution_preprocessing mappings")
    root = config_path.parent.resolve()
    temporal_root = resolve_path(str(preprocessing["temporal_output_dir"]), root)
    past_path = temporal_root / "split" / "past"
    future_a_path = temporal_root / "split" / "future_a"
    buckets = options.get("hash_buckets", {})
    if not isinstance(buckets, Mapping):
        raise ValueError("attribution_esmm.hash_buckets must be a mapping")
    bucket_sizes = tuple(
        int(buckets.get("user_id" if feature == "user_id" else "campaign_id" if feature == "campaign_id" else "categories"))
        for feature in CATEGORICAL_FEATURES
    )
    sanity = options.get("sanity", {})
    if not isinstance(sanity, Mapping):
        raise ValueError("attribution_esmm.sanity must be a mapping")
    config = AttributionESMMConfig(
        enabled=bool(options.get("enabled", True)),
        seed=int(options.get("seed", raw_config.get("project", {}).get("seed", 2026))),
        device=str(options.get("device", "auto")),
        past_path=past_path,
        future_a_path=future_a_path,
        checkpoint_dir=resolve_path(str(options.get("checkpoint_dir", "outputs/attribution/models")), root),
        metrics_dir=resolve_path(str(options.get("metrics_dir", "outputs/attribution/metrics")), root),
        embedding_dim=int(options.get("embedding_dim", 16)),
        bucket_sizes=bucket_sizes,
        shared_hidden_dims=tuple(int(value) for value in options.get("shared_hidden_dims", (256, 128))),
        ctr_hidden_dims=tuple(int(value) for value in options.get("ctr_hidden_dims", (128, 64))),
        cvr_hidden_dims=tuple(int(value) for value in options.get("cvr_hidden_dims", (128, 64))),
        batch_size=int(options.get("batch_size", 8192)),
        inference_batch_size=int(options.get("inference_batch_size", 32768)),
        io_chunk_size=int(options.get("io_chunk_size", 200000)),
        epochs=int(options.get("epochs", 3)),
        learning_rate=float(options.get("learning_rate", 0.001)),
        weight_decay=float(options.get("weight_decay", 1e-5)),
        lambda_ctcvr=float(options.get("lambda_ctcvr", 1.0)),
        num_workers=int(options.get("num_workers", min(8, max(0, os.cpu_count() or 0)))),
        pin_memory=bool(options.get("pin_memory", True)),
        persistent_workers=bool(options.get("persistent_workers", True)),
        mixed_precision=bool(options.get("mixed_precision", True)),
        max_train_rows=_optional_positive_int(options.get("max_train_rows")),
        max_validation_rows=_optional_positive_int(options.get("max_validation_rows")),
        early_stopping_patience=int(options.get("early_stopping_patience", 1)),
        sanity_max_train_rows=int(sanity.get("max_train_rows", 200000)),
        sanity_max_validation_rows=int(sanity.get("max_validation_rows", 100000)),
        sanity_epochs=int(sanity.get("epochs", 1)),
    )
    _validate_config(config)
    return config


def sanity_config(config: AttributionESMMConfig) -> AttributionESMMConfig:
    """Return a bounded, one-epoch config without changing formal artifacts."""

    return replace(
        config,
        max_train_rows=config.sanity_max_train_rows,
        max_validation_rows=config.sanity_max_validation_rows,
        epochs=config.sanity_epochs,
    )


class AttributionBatchDataset(IterableDataset[dict[str, Tensor]]):
    """CSV-part streaming dataset that produces already-encoded mini-batches."""

    def __init__(
        self,
        directory: Path,
        *,
        bucket_sizes: Sequence[int],
        normalization: NumericNormalization,
        batch_size: int,
        io_chunk_size: int,
        max_rows: int | None,
        clicked_only: bool = False,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.parts = sorted(directory.glob("part-*.csv"))
        if not self.parts:
            raise FileNotFoundError(f"No Attribution CSV parts found in {directory}")
        self.bucket_sizes = tuple(bucket_sizes)
        self.normalization = normalization
        self.batch_size = batch_size
        self.io_chunk_size = io_chunk_size
        self.max_rows = max_rows
        self.clicked_only = clicked_only
        self.part_counts = _part_counts(self.parts)
        self.part_offsets = _offsets_from_counts(self.part_counts)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        for part_index, (part, offset, row_count) in enumerate(zip(self.parts, self.part_offsets, self.part_counts)):
            if part_index % worker_count != worker_id:
                continue
            if self.max_rows is not None and offset >= self.max_rows:
                continue
            allowed = row_count if self.max_rows is None else min(row_count, self.max_rows - offset)
            consumed = 0
            for frame in pd.read_csv(part, usecols=list(_REQUIRED_COLUMNS), chunksize=self.io_chunk_size, low_memory=False):
                if consumed >= allowed:
                    break
                frame = frame.iloc[: allowed - consumed]
                consumed += len(frame)
                if self.clicked_only:
                    frame = frame.loc[pd.to_numeric(frame["click"], errors="raise").eq(1)]
                if frame.empty:
                    continue
                for start in range(0, len(frame), self.batch_size):
                    yield _encode_frame(frame.iloc[start:start + self.batch_size], self.bucket_sizes, self.normalization)


def fit_past_normalization(config: AttributionESMMConfig) -> NumericNormalization:
    """Fit log1p time-since-last-click moments from Past only."""

    count = 0
    total = 0.0
    squared_total = 0.0
    missing = 0
    consumed = 0
    for part in sorted(config.past_path.glob("part-*.csv")):
        if config.max_train_rows is not None and consumed >= config.max_train_rows:
            break
        for frame in pd.read_csv(part, usecols=["time_since_last_click"], chunksize=config.io_chunk_size, low_memory=False):
            if config.max_train_rows is not None:
                remaining = config.max_train_rows - consumed
                if remaining <= 0:
                    break
                frame = frame.iloc[:remaining]
            consumed += len(frame)
            raw = pd.to_numeric(frame["time_since_last_click"], errors="coerce")
            valid = raw.notna() & raw.ge(0)
            missing += int((~valid).sum())
            values = np.log1p(raw.loc[valid].to_numpy(dtype=np.float64))
            count += len(values)
            total += float(values.sum())
            squared_total += float(np.square(values).sum())
    if count == 0:
        raise ValueError("Past contains no valid time_since_last_click values for normalization")
    mean = total / count
    variance = max(squared_total / count - mean * mean, 0.0)
    return NumericNormalization(mean=mean, std=max(math.sqrt(variance), 1e-6), valid_count=count, missing_count=missing)


def train_all_models(config: AttributionESMMConfig, *, artifact_suffix: str = "") -> dict[str, Any]:
    """Train CTR, clicked-only CVR, and ESMM using Past and Future-A only."""

    _set_seed(config.seed)
    device = resolve_device(config.device)
    normalization = fit_past_normalization(config)
    data_summary = {
        "past": _label_summary(config.past_path, config.io_chunk_size, config.max_train_rows),
        "future_a": _label_summary(config.future_a_path, config.io_chunk_size, config.max_validation_rows),
    }
    results: dict[str, Any] = {}
    started = time.perf_counter()
    for kind in MODEL_KINDS:
        results[kind] = train_model(config, kind, normalization, device, artifact_suffix=artifact_suffix)
    return {
        "device": str(device),
        "normalization": asdict(normalization),
        "data_summary": data_summary,
        "models": results,
        "total_elapsed_seconds": time.perf_counter() - started,
        "future_b_read_for_model_selection": False,
    }


def train_model(
    config: AttributionESMMConfig,
    kind: str,
    normalization: NumericNormalization,
    device: torch.device,
    *,
    artifact_suffix: str = "",
) -> dict[str, Any]:
    """Train one model and checkpoint the best Future-A development result."""

    model = build_model(config, kind).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    train_loader = make_loader(
        config.past_path, config, normalization, batch_size=config.batch_size, max_rows=config.max_train_rows,
        clicked_only=kind == "naive_cvr",
    )
    checkpoint_path = model_checkpoint_path(config, kind, artifact_suffix)
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"total": 0.0, "ctr": 0.0, "ctcvr": 0.0}
        rows = 0
        batches = 0
        started = time.perf_counter()
        for batch in train_loader:
            sparse, dense, click, conversion, ctcvr = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.mixed_precision and device.type == "cuda"):
                if kind == "single_ctr":
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(sparse, dense), click)
                    losses = {"total": loss, "ctr": loss, "ctcvr": loss.new_zeros(())}
                elif kind == "naive_cvr":
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(sparse, dense), conversion)
                    losses = {"total": loss, "ctr": loss.new_zeros(()), "ctcvr": loss.new_zeros(())}
                else:
                    losses = esmm_loss(model(sparse, dense), click, ctcvr, config.lambda_ctcvr)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"{kind} loss became non-finite")
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            _assert_finite_parameters(model)
            current_rows = len(click)
            rows += current_rows
            batches += 1
            for name in totals:
                totals[name] += float(losses[name].detach()) * current_rows
        if rows == 0:
            raise ValueError(f"{kind} received zero Past training rows")
        elapsed = time.perf_counter() - started
        validation = evaluate_model(model, config, kind, normalization, device)
        score = _selection_score(validation, kind)
        record = {
            "epoch": epoch,
            "train_rows": rows,
            "batches": batches,
            "ctr_loss": totals["ctr"] / rows,
            "ctcvr_loss": totals["ctcvr"] / rows,
            "total_loss": totals["total"] / rows,
            "elapsed_seconds": elapsed,
            "rows_per_second": rows / elapsed if elapsed else 0.0,
            "validation": validation,
        }
        history.append(record)
        LOGGER.info("Attribution %s epoch=%s device=%s rows=%s batches=%s ctr_loss=%.6f ctcvr_loss=%.6f total_loss=%.6f rows/s=%.1f", kind, epoch, device, rows, batches, record["ctr_loss"], record["ctcvr_loss"], record["total_loss"], record["rows_per_second"])
        if score > best_score:
            best_score, stale = score, 0
            _save_checkpoint(checkpoint_path, model, optimizer, config, kind, normalization, epoch, history)
        else:
            stale += 1
            if stale > config.early_stopping_patience:
                LOGGER.info("Attribution %s stopped early after epoch=%s", kind, epoch)
                break
    return {"checkpoint_path": str(checkpoint_path), "history": history, "best_selection_score": best_score}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    config: AttributionESMMConfig,
    kind: str,
    normalization: NumericNormalization,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate exclusively on Future-A; Future-B is never opened or referenced."""

    model.eval()
    loader = make_loader(
        config.future_a_path, config, normalization, batch_size=config.inference_batch_size,
        max_rows=config.max_validation_rows, clicked_only=False,
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in ("click", "conversion", "ctcvr", "pctr", "pcvr", "pctcvr")}
    max_consistency_error = 0.0
    for batch in loader:
        sparse, dense, click, conversion, ctcvr = _to_device(batch, device)
        if kind == "single_ctr":
            pctr = torch.sigmoid(model(sparse, dense)); pcvr = pctcvr = None
        elif kind == "naive_cvr":
            pcvr = torch.sigmoid(model(sparse, dense)); pctr = pctcvr = None
        else:
            outputs = model(sparse, dense)
            pctr, pcvr, pctcvr = outputs["pctr"], outputs["pcvr"], outputs["pctcvr"]
            max_consistency_error = max(max_consistency_error, float((pctcvr - pctr * pcvr).abs().max().cpu()))
        for name, value in (("click", click), ("conversion", conversion), ("ctcvr", ctcvr)):
            collected[name].append(value.cpu().numpy())
        if pctr is not None:
            collected["pctr"].append(pctr.cpu().numpy())
        if pcvr is not None:
            collected["pcvr"].append(pcvr.cpu().numpy())
        if pctcvr is not None:
            collected["pctcvr"].append(pctcvr.cpu().numpy())
    values = {name: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32) for name, parts in collected.items()}
    result: dict[str, Any] = {"validation_source": "future_a_only", "future_b_read_for_model_selection": False}
    if kind in {"single_ctr", "esmm"}:
        result["ctr"] = _binary_metrics(values["click"], values["pctr"])
    if kind == "esmm":
        result["ctcvr"] = _binary_metrics(values["ctcvr"], values["pctcvr"])
        clicked = values["click"] == 1
        result["cvr_clicked_subset"] = _binary_metrics(values["conversion"][clicked], values["pcvr"][clicked])
        result["consistency"] = {"max_abs_error": max_consistency_error, "passed": max_consistency_error <= 1e-7}
    if kind == "naive_cvr":
        clicked = values["click"] == 1
        result["cvr_clicked_subset"] = _binary_metrics(values["conversion"][clicked], values["pcvr"][clicked])
    return result


def evaluate_checkpoints(config: AttributionESMMConfig, *, artifact_suffix: str = "") -> dict[str, Any]:
    """Load saved checkpoints and produce a Future-A-only report payload."""

    device = resolve_device(config.device)
    normalization = fit_past_normalization(config)
    data_summary = {
        "past": _label_summary(config.past_path, config.io_chunk_size, config.max_train_rows),
        "future_a": _label_summary(config.future_a_path, config.io_chunk_size, config.max_validation_rows),
    }
    metrics: dict[str, Any] = {}
    architecture: dict[str, Any] = {}
    for kind in MODEL_KINDS:
        checkpoint = torch.load(model_checkpoint_path(config, kind, artifact_suffix), map_location=device, weights_only=False)
        model = build_model(config, kind).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        metrics[kind] = evaluate_model(model, config, kind, normalization, device)
        architecture[kind] = {"parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    return {
        "dataset_contract": {"train": str(config.past_path), "model_selection_and_evaluation": str(config.future_a_path), "future_b_read_for_model_selection": False},
        "feature_contract": {"categorical": list(CATEGORICAL_FEATURES), "dense": list(DENSE_FEATURES), "timestamp_used_as_model_feature": False},
        "normalization": asdict(normalization),
        "data_summary": data_summary,
        "architecture": {"embedding_dim": config.embedding_dim, "bucket_sizes": dict(zip(CATEGORICAL_FEATURES, config.bucket_sizes)), "shared_hidden_dims": list(config.shared_hidden_dims), "ctr_hidden_dims": list(config.ctr_hidden_dims), "cvr_hidden_dims": list(config.cvr_hidden_dims), "models": architecture},
        "metrics": metrics,
        "leakage_check": "passed",
        "future_b_read_for_model_selection": False,
        "caveat": "Future-A is a development evaluation set. Future-B was not read and remains the final holdout.",
        "device": str(device),
    }


def write_metrics_report(config: AttributionESMMConfig, report: Mapping[str, Any], *, artifact_suffix: str = "") -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown reports for formal or sanity runs."""

    suffix = "_sanity" if artifact_suffix else ""
    json_path = config.metrics_dir / f"esmm_metrics{suffix}.json"
    markdown_path = config.metrics_dir / f"esmm_metrics{suffix}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = ["# Attribution ESMM Metrics", "", "Future-A is the development set; Future-B was not read.", ""]
    for kind, metric_groups in report["metrics"].items():
        lines.extend((f"## {kind}", ""))
        for name, metrics in metric_groups.items():
            if isinstance(metrics, Mapping) and "roc_auc" in metrics:
                lines.append(f"- {name}: ROC-AUC={metrics['roc_auc']}, PR-AUC={metrics['pr_auc']}, LogLoss={metrics['logloss']}, prediction_mean={metrics['prediction_mean']}, label_mean={metrics['label_mean']}, positives={metrics['positive_count']}, negatives={metrics['negative_count']}")
            elif name == "consistency":
                lines.append(f"- ESMM consistency max abs error: {metrics['max_abs_error']} (passed={metrics['passed']})")
        lines.append("")
    lines.extend(("## Contract", "", "- future_b_read_for_model_selection: false", "- leakage_check: passed", "- `cost`, `cpo`, labels, attribution outcomes, and timestamp are not model inputs.", ""))
    _atomic_write(markdown_path, "\n".join(lines))
    return json_path, markdown_path


def build_model(config: AttributionESMMConfig, kind: str) -> nn.Module:
    if kind == "single_ctr":
        return AttributionSingleTask(config.bucket_sizes, config.embedding_dim, config.shared_hidden_dims, config.ctr_hidden_dims)
    if kind == "naive_cvr":
        return AttributionSingleTask(config.bucket_sizes, config.embedding_dim, config.shared_hidden_dims, config.cvr_hidden_dims)
    if kind == "esmm":
        return AttributionESMM(config.bucket_sizes, config.embedding_dim, config.shared_hidden_dims, config.ctr_hidden_dims, config.cvr_hidden_dims)
    raise ValueError(f"Unknown Attribution model kind: {kind}")


def make_loader(directory: Path, config: AttributionESMMConfig, normalization: NumericNormalization, *, batch_size: int, max_rows: int | None, clicked_only: bool) -> DataLoader:
    dataset = AttributionBatchDataset(directory, bucket_sizes=config.bucket_sizes, normalization=normalization, batch_size=batch_size, io_chunk_size=config.io_chunk_size, max_rows=max_rows, clicked_only=clicked_only)
    return DataLoader(dataset, batch_size=None, num_workers=config.num_workers, pin_memory=config.pin_memory and resolve_device(config.device).type == "cuda", persistent_workers=config.persistent_workers and config.num_workers > 0)


def model_checkpoint_path(config: AttributionESMMConfig, kind: str, artifact_suffix: str = "") -> Path:
    suffix = "_sanity" if artifact_suffix else ""
    return config.checkpoint_dir / f"{kind}{suffix}.pt"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Attribution ESMM but is unavailable")
    return device


def _encode_frame(frame: pd.DataFrame, bucket_sizes: Sequence[int], normalization: NumericNormalization) -> dict[str, Tensor]:
    sparse = np.column_stack([stable_hash_series(frame[name], bucket) for name, bucket in zip(CATEGORICAL_FEATURES, bucket_sizes)])
    raw_time = pd.to_numeric(frame["time_since_last_click"], errors="coerce")
    valid = raw_time.notna() & raw_time.ge(0)
    transformed = np.zeros(len(frame), dtype=np.float32)
    transformed[valid.to_numpy()] = ((np.log1p(raw_time.loc[valid].to_numpy(dtype=np.float64)) - normalization.mean) / normalization.std).astype(np.float32)
    dense = np.column_stack((transformed, (~valid).to_numpy(dtype=np.float32)))
    click = pd.to_numeric(frame["click"], errors="raise").to_numpy(dtype=np.float32)
    conversion = pd.to_numeric(frame["conversion"], errors="raise").to_numpy(dtype=np.float32)
    persisted_ctcvr = pd.to_numeric(frame["click_and_conversion"], errors="raise").to_numpy(dtype=np.float32)
    derived_ctcvr = click * conversion
    if not np.array_equal(persisted_ctcvr, derived_ctcvr):
        raise ValueError("Attribution click_and_conversion must equal click AND conversion")
    return {
        "sparse": torch.from_numpy(sparse.astype(np.int64, copy=False)),
        "dense": torch.from_numpy(dense.astype(np.float32, copy=False)),
        "click": torch.from_numpy(click),
        "conversion": torch.from_numpy(conversion),
        "ctcvr": torch.from_numpy(derived_ctcvr),
    }


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(batch[name].to(device, non_blocking=True) for name in ("sparse", "dense", "click", "conversion", "ctcvr"))  # type: ignore[return-value]


def _part_counts(parts: Sequence[Path]) -> tuple[int, ...]:
    return tuple(_count_rows(part) for part in parts)


def _part_offsets(parts: Sequence[Path]) -> tuple[int, ...]:
    return _offsets_from_counts(_part_counts(parts))


def _offsets_from_counts(counts: Sequence[int]) -> tuple[int, ...]:
    offsets: list[int] = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += count
    return tuple(offsets)


def _count_rows(path: Path) -> int:
    with path.open("rb") as file:
        return max(sum(block.count(b"\n") for block in iter(lambda: file.read(1024 * 1024), b"")) - 1, 0)


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if len(labels) != len(predictions) or len(labels) == 0:
        raise ValueError("Metric labels and predictions must be non-empty and aligned")
    if not np.isfinite(predictions).all() or (predictions < 0.0).any() or (predictions > 1.0).any():
        raise FloatingPointError("Predictions must be finite probabilities in [0, 1]")
    positives = int(labels.sum())
    result: dict[str, Any] = {"rows": int(len(labels)), "positive_count": positives, "negative_count": int(len(labels) - positives), "label_mean": float(labels.mean()), "prediction_mean": float(predictions.mean()), "logloss": float(log_loss(labels, predictions, labels=[0, 1]))}
    if positives in {0, len(labels)}:
        result.update({"roc_auc": None, "pr_auc": None})
    else:
        result.update({"roc_auc": float(roc_auc_score(labels, predictions)), "pr_auc": float(average_precision_score(labels, predictions))})
    return result


def _label_summary(directory: Path, chunk_size: int, max_rows: int | None) -> dict[str, Any]:
    """Small streaming summary used in reports; it opens only Past or Future-A."""

    rows = click = conversion = ctcvr = 0
    for part in sorted(directory.glob("part-*.csv")):
        if max_rows is not None and rows >= max_rows:
            break
        for frame in pd.read_csv(part, usecols=["click", "conversion", "click_and_conversion"], chunksize=chunk_size, low_memory=False):
            if max_rows is not None:
                frame = frame.iloc[:max_rows - rows]
            if frame.empty:
                break
            rows += len(frame)
            click += int(pd.to_numeric(frame["click"], errors="raise").sum())
            conversion += int(pd.to_numeric(frame["conversion"], errors="raise").sum())
            ctcvr += int(pd.to_numeric(frame["click_and_conversion"], errors="raise").sum())
    rate = lambda value: value / rows if rows else 0.0
    return {"rows": rows, "click_positives": click, "conversion_positives": conversion, "ctcvr_positives": ctcvr, "ctr_rate": rate(click), "conversion_rate": rate(conversion), "ctcvr_rate": rate(ctcvr)}


def _selection_score(validation: Mapping[str, Any], kind: str) -> float:
    key = "ctr" if kind == "single_ctr" else "cvr_clicked_subset" if kind == "naive_cvr" else "ctcvr"
    metrics = validation[key]
    return float(metrics["pr_auc"]) if metrics["pr_auc"] is not None else -float(metrics["logloss"])


def _save_checkpoint(path: Path, model: nn.Module, optimizer: AdamW, config: AttributionESMMConfig, kind: str, normalization: NumericNormalization, epoch: int, history: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "kind": kind, "epoch": epoch, "normalization": asdict(normalization), "config": {"embedding_dim": config.embedding_dim, "bucket_sizes": config.bucket_sizes, "shared_hidden_dims": config.shared_hidden_dims, "ctr_hidden_dims": config.ctr_hidden_dims, "cvr_hidden_dims": config.cvr_hidden_dims}, "history": list(history), "future_b_read_for_model_selection": False}, temporary)
    temporary.replace(path)


def _validate_config(config: AttributionESMMConfig) -> None:
    validate_feature_contract((*CATEGORICAL_FEATURES, "time_since_last_click"))
    if not config.enabled:
        raise ValueError("attribution_esmm.enabled is false")
    if any(size <= 1 for size in config.bucket_sizes) or len(config.bucket_sizes) != len(CATEGORICAL_FEATURES):
        raise ValueError("Attribution ESMM hash buckets must be positive for every categorical feature")
    if min(config.batch_size, config.inference_batch_size, config.io_chunk_size, config.epochs) <= 0:
        raise ValueError("Attribution ESMM batch sizes, io_chunk_size, and epochs must be positive")
    if config.lambda_ctcvr < 0.0 or config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("Attribution ESMM optimization settings are invalid")
    if config.num_workers < 0:
        raise ValueError("Attribution ESMM num_workers must be non-negative")


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("row limits must be positive or null")
    return parsed


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_finite_parameters(model: nn.Module) -> None:
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Attribution model parameters became non-finite")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
