"""Strict-temporal Search Conversion multi-task fine ranking.

This is intentionally separate from the legacy candidate fine-rank experiment
and from Attribution ESMM.  Each source row is an observed *clicked*
interaction: CVR means ``P(conversion | clicked interaction)`` and value means
``E[value | conversion, clicked interaction]``.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from search_ads_system.common.config import resolve_path
from search_ads_system.data.storage import iter_csv_parts
from search_ads_system.ranking.dcnv2 import DCNv2MultiTask
from search_ads_system.ranking.deepfm import DeepFMMultiTask
from search_ads_system.ranking.fine_rank_dataset import (
    DEFAULT_BUCKET_SIZES, DENSE_FEATURES, LEAKAGE_COLUMNS, SPARSE_FEATURES,
    assert_no_fine_rank_leakage, encode_feature_frame,
)


TARGET_COLUMNS = ("conversion_label", "has_conversion_value", "conversion_value_eur")
SEQUENCE_FEATURES: tuple[str, ...] = ()


@dataclass(frozen=True)
class FineRankMultiTaskConfig:
    past_path: Path
    future_a_path: Path
    future_b_path: Path
    model_dir: Path
    metrics_dir: Path
    batch_size: int = 8192
    epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    amp: bool = True
    patience: int = 2
    lambda_cvr: float = 1.0
    lambda_value: float = 0.2
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    cross_layers: int = 3
    chunk_size: int = 200_000
    num_workers: int = 0
    device: str = "auto"
    seed: int = 2026
    max_train_rows: int | None = None
    max_validation_rows: int | None = None
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES


def parse_fine_rank_multitask_config(raw: Mapping[str, Any], config_path: Path, *, stage: str) -> FineRankMultiTaskConfig:
    root = config_path.parent.resolve()
    temporal = raw.get("temporal", {})
    options = raw.get("fine_rank_multitask", {})
    if not isinstance(temporal, Mapping) or not isinstance(options, Mapping):
        raise ValueError("config requires temporal and fine_rank_multitask mappings")
    temporal_root = resolve_path(str(temporal.get("output_dir", "outputs/temporal")), root)
    effective = {**options, **(options.get("sanity", {}) if stage == "sanity" else {})}
    buckets = effective.get("hash_buckets", {})
    if buckets and not isinstance(buckets, Mapping):
        raise ValueError("fine_rank_multitask.hash_buckets must be a mapping")
    cfg = FineRankMultiTaskConfig(
        past_path=temporal_root / "split" / "past",
        future_a_path=temporal_root / "split" / "future_a",
        future_b_path=temporal_root / "split" / "future_b",
        model_dir=resolve_path(str(effective.get("model_dir", "outputs/fine_rank/models")), root),
        metrics_dir=resolve_path(str(effective.get("metrics_dir", "outputs/fine_rank/metrics")), root),
        batch_size=int(effective.get("batch_size", 8192)), epochs=int(effective.get("epochs", 3)),
        learning_rate=float(effective.get("learning_rate", 5e-4)), weight_decay=float(effective.get("weight_decay", 1e-5)),
        amp=bool(effective.get("amp", True)), patience=int(effective.get("patience", 2)),
        lambda_cvr=float(effective.get("lambda_cvr", 1.0)), lambda_value=float(effective.get("lambda_value", 0.2)),
        embedding_dim=int(effective.get("embedding_dim", 32)), hidden_dims=tuple(int(v) for v in effective.get("hidden_dims", (256, 128, 64))),
        cross_layers=int(effective.get("cross_layers", 3)), chunk_size=int(effective.get("chunk_size", temporal.get("chunk_size", 200_000))),
        num_workers=int(effective.get("num_workers", 0)), device=str(effective.get("device", "auto")),
        seed=int(effective.get("seed", raw.get("project", {}).get("seed", 2026))),
        max_train_rows=_optional_positive_int(effective.get("max_train_rows")),
        max_validation_rows=_optional_positive_int(effective.get("max_validation_rows")),
        bucket_sizes=tuple(int(buckets.get(name, default)) for name, default in zip(SPARSE_FEATURES, DEFAULT_BUCKET_SIZES)),
    )
    if min(cfg.batch_size, cfg.epochs, cfg.embedding_dim, cfg.cross_layers, cfg.chunk_size) <= 0 or cfg.patience < 0 or cfg.lambda_cvr < 0 or cfg.lambda_value < 0:
        raise ValueError("invalid fine_rank_multitask numeric configuration")
    return cfg


def feature_contract() -> dict[str, Any]:
    """The serving-time feature allowlist for the clicked-interaction model."""
    assert_no_fine_rank_leakage([*DENSE_FEATURES, *SPARSE_FEATURES])
    return {
        "categorical_features": list(SPARSE_FEATURES), "dense_features": list(DENSE_FEATURES),
        "sequence_features": list(SEQUENCE_FEATURES), "target_columns": list(TARGET_COLUMNS),
        "excluded_leakage_features": sorted(set(LEAKAGE_COLUMNS) | {"conversion_delay_seconds", "time_delay_for_conversion", "conversion_timestamp", "post_conversion"}),
        "definition": "Features available at clicked-interaction scoring time; no conversion outcome or post-conversion field is an input.",
    }


class ClickInteractionDataset(IterableDataset[dict[str, Tensor]]):
    """Chunked CSV reader which never creates pseudo impression negatives."""

    def __init__(self, path: Path, config: FineRankMultiTaskConfig, *, max_rows: int | None, include_identifiers: bool = False) -> None:
        self.path, self.config, self.max_rows, self.include_identifiers = path, config, max_rows, include_identifiers

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        info = get_worker_info()
        parts = sorted(self.path.glob("part-*.csv"))
        if not parts:
            raise FileNotFoundError(f"No click-interaction parts at {self.path}")
        parts = parts if info is None else parts[info.id::info.num_workers]
        emitted = 0
        # Each worker receives a bounded quota. This preserves bounded-memory
        # sanity runs without every worker independently reading the full cap.
        worker_limit = self.max_rows
        if info is not None and self.max_rows is not None:
            worker_limit = math.ceil(self.max_rows / info.num_workers)
        for part in parts:
            for frame in pd.read_csv(part, chunksize=self.config.chunk_size, low_memory=False):
                required = {"user_id", "product_id", "conversion_label"}
                if missing := required - set(frame.columns):
                    raise ValueError(f"Search Conversion source missing {sorted(missing)}")
                prepared = frame.copy()
                prepared["candidate_ad_id"] = prepared["product_id"]
                label = pd.to_numeric(prepared["conversion_label"], errors="coerce")
                prepared = prepared.loc[label.isin((0, 1)) & prepared.user_id.notna() & prepared.candidate_ad_id.notna()].copy()
                if prepared.empty:
                    continue
                prepared["conversion_label"] = pd.to_numeric(prepared["conversion_label"], errors="coerce").astype(np.float32)
                encoded = encode_feature_frame(prepared, bucket_sizes=self.config.bucket_sizes, random_seed=self.config.seed)
                # Newer Search Conversion artifacts explicitly carry this
                # field. Older compatible artifacts do not, in which case a
                # finite non-negative value is its conservative equivalent.
                if "has_conversion_value" in prepared:
                    declared = pd.to_numeric(prepared["has_conversion_value"], errors="coerce").eq(1).to_numpy()
                    encoded["value_mask"] = (encoded["value_mask"].to_numpy(bool) & declared).astype(np.float32)
                    encoded.loc[encoded["value_mask"].eq(0), "log_conversion_value"] = np.nan
                remaining = None if worker_limit is None else max(0, worker_limit - emitted)
                if remaining == 0:
                    return
                if remaining is not None:
                    encoded = encoded.iloc[:remaining]
                emitted += len(encoded)
                dense = encoded[[f"dense__{name}" for name in DENSE_FEATURES]].to_numpy(np.float32)
                sparse = encoded[[f"sparse__{name}" for name in SPARSE_FEATURES]].to_numpy(np.int64)
                labels = encoded.conversion_label.to_numpy(np.float32)
                masks = encoded.value_mask.to_numpy(np.float32)
                log_values = np.nan_to_num(encoded.log_conversion_value.to_numpy(np.float32), nan=0.0)
                values = np.nan_to_num(encoded.conversion_value_eur.to_numpy(np.float32), nan=0.0)
                for index in range(len(encoded)):
                    row: dict[str, Any] = {"dense": torch.from_numpy(dense[index]), "sparse": torch.from_numpy(sparse[index]), "label": torch.tensor(labels[index]), "log_value": torch.tensor(log_values[index]), "value_mask": torch.tensor(masks[index]), "observed_value": torch.tensor(values[index])}
                    if self.include_identifiers:
                        row.update(user_id=str(encoded.user_id.iat[index]), product_id=str(encoded.candidate_ad_id.iat[index]))
                    yield row


def multitask_loss(logits: Tensor, predicted_log_value: Tensor, labels: Tensor, log_values: Tensor, value_mask: Tensor, *, lambda_cvr: float, lambda_value: float) -> tuple[Tensor, Tensor, Tensor]:
    """AMP-safe BCE and a positive-conversion-only SmoothL1 value loss."""
    cvr = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
    valid = value_mask.bool()
    value = F.smooth_l1_loss(predicted_log_value.float()[valid], log_values.float()[valid]) if bool(valid.any()) else logits.new_zeros((), dtype=torch.float32)
    return lambda_cvr * cvr + lambda_value * value, cvr, value


def decode_predictions(logits: Tensor, predicted_log_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pcvr = torch.sigmoid(logits.float())
    conditional_value = torch.expm1(predicted_log_value.float().clamp(min=0.0, max=20.0)).clamp_min(0.0)
    return pcvr, conditional_value, pcvr * conditional_value


def build_model(name: str, config: FineRankMultiTaskConfig) -> torch.nn.Module:
    kwargs = dict(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=config.bucket_sizes, embedding_dim=config.embedding_dim, hidden_dims=config.hidden_dims)
    if name == "deepfm":
        return DeepFMMultiTask(**kwargs)
    if name == "dcnv2":
        return DCNv2MultiTask(**kwargs, num_cross_layers=config.cross_layers)
    raise ValueError(f"Unsupported available backbone: {name}")


def run_fine_rank_multitask(config: FineRankMultiTaskConfig, *, stage: str) -> dict[str, Any]:
    """Train DeepFM/DCNv2 on Past and select strictly on Future-A."""
    _assert_temporal_contract(config)
    _seed(config.seed)
    device = _device(config.device)
    contract = feature_contract()
    availability = _din_availability(config.past_path)
    models: dict[str, Any] = {}
    for name in ("deepfm", "dcnv2"):
        models[name] = _train_one(name, config, device)
    models["din"] = {"model": "din", "available": False, "reason": availability["reason"], "parameter_count": 0}
    selected = max((models[name] for name in ("deepfm", "dcnv2")), key=lambda result: (result["future_a"]["cvr"]["pr_auc"] or float("-inf"), result["future_a"]["derived"]["top_decile_actual_value_per_click_lift"]))
    report = {
        "task_contract": {"pCVR_clicked": "P(conversion | clicked interaction)", "conditional_conversion_value": "E[conversion_value_eur | conversion=1, clicked interaction]", "derived_score": "expected conversion value per clicked interaction = pCVR_clicked * predicted_conditional_value"},
        "feature_contract": contract, "temporal_contract": {"train": str(config.past_path), "model_selection": str(config.future_a_path), "future_b_path": str(config.future_b_path), "future_b_read_for_model_selection": False},
        "stage": stage, "device": str(device), "amp_enabled": bool(config.amp and device.type == "cuda"), "models": models,
        "selected_model": selected["model"], "selection_reason": "Highest Future-A PR-AUC; top-decile actual value-per-click lift is the tie-breaker.",
    }
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_sanity" if stage == "sanity" else ""
    json_path = config.metrics_dir / f"fine_rank_metrics{suffix}.json"
    md_path = config.metrics_dir / f"fine_rank_metrics{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"metrics_json": str(json_path), "metrics_markdown": str(md_path), "selected_model": selected["model"], "future_b_read_for_model_selection": False}


def _train_one(name: str, config: FineRankMultiTaskConfig, device: torch.device) -> dict[str, Any]:
    _seed(config.seed)
    model = build_model(name, config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    best_pr, stale, history = float("-inf"), 0, []
    checkpoint = config.model_dir / f"{name}.pt"
    for epoch in range(1, config.epochs + 1):
        model.train(); rows = 0; totals = np.zeros(3, dtype=float)
        loader = _loader(config.past_path, config, config.max_train_rows)
        for batch in loader:
            dense, sparse, label, log_value, mask, _ = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits, predicted = model(dense, sparse)
                loss, cvr, value = multitask_loss(logits, predicted, label, log_value, mask, lambda_cvr=config.lambda_cvr, lambda_value=config.lambda_value)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite multi-task fine-rank loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            size = len(label); rows += size; totals += np.array([float(loss.detach()), float(cvr.detach()), float(value.detach())]) * size
        metrics = _evaluate(model, config, device)
        record = {"epoch": epoch, "train_rows": rows, "train_loss": float(totals[0] / max(rows, 1)), "cvr_loss": float(totals[1] / max(rows, 1)), "value_loss": float(totals[2] / max(rows, 1)), "future_a_pr_auc": metrics["cvr"]["pr_auc"]}
        history.append(record)
        score = metrics["cvr"]["pr_auc"] if metrics["cvr"]["pr_auc"] is not None else float("-inf")
        if score > best_pr:
            best_pr, stale = score, 0
            _save_checkpoint(checkpoint, model, config, name, epoch)
            best_metrics = metrics
        else:
            stale += 1
            if stale > config.patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return {"model": name, "available": True, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "cross_layers": config.cross_layers if name == "dcnv2" else None, "checkpoint": str(checkpoint), "best_epoch": saved["epoch"], "history": history, "future_a": _evaluate(model, config, device), "future_b_read_for_model_selection": False}


@torch.no_grad()
def _evaluate(model: torch.nn.Module, config: FineRankMultiTaskConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    # Exact CVR metrics need the predictions, but store them on disk rather
    # than building an in-RAM list for a full Future-A evaluation.
    rows = _count_rows(config.future_a_path, config, config.max_validation_rows)
    if not rows:
        return _metrics(np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool))
    with tempfile.TemporaryDirectory(prefix="fine-rank-metrics-") as directory:
        root = Path(directory)
        y = np.memmap(root / "labels.bin", dtype=np.float32, mode="w+", shape=(rows,))
        p = np.memmap(root / "pcvr.bin", dtype=np.float32, mode="w+", shape=(rows,))
        predicted = np.memmap(root / "value.bin", dtype=np.float32, mode="w+", shape=(rows,))
        actual = np.memmap(root / "actual.bin", dtype=np.float32, mode="w+", shape=(rows,))
        valid = np.memmap(root / "valid.bin", dtype=np.bool_, mode="w+", shape=(rows,))
        position = 0
        for batch in _loader(config.future_a_path, config, config.max_validation_rows):
            dense, sparse, label, _, mask, observed = _to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits, log_value = model(dense, sparse)
            pcvr, value, _ = decode_predictions(logits, log_value)
            end = position + len(label)
            y[position:end] = label.cpu().numpy(); p[position:end] = pcvr.cpu().numpy(); predicted[position:end] = value.cpu().numpy(); actual[position:end] = observed.cpu().numpy(); valid[position:end] = mask.cpu().numpy().astype(bool)
            position = end
        return _metrics(y[:position], p[:position], predicted[:position], actual[:position], valid[:position])


def _metrics(y: np.ndarray, p: np.ndarray, predicted: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    has_both = len(np.unique(y)) == 2
    cvr = {"roc_auc": float(roc_auc_score(y, p)) if has_both else None, "pr_auc": float(average_precision_score(y, p)) if has_both else None, "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None, "brier_score": float(np.mean((p - y) ** 2)) if len(y) else None, "label_mean": float(y.mean()) if len(y) else None, "prediction_mean": float(p.mean()) if len(p) else None, "positive_count": int(y.sum()), "negative_count": int(len(y) - y.sum())}
    actual_valid, predicted_valid = actual[valid], predicted[valid]
    value = {"rows": int(valid.sum()), "mae": float(mean_absolute_error(actual_valid, predicted_valid)) if len(actual_valid) else None, "rmse": float(math.sqrt(mean_squared_error(actual_valid, predicted_valid))) if len(actual_valid) else None, "rmsle": float(math.sqrt(mean_squared_error(np.log1p(actual_valid), np.log1p(predicted_valid)))) if len(actual_valid) else None, "label_mean": float(actual_valid.mean()) if len(actual_valid) else None, "prediction_mean": float(predicted_valid.mean()) if len(predicted_valid) else None}
    expected = p * predicted
    overall = float(actual.sum() / len(actual)) if len(actual) else 0.0
    deciles: list[dict[str, Any]] = []
    if len(expected):
        order = np.argsort(expected); groups = np.array_split(order, 10)
        for index, group in enumerate(groups, 1):
            deciles.append({"decile": index, "rows": int(len(group)), "predicted_pCVR_mean": float(p[group].mean()), "actual_CVR": float(y[group].mean()), "predicted_conditional_value_mean": float(predicted[group].mean()), "actual_conversion_value_mean": float(actual[group].sum() / max(int(y[group].sum()), 1)), "predicted_expected_value_mean": float(expected[group].mean()), "actual_value_per_click": float(actual[group].sum() / len(group))})
    top_actual = deciles[-1]["actual_value_per_click"] if deciles else 0.0
    return {"cvr": cvr, "value": value, "derived": {"expected_conversion_value_per_click_prediction_mean": float(expected.mean()) if len(expected) else None, "overall_actual_value_per_click": overall, "top_decile_actual_value_per_click": top_actual, "top_decile_actual_value_per_click_lift": float(top_actual / overall) if overall else 0.0, "deciles": deciles}}


def _loader(path: Path, config: FineRankMultiTaskConfig, limit: int | None) -> DataLoader[Any]:
    return DataLoader(ClickInteractionDataset(path, config, max_rows=limit), batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.device != "cpu")


def _count_rows(path: Path, config: FineRankMultiTaskConfig, limit: int | None) -> int:
    """Count valid clicked interactions in a bounded first pass."""
    count = 0
    for frame in iter_csv_parts(path, config.chunk_size):
        labels = pd.to_numeric(frame.get("conversion_label"), errors="coerce")
        valid = labels.isin((0, 1)) & frame.get("user_id", pd.Series(index=frame.index)).notna() & frame.get("product_id", pd.Series(index=frame.index)).notna()
        count += int(valid.sum())
        if limit is not None and count >= limit:
            return limit
    return count


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(batch[key].to(device, non_blocking=True) for key in ("dense", "sparse", "label", "log_value", "value_mask", "observed_value"))  # type: ignore[return-value]


def _assert_temporal_contract(config: FineRankMultiTaskConfig) -> None:
    assert_no_fine_rank_leakage([*DENSE_FEATURES, *SPARSE_FEATURES])
    if not config.past_path.is_dir() or not config.future_a_path.is_dir() or not config.future_b_path.is_dir():
        raise FileNotFoundError("Search Conversion temporal split must already contain Past, Future-A, and Future-B directories")
    # Future-B is deliberately not opened: its existence establishes the
    # three-window contract, while only Past/Future-A are read below.
    bounds = {"past": _timestamp_bounds(config.past_path, config.chunk_size), "future_a": _timestamp_bounds(config.future_a_path, config.chunk_size)}
    if None in bounds["past"] or None in bounds["future_a"] or not bounds["past"][1] < bounds["future_a"][0]:
        raise ValueError(f"Temporal contract requires Past < Future-A, got {bounds}")


def _timestamp_bounds(path: Path, chunk_size: int) -> tuple[int | None, int | None]:
    low = high = None
    for chunk in iter_csv_parts(path, chunk_size):
        times = pd.to_numeric(chunk.get("click_timestamp"), errors="coerce").dropna()
        if len(times):
            low = int(times.min()) if low is None else min(low, int(times.min()))
            high = int(times.max()) if high is None else max(high, int(times.max()))
    return low, high


def _din_availability(past: Path) -> dict[str, Any]:
    header = next(iter(sorted(past.glob("part-*.csv"))), None)
    columns = set(pd.read_csv(header, nrows=0).columns) if header else set()
    sequence_columns = [name for name in columns if any(token in name.lower() for token in ("history", "sequence", "behavior"))]
    if sequence_columns:
        return {"available": False, "reason": "Sequence-like columns exist but no validated candidate-aware history contract is implemented; DIN is not enabled from ambiguous fields."}
    return {"available": False, "reason": "Search Conversion source has no reliable user behavior/history sequence; DIN history is not fabricated."}


def _save_checkpoint(path: Path, model: torch.nn.Module, config: FineRankMultiTaskConfig, name: str, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    serialized_config = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    torch.save({"model": name, "epoch": epoch, "state_dict": model.state_dict(), "config": serialized_config, "feature_contract": feature_contract(), "future_b_read_for_model_selection": False}, temporary)
    temporary.replace(path)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed(seed: int) -> None:
    # Required by CUDA for deterministic GEMM kernels when deterministic
    # algorithms are requested. Set before any CUDA work rather than disabling
    # reproducibility to silence CuBLAS warnings.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _optional_positive_int(value: Any) -> int | None:
    if value is None: return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Search Conversion conversion-only fine-rank metrics", "", "- Future-B was not read for model selection: `false`.", "- Scores are expected conversion value per clicked interaction, not CTR, eCPM, impression value, ROI, or auction value.", "", "| Model | Future-A PR-AUC | ROC-AUC | LogLoss | Top-decile value/click lift |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in report["models"].items():
        if not result.get("available"):
            lines.append(f"| {name} | unavailable | unavailable | unavailable | unavailable |")
            continue
        cvr, derived = result["future_a"]["cvr"], result["future_a"]["derived"]
        lines.append(f"| {name} | {cvr['pr_auc']:.6f} | {cvr['roc_auc']:.6f} | {cvr['logloss']:.6f} | {derived['top_decile_actual_value_per_click_lift']:.4f} |")
    lines.extend(("", f"Selected model: `{report['selected_model']}`.", report["selection_reason"], ""))
    return "\n".join(lines)
