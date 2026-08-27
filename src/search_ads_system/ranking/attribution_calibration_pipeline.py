"""Strict-time, Future-A-only calibration pipeline for a trained Attribution ESMM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from search_ads_system.common.config import resolve_path
from search_ads_system.data.storage import prepare_output_directory, write_csv_part
from search_ads_system.ranking.attribution_calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    RawCalibrator,
    UnavailableCalibrator,
    calibration_metrics,
    clip_probabilities,
    fit_isotonic,
    fit_platt,
    select_calibrator,
    serving_consistent_probabilities,
)
from search_ads_system.ranking.attribution_esmm import CATEGORICAL_FEATURES
from search_ads_system.ranking.attribution_esmm_pipeline import (
    AttributionESMMConfig,
    NumericNormalization,
    _encode_frame,
    build_model,
    parse_attribution_esmm_config,
    resolve_device,
)


LOGGER = logging.getLogger(__name__)
_INFERENCE_COLUMNS = ("event_id", "timestamp", *CATEGORICAL_FEATURES, "time_since_last_click", "click", "conversion", "click_and_conversion")
_TARGETS: dict[str, tuple[str, str, bool]] = {
    "ctr": ("click", "raw_pctr", False),
    "ctcvr": ("click_and_conversion", "raw_pctcvr", False),
    "cvr": ("conversion", "raw_pcvr", True),
}


@dataclass(frozen=True)
class AttributionCalibrationConfig:
    esmm: AttributionESMMConfig
    checkpoint_path: Path
    output_dir: Path
    metrics_dir: Path
    fit_ratio: float
    epsilon: float
    reliability_bins: int
    inference_batch_size: int
    io_chunk_size: int
    mixed_precision: bool
    sanity_max_rows: int


@dataclass
class _SplitStats:
    rows: int = 0
    timestamp_min: int | None = None
    timestamp_max: int | None = None
    ctr_positives: int = 0
    ctcvr_positives: int = 0
    clicked_rows: int = 0
    clicked_conversions: int = 0

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
        click = pd.to_numeric(frame["click"], errors="raise")
        conversion = pd.to_numeric(frame["conversion"], errors="raise")
        ctcvr = pd.to_numeric(frame["click_and_conversion"], errors="raise")
        self.rows += len(frame)
        self.timestamp_min = int(timestamp.min()) if self.timestamp_min is None else min(self.timestamp_min, int(timestamp.min()))
        self.timestamp_max = int(timestamp.max()) if self.timestamp_max is None else max(self.timestamp_max, int(timestamp.max()))
        self.ctr_positives += int(click.sum())
        self.ctcvr_positives += int(ctcvr.sum())
        self.clicked_rows += int(click.sum())
        self.clicked_conversions += int(conversion.loc[click.eq(1)].sum())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_attribution_calibration_config(raw_config: Mapping[str, Any], config_path: Path) -> AttributionCalibrationConfig:
    """Build an independent calibration config containing only Future-A paths."""

    options = raw_config.get("attribution_calibration", {})
    if not isinstance(options, Mapping):
        raise ValueError("attribution_calibration must be a mapping")
    root = config_path.parent.resolve()
    esmm = parse_attribution_esmm_config(raw_config, config_path)
    config = AttributionCalibrationConfig(
        esmm=esmm,
        checkpoint_path=resolve_path(str(options.get("checkpoint_path", "outputs/attribution/models/esmm.pt")), root),
        output_dir=resolve_path(str(options.get("output_dir", "outputs/attribution/calibration")), root),
        metrics_dir=resolve_path(str(options.get("metrics_dir", "outputs/attribution/metrics")), root),
        fit_ratio=float(options.get("fit_ratio", 0.5)),
        epsilon=float(options.get("epsilon", 1e-7)),
        reliability_bins=int(options.get("reliability_bins", 20)),
        inference_batch_size=int(options.get("inference_batch_size", esmm.inference_batch_size)),
        io_chunk_size=int(options.get("io_chunk_size", esmm.io_chunk_size)),
        mixed_precision=bool(options.get("mixed_precision", esmm.mixed_precision)),
        sanity_max_rows=int(options.get("sanity", {}).get("max_rows", 100000)),
    )
    if not 0.0 < config.fit_ratio < 1.0 or not 0.0 < config.epsilon < 0.5 or config.reliability_bins <= 0:
        raise ValueError("Attribution calibration fit_ratio, epsilon, or reliability_bins is invalid")
    if config.inference_batch_size <= 0 or config.io_chunk_size <= 0 or config.sanity_max_rows <= 0:
        raise ValueError("Attribution calibration batch/chunk/sanity sizes must be positive")
    return config


def generate_prediction_artifacts(config: AttributionCalibrationConfig, *, artifact_suffix: str = "", max_rows: int | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Split Future-A by timestamp and stream checkpoint predictions to disk."""

    source = config.esmm.future_a_path
    limit = max_rows
    boundary = _future_a_boundary(source, config.io_chunk_size, config.fit_ratio, limit)
    fit_directory, eval_directory, metadata_path = _artifact_paths(config, artifact_suffix)
    prepare_output_directory(fit_directory, overwrite=overwrite)
    prepare_output_directory(eval_directory, overwrite=overwrite)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
    normalization = NumericNormalization(**checkpoint["normalization"])
    device = resolve_device(config.esmm.device)
    model = build_model(config.esmm, "esmm").to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_parts = {"fit": 0, "eval": 0}
    stats = {"calibration_fit": _SplitStats(), "calibration_eval": _SplitStats()}
    source_rows = 0
    previous_timestamp: int | None = None
    started = time.perf_counter()
    for part in sorted(source.glob("part-*.csv")):
        if limit is not None and source_rows >= limit:
            break
        for frame in pd.read_csv(part, usecols=list(_INFERENCE_COLUMNS), chunksize=config.io_chunk_size, low_memory=False):
            if limit is not None:
                frame = frame.iloc[:limit - source_rows]
            if frame.empty:
                break
            timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
            if previous_timestamp is not None and int(timestamp.iloc[0]) < previous_timestamp or (timestamp.diff().dropna() < 0).any():
                raise ValueError("Future-A timestamp order is not non-decreasing")
            previous_timestamp = int(timestamp.iloc[-1])
            predicted = _predict_frame(model, frame, config, normalization, device)
            augmented = frame.loc[:, ["event_id", "timestamp", "click", "conversion", "click_and_conversion"]].copy()
            augmented["raw_pctr"] = predicted["pctr"]
            augmented["raw_pctcvr"] = predicted["pctcvr"]
            augmented["raw_pcvr"] = predicted["pcvr"]
            partitions = {
                "fit": augmented.loc[timestamp < boundary],
                "eval": augmented.loc[timestamp >= boundary],
            }
            if sum(len(value) for value in partitions.values()) != len(augmented):
                raise AssertionError("Calibration temporal split did not assign every Future-A row")
            for name, partition in partitions.items():
                if partition.empty:
                    continue
                write_csv_part(partition, fit_directory if name == "fit" else eval_directory, output_parts[name])
                output_parts[name] += 1
                stats["calibration_fit" if name == "fit" else "calibration_eval"].update(partition)
            source_rows += len(frame)
    if not stats["calibration_fit"].rows or not stats["calibration_eval"].rows:
        raise ValueError("Future-A calibration split produced an empty window")
    if not stats["calibration_fit"].timestamp_max < stats["calibration_eval"].timestamp_min:
        raise ValueError("Calibration temporal contract failed: max(fit.timestamp) < min(eval.timestamp)")
    elapsed = time.perf_counter() - started
    metadata = {
        "temporal_contract": "Future-A contiguous timestamp split: max(calibration_fit.timestamp) < min(calibration_eval.timestamp)",
        "boundary_eval_timestamp": boundary,
        "source_rows_processed": source_rows,
        "splits": {name: value.as_dict() for name, value in stats.items()},
        "inference": {"checkpoint": str(config.checkpoint_path), "device": str(device), "elapsed_seconds": elapsed, "rows_per_second": source_rows / elapsed if elapsed else 0.0, "inference_batch_size": config.inference_batch_size},
        "future_b_read_for_calibration_selection": False,
    }
    _write_json(metadata_path, metadata)
    return metadata


def calibrate_and_evaluate(config: AttributionCalibrationConfig, *, artifact_suffix: str = "") -> dict[str, Any]:
    """Fit on calibration-fit artifacts, evaluate/select only on calibration-eval."""

    fit_directory, eval_directory, metadata_path = _artifact_paths(config, artifact_suffix)
    temporal_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    calibrator_directory = _calibrator_directory(config, artifact_suffix)
    calibrator_directory.mkdir(parents=True, exist_ok=True)
    target_reports: dict[str, Any] = {}
    selected_predictions: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for target, (label_column, prediction_column, clicked_only) in _TARGETS.items():
        fit_labels, fit_predictions = _load_target(fit_directory, label_column, prediction_column, clicked_only)
        eval_labels, eval_predictions = _load_target(eval_directory, label_column, prediction_column, clicked_only)
        calibrators = {
            "raw": RawCalibrator(),
            "platt": fit_platt(fit_labels, fit_predictions, config.epsilon),
            "isotonic": fit_isotonic(fit_labels, fit_predictions, config.epsilon),
        }
        methods: dict[str, Any] = {}
        predictions_by_kind: dict[str, np.ndarray] = {}
        for kind, calibrator in calibrators.items():
            _save_calibrator(calibrator_directory, target, calibrator)
            if isinstance(calibrator, UnavailableCalibrator):
                methods[kind] = {"available": False, "reason": calibrator.reason}
                continue
            method_predictions = calibrator.predict(eval_predictions)
            predictions_by_kind[kind] = method_predictions
            methods[kind] = {"available": True, "metrics": calibration_metrics(eval_labels, method_predictions, config.reliability_bins, config.epsilon)}
        selection_inputs = {kind: payload["metrics"] if payload.get("available") else {"available": False} for kind, payload in methods.items()}
        selected, reason = select_calibrator(selection_inputs)
        selected_predictions[target] = predictions_by_kind[selected]
        target_reports[target] = {
            "fit_rows": int(len(fit_labels)),
            "eval_rows": int(len(eval_labels)),
            "methods": methods,
            "selected_calibrator": selected,
            "selection_reason": reason,
            "fit_parameters": {kind: _calibrator_metadata(value) for kind, value in calibrators.items()},
        }
    serving_pcvr, consistency = serving_consistent_probabilities(selected_predictions["ctr"], selected_predictions["ctcvr"], config.epsilon)
    report = {
        "temporal_contract": temporal_metadata,
        "future_b_read_for_calibration_selection": False,
        "comparison_contract": "All calibrators fit only on Future-A calibration-fit and are compared/selected only on the later Future-A calibration-eval window.",
        "targets": target_reports,
        "selected_calibrator": {target: payload["selected_calibrator"] for target, payload in target_reports.items()},
        "serving_consistency": {
            "policy": "select calibrated pCTR and pCTCVR independently; derive serving pCVR = clip(pCTCVR / max(pCTR, eps), 0, 1)",
            **consistency,
            "serving_pcvr_mean": float(serving_pcvr.mean()),
            "independent_cvr_calibrator_is_diagnostic_only": True,
        },
        "calibration_elapsed_seconds": time.perf_counter() - started,
        "caveats": ["Calibration aims to improve probability quality, not ranking AUC.", "Platt and Isotonic are monotonic transforms, so ROC-AUC/PR-AUC should normally remain effectively unchanged.", "Future-B remains untouched."],
    }
    json_path, markdown_path = _metrics_paths(config, artifact_suffix)
    _write_json(json_path, report)
    _write_markdown(markdown_path, report)
    return report


def _predict_frame(model: torch.nn.Module, frame: pd.DataFrame, config: AttributionCalibrationConfig, normalization: NumericNormalization, device: torch.device) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {"pctr": [], "pctcvr": [], "pcvr": []}
    with torch.no_grad():
        for start in range(0, len(frame), config.inference_batch_size):
            encoded = _encode_frame(frame.iloc[start:start + config.inference_batch_size], config.esmm.bucket_sizes, normalization)
            sparse = encoded["sparse"].to(device, non_blocking=True)
            dense = encoded["dense"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=config.mixed_precision and device.type == "cuda"):
                model_outputs = model(sparse, dense)
            pctr = model_outputs["pctr"].float().cpu().numpy()
            pctcvr = model_outputs["pctcvr"].float().cpu().numpy()
            pcvr = np.clip(pctcvr / np.maximum(pctr, config.epsilon), 0.0, 1.0)
            outputs["pctr"].append(np.clip(pctr, 0.0, 1.0))
            outputs["pctcvr"].append(np.clip(pctcvr, 0.0, 1.0))
            outputs["pcvr"].append(pcvr)
    return {name: np.concatenate(values) if values else np.empty(0, dtype=np.float32) for name, values in outputs.items()}


def _future_a_boundary(directory: Path, chunk_size: int, fit_ratio: float, max_rows: int | None) -> int:
    total = 0
    previous: int | None = None
    for part in sorted(directory.glob("part-*.csv")):
        if max_rows is not None and total >= max_rows:
            break
        for frame in pd.read_csv(part, usecols=["timestamp"], chunksize=chunk_size, low_memory=False):
            if max_rows is not None:
                frame = frame.iloc[:max_rows - total]
            if frame.empty:
                break
            timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
            if previous is not None and int(timestamp.iloc[0]) < previous or (timestamp.diff().dropna() < 0).any():
                raise ValueError("Future-A timestamp order is not non-decreasing")
            previous = int(timestamp.iloc[-1])
            total += len(frame)
    target = int(total * fit_ratio)
    if not 0 < target < total:
        raise ValueError("Future-A has too few rows for a calibration fit/eval split")
    base: int | None = None
    row_index = 0
    for part in sorted(directory.glob("part-*.csv")):
        if max_rows is not None and row_index >= max_rows:
            break
        for frame in pd.read_csv(part, usecols=["timestamp"], chunksize=chunk_size, low_memory=False):
            if max_rows is not None:
                frame = frame.iloc[:max_rows - row_index]
            for timestamp in pd.to_numeric(frame["timestamp"], errors="raise"):
                if row_index == target - 1:
                    base = int(timestamp)
                elif base is not None and int(timestamp) > base:
                    return int(timestamp)
                row_index += 1
    raise ValueError("Future-A timestamp ties prevent a strict calibration fit/eval split")


def _load_target(directory: Path, label_column: str, prediction_column: str, clicked_only: bool) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for part in sorted(directory.glob("part-*.csv")):
        for frame in pd.read_csv(part, usecols=["click", label_column, prediction_column], chunksize=200000, low_memory=False):
            if clicked_only:
                frame = frame.loc[pd.to_numeric(frame["click"], errors="raise").eq(1)]
            if not frame.empty:
                labels.append(pd.to_numeric(frame[label_column], errors="raise").to_numpy(dtype=np.int64))
                predictions.append(pd.to_numeric(frame[prediction_column], errors="raise").to_numpy(dtype=np.float64))
    if not labels:
        raise ValueError(f"No calibration rows for target {label_column} in {directory}")
    return np.concatenate(labels), np.concatenate(predictions)


def _artifact_paths(config: AttributionCalibrationConfig, artifact_suffix: str) -> tuple[Path, Path, Path]:
    root = config.output_dir / "predictions"
    if artifact_suffix:
        root = root / artifact_suffix
    return root / "calibration_fit", root / "calibration_eval", root / "split_metadata.json"


def _calibrator_directory(config: AttributionCalibrationConfig, artifact_suffix: str) -> Path:
    return config.output_dir / "calibrators" / artifact_suffix if artifact_suffix else config.output_dir / "calibrators"


def _metrics_paths(config: AttributionCalibrationConfig, artifact_suffix: str) -> tuple[Path, Path]:
    suffix = "_sanity" if artifact_suffix else ""
    return config.metrics_dir / f"calibration_metrics{suffix}.json", config.metrics_dir / f"calibration_metrics{suffix}.md"


def _save_calibrator(directory: Path, target: str, calibrator: Any) -> None:
    if isinstance(calibrator, IsotonicCalibrator):
        path = directory / f"{target}_isotonic.pkl"
        temporary = path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as file:
            pickle.dump(calibrator, file)
        temporary.replace(path)
    else:
        _write_json(directory / f"{target}_{calibrator.kind}.json", _calibrator_metadata(calibrator))


def _calibrator_metadata(calibrator: Any) -> dict[str, Any]:
    if isinstance(calibrator, PlattCalibrator):
        return calibrator.metadata()
    if isinstance(calibrator, UnavailableCalibrator):
        return {"kind": calibrator.kind, "available": False, "reason": calibrator.reason}
    if isinstance(calibrator, RawCalibrator):
        return {"kind": "raw"}
    if isinstance(calibrator, IsotonicCalibrator):
        return {"kind": "isotonic", "x_threshold_count": int(len(calibrator.model.X_thresholds_)), "y_threshold_count": int(len(calibrator.model.y_thresholds_))}
    raise TypeError(f"Unexpected calibrator type: {type(calibrator)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = ["# Attribution ESMM Calibration", "", "Future-A is split into earlier calibration-fit and later calibration-eval windows. Future-B was not read.", ""]
    for target, payload in report["targets"].items():
        lines.extend((f"## {target}", ""))
        for method, method_payload in payload["methods"].items():
            if not method_payload.get("available"):
                lines.append(f"- {method}: unavailable ({method_payload['reason']})")
                continue
            metrics = method_payload["metrics"]
            lines.append(f"- {method}: LogLoss={metrics['logloss']}, Brier={metrics['brier_score']}, ECE={metrics['ece']}, ROC-AUC={metrics['roc_auc']}, PR-AUC={metrics['pr_auc']}")
        lines.append(f"- selected: {payload['selected_calibrator']} ({payload['selection_reason']['reason']})")
        lines.append("")
    consistency = report["serving_consistency"]
    lines.extend(("## Serving consistency", "", f"- max error: {consistency['max_abs_error']}", f"- mean error: {consistency['mean_abs_error']}", f"- fraction requiring clipping: {consistency['fraction_requiring_clipping']}", "", "future_b_read_for_calibration_selection: false", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
