"""Strict, read-only diagnostics for an existing DCNv2 fine-rank checkpoint.

The audit intentionally does not change weights or rebuild the cache.  In full
mode it makes the row-level split limitation explicit and measures the degree
to which validation reuses training identities and user-product interactions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from search_ads_system.common.config import resolve_path
from search_ads_system.ranking.fine_rank import (
    FineRankConfig, _loader, _prediction_kwargs, _to_device, _value_transform,
    build_model, dataset_spec, load_fine_ranker, resolve_device,
)
from search_ads_system.ranking.fine_rank_dataset import DENSE_FEATURES, LEAKAGE_COLUMNS, SPARSE_FEATURES, assert_no_fine_rank_leakage

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FineRankAuditConfig:
    output_path: Path
    full_reference_path: Path | None = None
    calibration_bins: int = 20
    output_chunk_size: int = 500_000
    ablation_train_rows: int = 100_000
    ablation_validation_rows: int = 50_000
    ablation_epochs: int = 1
    ablation_batch_size: int = 8192
    random_seed: int = 2026


def parse_fine_rank_audit_config(raw_config: Mapping[str, Any], config_path: Path, fine_rank: FineRankConfig) -> FineRankAuditConfig:
    options = raw_config.get("fine_rank", {})
    if not isinstance(options, Mapping):
        raise ValueError("fine_rank must be a mapping")
    temporal = raw_config.get("temporal", {})
    temporal_options = temporal.get("fine_rank", {}) if isinstance(temporal, Mapping) else {}
    if fine_rank.mode == "temporal" and not isinstance(temporal_options, Mapping):
        raise ValueError("temporal.fine_rank must be a mapping")
    # Do not inherit a full-mode audit output path in temporal mode: every
    # temporal report must remain below outputs/temporal.
    effective = temporal_options if fine_rank.mode == "temporal" else options
    audit = effective.get("audit", {}) if isinstance(effective, Mapping) else {}
    if not isinstance(audit, Mapping):
        raise ValueError("fine_rank.audit must be a mapping")
    root = config_path.parent.resolve()
    default_output = fine_rank.metrics_path.parent / "fine_rank_audit.json" if fine_rank.mode == "temporal" else fine_rank.output_path.parent / "fine_rank_audit.json"
    full_audit = options.get("audit", {})
    full_reference = resolve_path(str(full_audit.get("output_path", "outputs/metrics/fine_rank_audit.json")), root) if isinstance(full_audit, Mapping) else None
    config = FineRankAuditConfig(
        output_path=resolve_path(str(audit.get("output_path", default_output)), root),
        full_reference_path=full_reference if fine_rank.mode == "temporal" else None,
        calibration_bins=int(audit.get("calibration_bins", 20)),
        output_chunk_size=int(audit.get("output_chunk_size", 500_000)),
        ablation_train_rows=int(audit.get("ablation_train_rows", 100_000)),
        ablation_validation_rows=int(audit.get("ablation_validation_rows", 50_000)),
        ablation_epochs=int(audit.get("ablation_epochs", 1)),
        ablation_batch_size=int(audit.get("ablation_batch_size", 8192)),
        random_seed=int(audit.get("random_seed", fine_rank.random_seed)),
    )
    if config.calibration_bins < 2 or config.output_chunk_size <= 0 or min(config.ablation_train_rows, config.ablation_validation_rows, config.ablation_epochs, config.ablation_batch_size) <= 0:
        raise ValueError("fine_rank.audit numeric configuration is invalid")
    if fine_rank.mode == "temporal" and "temporal" not in config.output_path.resolve().parts:
        raise ValueError("Temporal fine-rank audit output must be isolated below outputs/temporal")
    return config


def run_fine_rank_audit(config: FineRankConfig, audit_config: FineRankAuditConfig, *, include_ablation: bool = False) -> dict[str, Any]:
    """Create a JSON and Markdown audit using a checkpoint without updating it."""
    spec = dataset_spec(config)
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Fine-rank checkpoint not found: {config.model_path}")
    if not any(spec.validation_dir.glob("part-*.parquet")):
        raise FileNotFoundError(f"Fine-rank validation cache not found: {spec.validation_dir}")
    audit_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="fine_rank_audit_", suffix=".sqlite", dir=audit_config.output_path.parent, delete=False) as handle:
        database_path = Path(handle.name)
    connection = sqlite3.connect(database_path)
    try:
        _create_overlap_tables(connection)
        overlap = _build_overlap_audit(connection, spec.cache_dir, spec.validation_dir)
        model, checkpoint = load_fine_ranker(config)
        transform = _value_transform(checkpoint.get("dataset_metadata", {}))
        validation = _score_validation(model, config, transform, connection, audit_config.calibration_bins)
        report: dict[str, Any] = {
            "audit_version": "fine-rank-effect-audit-v1",
            "checkpoint": str(config.model_path),
            "mode": config.mode,
            "rows": {"train": overlap["train_rows"], "validation": validation["rows"]},
            "validation_prediction_distribution": validation["prediction_distribution"],
            "classification_metrics": validation["classification_metrics"],
            "calibration": validation["calibration"],
            "train_validation_overlap": overlap,
            "strict_holdout_slices": _strict_holdout_metrics(connection),
            "feature_usage": _feature_usage(spec.cache_dir, spec.validation_dir),
            "leakage_audit": _leakage_audit(config),
            "split_audit": _split_audit(config),
            "fine_rank_output_diagnostics": _output_diagnostics(config.output_path, audit_config.output_chunk_size),
        }
        if include_ablation:
            report["id_memorization_ablation"] = run_id_memorization_ablation(config, audit_config)
        else:
            report["id_memorization_ablation"] = {
                "ran": False,
                "command_flag": "--with-id-ablation",
                "description": "Small temporary A/B/C/D training experiment; it never writes or changes the production checkpoint.",
            }
        report["recommendations"] = _recommendations(report)
        audit_config.output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        markdown_path = audit_config.output_path.with_suffix(".md")
        markdown_path.write_text(_render_markdown(report), encoding="utf-8")
        comparison_path = _write_full_vs_temporal_report(report, audit_config) if config.mode == "temporal" else None
        LOGGER.info("Fine-rank audit written to %s and %s", audit_config.output_path, markdown_path)
        return {"report_path": str(audit_config.output_path), "markdown_path": str(markdown_path), "comparison_path": str(comparison_path) if comparison_path else None, "classification_metrics": report["classification_metrics"], "overlap": overlap, "ablation_ran": include_ablation}
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def _create_overlap_tables(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE train_pairs (user_id TEXT NOT NULL, candidate_ad_id TEXT NOT NULL, row_count INTEGER NOT NULL, PRIMARY KEY(user_id, candidate_ad_id));
        CREATE TABLE validation_pairs (user_id TEXT NOT NULL, candidate_ad_id TEXT NOT NULL, row_count INTEGER NOT NULL, PRIMARY KEY(user_id, candidate_ad_id));
        CREATE TABLE train_users (user_id TEXT PRIMARY KEY);
        CREATE TABLE validation_users (user_id TEXT PRIMARY KEY);
        CREATE TABLE train_products (candidate_ad_id TEXT PRIMARY KEY);
        CREATE TABLE validation_products (candidate_ad_id TEXT PRIMARY KEY);
        CREATE TABLE validation_predictions (row_id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, candidate_ad_id TEXT NOT NULL, label INTEGER NOT NULL, pcvr REAL NOT NULL);
    """)


def _build_overlap_audit(connection: sqlite3.Connection, train_directory: Path, validation_directory: Path) -> dict[str, Any]:
    train_rows = _insert_identifier_rows(connection, train_directory, "train")
    validation_rows = _insert_identifier_rows(connection, validation_directory, "validation")
    connection.commit()
    train_pairs = _count(connection, "train_pairs")
    validation_pairs = _count(connection, "validation_pairs")
    unique_all_pairs = int(connection.execute("SELECT COUNT(*) FROM (SELECT user_id, candidate_ad_id FROM train_pairs UNION SELECT user_id, candidate_ad_id FROM validation_pairs)").fetchone()[0])
    pair_overlap = int(connection.execute("SELECT COUNT(*) FROM validation_pairs AS v INNER JOIN train_pairs AS t USING(user_id, candidate_ad_id)").fetchone()[0])
    validation_rows_with_seen_pair = int(connection.execute("SELECT COALESCE(SUM(v.row_count), 0) FROM validation_pairs AS v INNER JOIN train_pairs AS t USING(user_id, candidate_ad_id)").fetchone()[0])
    user = _overlap_counts(connection, "users", "user_id")
    product = _overlap_counts(connection, "products", "candidate_ad_id")
    return {
        "definition": "A duplicate interaction is a repeated exact (user_id, product_id) pair in the audited cached rows; timestamps are not retained in the cache.",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_unique_user_product_pairs": train_pairs,
        "validation_unique_user_product_pairs": validation_pairs,
        "exact_user_product_pair_overlap": {"pairs": pair_overlap, "validation_unique_pair_rate": _divide(pair_overlap, validation_pairs), "validation_row_rate": _divide(validation_rows_with_seen_pair, validation_rows)},
        "user_overlap": user,
        "product_overlap": product,
        "duplicate_interaction_rate": _divide(train_rows + validation_rows - unique_all_pairs, train_rows + validation_rows),
        "train_duplicate_interaction_rate": _divide(train_rows - train_pairs, train_rows),
        "validation_duplicate_interaction_rate": _divide(validation_rows - validation_pairs, validation_rows),
    }


def _insert_identifier_rows(connection: sqlite3.Connection, directory: Path, split: str) -> int:
    pair_table = f"{split}_pairs"; user_table = f"{split}_users"; product_table = f"{split}_products"; rows = 0
    for part in sorted(directory.glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=100_000, columns=["user_id", "candidate_ad_id"]):
            data = batch.to_pydict(); pairs = [(str(user), str(product)) for user, product in zip(data["user_id"], data["candidate_ad_id"])]
            rows += len(pairs)
            connection.executemany(f"INSERT INTO {pair_table} VALUES (?, ?, 1) ON CONFLICT(user_id, candidate_ad_id) DO UPDATE SET row_count=row_count+1", pairs)
            connection.executemany(f"INSERT OR IGNORE INTO {user_table} VALUES (?)", ((user,) for user, _ in pairs))
            connection.executemany(f"INSERT OR IGNORE INTO {product_table} VALUES (?)", ((product,) for _, product in pairs))
        connection.commit()
    return rows


def _overlap_counts(connection: sqlite3.Connection, noun: str, column: str) -> dict[str, Any]:
    train = _count(connection, f"train_{noun}"); validation = _count(connection, f"validation_{noun}")
    shared = int(connection.execute(f"SELECT COUNT(*) FROM validation_{noun} AS v INNER JOIN train_{noun} AS t USING({column})").fetchone()[0])
    return {"train_unique": train, "validation_unique": validation, "shared": shared, "validation_coverage_rate": _divide(shared, validation), "jaccard": _divide(shared, train + validation - shared)}


def _score_validation(model: torch.nn.Module, config: FineRankConfig, transform: Mapping[str, float], connection: sqlite3.Connection, bins: int) -> dict[str, Any]:
    device = resolve_device(config.device)
    loader = _loader(dataset_spec(config).validation_dir, config, batch_size=config.inference_batch_size, value_transform=transform, include_identifiers=True, force_workers=0)
    labels: list[np.ndarray] = []; probabilities: list[np.ndarray] = []; row_id = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            dense, sparse, batch_labels, _, _ = _to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                probability, _, _, _ = model.predict_with_log(dense, sparse, **_prediction_kwargs(transform))
            y = batch_labels.detach().to(dtype=torch.int8).cpu().numpy()
            p = probability.detach().to(dtype=torch.float32).cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError("Fine-rank audit found non-finite validation pCVR")
            labels.append(y); probabilities.append(p)
            rows = [(row_id + index, str(user), str(product), int(label), float(score)) for index, (user, product, label, score) in enumerate(zip(batch["user_id"], batch["candidate_ad_id"], y, p))]
            connection.executemany("INSERT INTO validation_predictions VALUES (?, ?, ?, ?, ?)", rows)
            connection.commit(); row_id += len(rows)
    y = np.concatenate(labels) if labels else np.empty(0, dtype=np.int8)
    p = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    return {"rows": int(len(y)), "prediction_distribution": {"conversion_label_1": _distribution(p[y == 1]), "conversion_label_0": _distribution(p[y == 0])}, "classification_metrics": _classification_metrics(y, p), "calibration": _calibration(y, p, bins)}


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    if not len(labels):
        return {"roc_auc": None, "pr_auc": None, "logloss": None, "brier_score": None, "positive_rate": None}
    p = np.clip(np.asarray(probabilities, dtype=np.float64), np.finfo(np.float32).eps, 1 - np.finfo(np.float32).eps)
    has_both = len(np.unique(labels)) == 2
    return {"roc_auc": float(roc_auc_score(labels, p)) if has_both else None, "pr_auc": float(average_precision_score(labels, p)) if has_both else None, "logloss": float(log_loss(labels, p, labels=[0, 1])), "brier_score": float(np.mean(np.square(p - labels))), "positive_rate": float(np.mean(labels))}


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64); finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"rows": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {"rows": int(len(finite)), "min": float(finite.min()), "p50": float(np.quantile(finite, .50)), "p90": float(np.quantile(finite, .90)), "p95": float(np.quantile(finite, .95)), "p99": float(np.quantile(finite, .99)), "max": float(finite.max())}


def _calibration(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> dict[str, Any]:
    result: list[dict[str, float | int | None]] = []; total = len(labels); ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probabilities >= lower) & ((probabilities < upper) if index < bins - 1 else (probabilities <= upper))
        rows = int(mask.sum())
        if not rows:
            result.append({"bin": index, "lower": lower, "upper": upper, "rows": 0, "mean_predicted_pcvr": None, "actual_conversion_rate": None, "calibration_error": None})
            continue
        predicted = float(probabilities[mask].mean()); actual = float(labels[mask].mean()); error = abs(predicted - actual); ece += rows / total * error
        result.append({"bin": index, "lower": lower, "upper": upper, "rows": rows, "mean_predicted_pcvr": predicted, "actual_conversion_rate": actual, "calibration_error": error})
    return {"bin_count": bins, "ece": float(ece), "bins": result}


def _strict_holdout_metrics(connection: sqlite3.Connection) -> dict[str, Any]:
    queries = {
        "unseen_user": "NOT EXISTS (SELECT 1 FROM train_users t WHERE t.user_id=p.user_id)",
        "unseen_product": "NOT EXISTS (SELECT 1 FROM train_products t WHERE t.candidate_ad_id=p.candidate_ad_id)",
        "unseen_user_product_pair": "NOT EXISTS (SELECT 1 FROM train_pairs t WHERE t.user_id=p.user_id AND t.candidate_ad_id=p.candidate_ad_id)",
    }
    result: dict[str, Any] = {"definition": "These are stricter slices of the existing validation set. Only a separately trained temporal model is a strict future evaluation."}
    for name, where in queries.items():
        labels, probabilities = _prediction_query(connection, where)
        result[name] = {"rows": int(len(labels)), "metrics": _classification_metrics(labels, probabilities)}
    return result


def _prediction_query(connection: sqlite3.Connection, where: str) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []; probabilities: list[np.ndarray] = []
    cursor = connection.execute(f"SELECT label, pcvr FROM validation_predictions p WHERE {where}")
    while True:
        rows = cursor.fetchmany(100_000)
        if not rows:
            break
        data = np.asarray(rows, dtype=np.float64); labels.append(data[:, 0].astype(np.int8)); probabilities.append(data[:, 1].astype(np.float32))
    return (np.concatenate(labels) if labels else np.empty(0, dtype=np.int8), np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32))


def _feature_usage(train_directory: Path, validation_directory: Path) -> dict[str, Any]:
    names = [f"dense__{name}" for name in DENSE_FEATURES]
    totals = {name: {"nonzero": 0, "finite": 0} for name in names}; rows = 0
    for directory in (train_directory, validation_directory):
        for part in sorted(directory.glob("part-*.parquet")):
            for batch in pq.ParquetFile(part).iter_batches(batch_size=100_000, columns=names):
                data = batch.to_pydict(); rows += len(data[names[0]])
                for name in names:
                    values = np.asarray(data[name], dtype=np.float32); finite = np.isfinite(values)
                    totals[name]["finite"] += int(finite.sum()); totals[name]["nonzero"] += int(np.count_nonzero(values[finite]))
    return {"rows_scanned": rows, "dense_nonzero_rate": {name.removeprefix("dense__"): _divide(values["nonzero"], values["finite"]) for name, values in totals.items()}}


def _leakage_audit(config: FineRankConfig) -> dict[str, Any]:
    feature_columns = list(DENSE_FEATURES + SPARSE_FEATURES)
    try:
        assert_no_fine_rank_leakage(feature_columns); passed = True; error = None
    except ValueError as exception:
        passed = False; error = str(exception)
    temporal_contract: Mapping[str, Any] | None = None
    if config.mode == "temporal":
        metadata_path = dataset_spec(config).metadata_path
        if metadata_path.is_file():
            temporal_contract = json.loads(metadata_path.read_text(encoding="utf-8")).get("temporal_feature_semantics")
    result = {
        "passed_static_feature_guard": passed,
        "guard_error": error,
        "dense_features": list(DENSE_FEATURES),
        "sparse_features": list(SPARSE_FEATURES),
        "blocked_direct_columns": sorted(LEAKAGE_COLUMNS),
        "conversion_derived_features_in_model": [],
        "feature_provenance_risks": {
            "clicks_last_7d": "In temporal mode it is read from the Past-only user index and frozen at the Past cut-off. In full mode it is used directly from each source interaction, so source-data provenance must establish it was computed strictly before the click.",
            "coarse_score": "Present in the model schema, but observed-click training rows do not join coarse candidates; absent values encode to zero. The feature_usage section verifies whether it was nonzero in the cached training/validation data.",
            "rrf_score_and_source_count": "Present in the model schema, but observed-click training rows do not join recall candidates; absent values encode to zero. If later populated, their request-time generation and history cut-off must be audited before use.",
            "time_features": "click_hour_utc and click_day_of_week_utc are derived from click_timestamp. The raw timestamp itself is blocked, but these non-label calendar transforms are model inputs.",
        },
        "temporal_feature_contract": temporal_contract,
    }
    if config.mode == "temporal":
        result["temporal_candidate_requirement"] = "Inference candidates must be generated from a Past-only recall system and (if coarse_score is used) a model trained no later than Future-A before scoring Future-B. The temporal label cache itself strips all dynamic candidate-score columns from Future-A/B supervision rows."
    else:
        result["full_mode_index_warning"] = "The full-mode inference feature index uses latest source attributes per user/product. It is not used to encode full-mode observed-click training rows, but it is not a strict historical feature store for causal evaluation."
    return result


def _split_audit(config: FineRankConfig) -> dict[str, Any]:
    if config.mode == "temporal":
        return {"mode": "temporal", "definition": "Past constructs features/history; Future-A supplies train labels; Future-B supplies validation labels.", "strict_future_evaluation": True}
    return {"mode": "full", "definition": f"Each observed interaction is assigned by stable hash of user, product, timestamp/source and row offset; validation fraction={config.validation_fraction:.0%}.", "row_random_split": True, "group_disjoint": False, "strict_future_evaluation": False, "final_score_policy": "Do not treat this full-mode random-row validation score as the final project metric. Use a separately trained temporal cache/checkpoint (Past -> Future-A -> Future-B) for the final score."}


def _output_diagnostics(path: Path, chunk_size: int) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "fine-rank output file does not exist"}
    columns = ("pCVR", "predicted_conversion_value", "expected_value_score")
    values: dict[str, list[np.ndarray]] = {column: [] for column in columns}; rows = 0; high_90 = 0; high_99 = 0
    for chunk in pd.read_csv(path, usecols=["pCVR", "predicted_conversion_value", "expected_value_score", "rank"], chunksize=chunk_size, low_memory=False):
        rows += len(chunk)
        p = pd.to_numeric(chunk["pCVR"], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(p).all():
            raise FloatingPointError("Fine-rank output audit found non-finite pCVR")
        high_90 += int((p > .9).sum()); high_99 += int((p > .99).sum())
        values["pCVR"].append(p)
        for column in columns[1:]:
            current = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=np.float32)
            if not np.isfinite(current).all():
                raise FloatingPointError(f"Fine-rank output audit found non-finite {column}")
            values[column].append(current)
    combined = {column: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32) for column, parts in values.items()}
    value_p99 = float(np.quantile(combined["predicted_conversion_value"], .99)) if rows else None
    top1_rows = 0; top1_above_value_p99 = 0
    if value_p99 is not None:
        for chunk in pd.read_csv(path, usecols=["predicted_conversion_value", "rank"], chunksize=chunk_size, low_memory=False):
            rank = pd.to_numeric(chunk["rank"], errors="coerce").to_numpy(); predicted = pd.to_numeric(chunk["predicted_conversion_value"], errors="coerce").to_numpy(dtype=np.float32)
            top1 = rank == 1; top1_rows += int(top1.sum()); top1_above_value_p99 += int((predicted[top1] > value_p99).sum())
    return {"available": True, "path": str(path), "rows": rows, "pCVR": _distribution(combined["pCVR"]), "predicted_conversion_value": _distribution(combined["predicted_conversion_value"]), "expected_value_score": _distribution(combined["expected_value_score"]), "pCVR_gt_0_9_rate": _divide(high_90, rows), "pCVR_gt_0_99_rate": _divide(high_99, rows), "predicted_value_p99_threshold": value_p99, "top1_rows": top1_rows, "top1_predicted_value_gt_global_p99_rate": _divide(top1_above_value_p99, top1_rows)}


def run_id_memorization_ablation(config: FineRankConfig, audit_config: FineRankAuditConfig) -> dict[str, Any]:
    """Train four small temporary CVR-only models; production weights stay untouched."""
    transform = _value_transform(json.loads(dataset_spec(config).metadata_path.read_text(encoding="utf-8")))
    train = _sample_encoded_rows(dataset_spec(config).cache_dir, audit_config.ablation_train_rows, transform)
    validation = _sample_encoded_rows(dataset_spec(config).validation_dir, audit_config.ablation_validation_rows, transform)
    device = resolve_device(config.device); torch.manual_seed(audit_config.random_seed)
    variants = {"A_all_features": (), "B_without_user_id": ("user_id",), "C_without_product_id": ("product_id",), "D_without_user_id_and_product_id": ("user_id", "product_id")}
    result: dict[str, Any] = {"ran": True, "temporary": True, "objective": "CVR-only small-sample comparison; no checkpoint is written", "sample_rows": {"train": len(train["label"]), "validation": len(validation["label"])}, "epochs": audit_config.ablation_epochs, "variants": {}}
    feature_index = {name: index for index, name in enumerate(SPARSE_FEATURES)}
    for name, removed in variants.items():
        sparse_train = train["sparse"].copy(); sparse_validation = validation["sparse"].copy()
        for feature in removed:
            sparse_train[:, feature_index[feature]] = 0; sparse_validation[:, feature_index[feature]] = 0
        model = build_model(config).to(device); optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        dataset = TensorDataset(torch.from_numpy(train["dense"]), torch.from_numpy(sparse_train), torch.from_numpy(train["label"]))
        loader = DataLoader(dataset, batch_size=audit_config.ablation_batch_size, shuffle=True)
        model.train()
        for _ in range(audit_config.ablation_epochs):
            for dense, sparse, labels in loader:
                optimizer.zero_grad(set_to_none=True); logits, _ = model(dense.to(device), sparse.to(device)); loss = F.binary_cross_entropy_with_logits(logits, labels.to(device)); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = model.predict(torch.from_numpy(validation["dense"]).to(device), torch.from_numpy(sparse_validation).to(device))[0].detach().to(dtype=torch.float32).cpu().numpy()
        result["variants"][name] = _classification_metrics(validation["label"].astype(np.int8), probabilities)
    return result


def _sample_encoded_rows(directory: Path, maximum: int, transform: Mapping[str, float]) -> dict[str, np.ndarray]:
    dense_columns = [f"dense__{name}" for name in DENSE_FEATURES]; sparse_columns = [f"sparse__{name}" for name in SPARSE_FEATURES]; columns = [*dense_columns, *sparse_columns, "conversion_label"]
    dense_parts: list[np.ndarray] = []; sparse_parts: list[np.ndarray] = []; labels: list[np.ndarray] = []; rows = 0
    for part in sorted(directory.glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=min(100_000, maximum - rows), columns=columns):
            data = batch.to_pydict(); size = len(data["conversion_label"]); dense_parts.append(np.column_stack([data[column] for column in dense_columns]).astype(np.float32)); sparse_parts.append(np.column_stack([data[column] for column in sparse_columns]).astype(np.int64)); labels.append(np.asarray(data["conversion_label"], dtype=np.float32)); rows += size
            if rows >= maximum:
                break
        if rows >= maximum:
            break
    if not rows:
        raise ValueError(f"No rows available for small ablation sample: {directory}")
    return {"dense": np.concatenate(dense_parts), "sparse": np.concatenate(sparse_parts), "label": np.concatenate(labels)}


def _recommendations(report: Mapping[str, Any]) -> list[str]:
    overlap = report["train_validation_overlap"]["exact_user_product_pair_overlap"]["validation_row_rate"]
    recommendations = ["Use the report's ROC-AUC, PR-AUC, LogLoss, Brier score, calibration/ECE and distribution jointly; PR-AUC alone is not a sufficient quality claim.", "Do not clip high pCVR merely because it looks suspicious. First use overlap, strict-slice and ID-ablation results to identify the cause.", "For a final causal score, train and evaluate a separate temporal fine-rank artifact with Past-only features, Future-A labels and Future-B validation labels."]
    if overlap > .01:
        recommendations.insert(0, f"Validation has {overlap:.2%} row-weighted exact user-product overlap with train. Treat random-row metrics as potentially memorization-assisted until the unseen-pair and ablation results are reviewed.")
    return recommendations


def _render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["classification_metrics"]; overlap = report["train_validation_overlap"]; split = report["split_audit"]
    lines = ["# Fine Rank Effect Audit", "", f"- Mode: `{report['mode']}`", f"- Validation rows: {report['rows']['validation']}", f"- ROC-AUC: {metrics['roc_auc']}", f"- PR-AUC: {metrics['pr_auc']}", f"- LogLoss: {metrics['logloss']}", f"- Brier score: {metrics['brier_score']}", f"- ECE: {report['calibration']['ece']}", f"- Exact pair overlap (validation rows): {overlap['exact_user_product_pair_overlap']['validation_row_rate']:.4%}", "", "## Split conclusion", "", split.get("final_score_policy", split["definition"]), "", "## Recommendations", ""]
    lines.extend(f"- {item}" for item in report["recommendations"])
    return "\n".join(lines) + "\n"


def _write_full_vs_temporal_report(temporal_report: Mapping[str, Any], audit_config: FineRankAuditConfig) -> Path:
    """Compare the IID full reference with temporal final metrics, never rank them equally."""
    path = audit_config.output_path.parent / "fine_rank_full_vs_temporal.json"
    full: Mapping[str, Any] | None = None
    if audit_config.full_reference_path is not None and audit_config.full_reference_path.is_file():
        loaded = json.loads(audit_config.full_reference_path.read_text(encoding="utf-8"))
        if loaded.get("mode") == "full":
            full = loaded
    names = ("roc_auc", "pr_auc", "logloss", "brier_score", "positive_rate")
    result = {
        "policy": "Temporal validation is the final generalization metric. Full random-row validation is retained only as an IID/offline upper-bound reference.",
        "full_reference_available": full is not None,
        "metrics": {name: {"full_iid_upper_bound": full.get("classification_metrics", {}).get(name) if full else None, "temporal_final": temporal_report["classification_metrics"].get(name)} for name in names},
        "temporal_ece": temporal_report["calibration"].get("ece"),
        "full_ece": full.get("calibration", {}).get("ece") if full else None,
        "temporal_report": temporal_report.get("checkpoint"),
        "full_report_path": str(audit_config.full_reference_path) if audit_config.full_reference_path else None,
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown = path.with_suffix(".md")
    lines = ["# Fine Rank: Full vs Temporal", "", result["policy"], "", "| Metric | Full IID reference | Temporal final |", "| --- | ---: | ---: |"]
    lines.extend(f"| {name} | {values['full_iid_upper_bound']} | {values['temporal_final']} |" for name, values in result["metrics"].items())
    lines.extend([f"| ECE | {result['full_ece']} | {result['temporal_ece']} |", ""])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return path


def _count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0
