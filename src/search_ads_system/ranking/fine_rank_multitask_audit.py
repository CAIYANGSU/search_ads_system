"""Leakage and memorization audit for Search Conversion fine ranking.

Only Past and Future-A are opened. Future-B is deliberately represented only
by its path in the report, so accidental use is structurally avoided.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from search_ads_system.common.config import resolve_path
from search_ads_system.data.storage import iter_csv_parts
from search_ads_system.ranking.dcnv2 import DCNv2MultiTask
from search_ads_system.ranking.fine_rank_dataset import DENSE_FEATURES, SPARSE_FEATURES, encode_feature_frame
from search_ads_system.ranking.fine_rank_multitask import (
    ClickInteractionDataset, FineRankMultiTaskConfig, _device, _seed,
    _to_device, multitask_loss,
)

FORBIDDEN_FEATURES = frozenset({
    "conversion_label", "has_conversion_value", "conversion_value_eur", "conversion_timestamp",
    "conversion_delay_seconds", "conversion_delay_hours", "future_label", "future_timestamp",
    "post_conversion", "time_delay_for_conversion",
})
UPSTREAM_SCORE_FEATURES = frozenset({"rrf_score", "source_count", "coarse_score", "inverse_coarse_rank"})


@dataclass(frozen=True)
class FineRankMultiTaskAuditConfig:
    output_dir: Path
    train_rows: int | None = None
    validation_rows: int | None = None
    diagnostic_rows: int | None = None
    epochs: int = 3
    batch_size: int = 8192
    patience: int = 2


def parse_fine_rank_multitask_audit_config(raw: Mapping[str, Any], config_path: Path, *, stage: str) -> FineRankMultiTaskAuditConfig:
    root = config_path.parent.resolve(); options = raw.get("fine_rank_multitask", {})
    if not isinstance(options, Mapping):
        raise ValueError("fine_rank_multitask must be a mapping")
    audit = options.get("audit", {})
    if not isinstance(audit, Mapping):
        raise ValueError("fine_rank_multitask.audit must be a mapping")
    effective = {**audit, **(audit.get("sanity", {}) if stage == "sanity" else {})}
    return FineRankMultiTaskAuditConfig(
        output_dir=resolve_path(str(effective.get("output_dir", "outputs/fine_rank/audit")), root),
        train_rows=_optional_int(effective.get("max_train_rows")), validation_rows=_optional_int(effective.get("max_validation_rows")),
        diagnostic_rows=_optional_int(effective.get("max_diagnostic_rows")), epochs=int(effective.get("epochs", options.get("epochs", 3))),
        batch_size=int(effective.get("batch_size", options.get("batch_size", 8192))), patience=int(effective.get("patience", options.get("patience", 2))),
    )


class _SelectedFeatures(IterableDataset[dict[str, Any]]):
    def __init__(self, source: ClickInteractionDataset, dense_indices: list[int], sparse_indices: list[int], *, identifiers: bool = False) -> None:
        self.source, self.dense_indices, self.sparse_indices, self.identifiers = source, dense_indices, sparse_indices, identifiers

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for row in self.source:
            item = dict(row)
            item["dense"] = item["dense"][self.dense_indices]
            item["sparse"] = item["sparse"][self.sparse_indices]
            if not self.identifiers:
                item.pop("user_id", None); item.pop("product_id", None)
            yield item


def actual_model_input_features() -> list[str]:
    return [*DENSE_FEATURES, *SPARSE_FEATURES]


def forbidden_feature_intersection(features: Iterable[str]) -> list[str]:
    return sorted(set(features) & FORBIDDEN_FEATURES)


def ablation_features(name: str) -> tuple[list[str], list[str]]:
    dense, sparse = list(DENSE_FEATURES), list(SPARSE_FEATURES)
    remove: set[str] = set()
    if name == "no_recall_coarse_scores": remove = set(UPSTREAM_SCORE_FEATURES)
    elif name == "no_user_id": remove = {"user_id"}
    elif name == "no_product_id": remove = {"product_id"}
    elif name == "no_user_product_id": remove = {"user_id", "product_id"}
    elif name == "no_ids_no_upstream_scores": remove = {"user_id", "product_id", *UPSTREAM_SCORE_FEATURES}
    elif name != "full_features": raise ValueError(f"unknown ablation {name}")
    return [value for value in dense if value not in remove], [value for value in sparse if value not in remove]


def run_fine_rank_multitask_audit(config: FineRankMultiTaskConfig, audit: FineRankMultiTaskAuditConfig, *, stage: str) -> dict[str, Any]:
    """Run six DCNv2 probes and independent data-integrity diagnostics."""
    _assert_audit_temporal_contract(config)
    inputs = actual_model_input_features(); forbidden = forbidden_feature_intersection(inputs)
    if forbidden:
        raise ValueError(f"AUDIT FAILED: forbidden model inputs: {forbidden}")
    _seed(config.seed); device = _device(config.device)
    with tempfile.TemporaryDirectory(prefix="fine-rank-audit-") as directory:
        database = sqlite3.connect(Path(directory) / "overlap.sqlite")
        try:
            temporal_overlap = _overlap_audit(database, config)
            labels = _label_distribution_and_slices(database, config, audit)
            features = _feature_diagnostics(config, audit)
            ablation: dict[str, Any] = {}
            full_state: dict[str, torch.Tensor] | None = None
            for name in ("full_features", "no_recall_coarse_scores", "no_user_id", "no_product_id", "no_user_product_id", "no_ids_no_upstream_scores"):
                result, state = _run_ablation(name, config, audit, device)
                ablation[name] = result
                if name == "full_features": full_state = state
            assert full_state is not None
            slices = _seen_unseen_slices(full_state, database, config, audit, device)
        finally:
            database.close()
    flags, conclusion = _conclusion(forbidden, temporal_overlap, ablation, slices)
    report = {
        "stage": stage, "temporal_overlap": temporal_overlap, "label_distribution": labels,
        "feature_diagnostics": features, "actual_model_input_features": inputs,
        "forbidden_feature_intersection": forbidden, "ablation": ablation, "seen_unseen_slices": slices,
        "future_b_read_for_audit": False, "diagnostic_flags": flags, "conclusion": conclusion,
        "audit_contract": {"past": str(config.past_path), "future_a": str(config.future_a_path), "future_b_path_only": str(config.future_b_path), "device": str(device), "amp": bool(config.amp and device.type == "cuda")},
    }
    audit.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit.output_dir / "fine_rank_audit.json"; md_path = audit.output_dir / "fine_rank_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"report_path": str(json_path), "markdown_path": str(md_path), "future_b_read_for_audit": False}


def _assert_audit_temporal_contract(config: FineRankMultiTaskConfig) -> None:
    if not config.past_path.is_dir() or not config.future_a_path.is_dir() or not config.future_b_path.is_dir():
        raise FileNotFoundError("Audit requires existing Search Conversion Past/Future-A/Future-B directories")
    past, future_a = _bounds(config.past_path, config.chunk_size), _bounds(config.future_a_path, config.chunk_size)
    if past[0] is None or future_a[0] is None or not past[1] < future_a[0]:
        raise ValueError(f"Audit temporal contract requires Past < Future-A, got past={past}, future_a={future_a}")


def _bounds(path: Path, chunk_size: int) -> tuple[int | None, int | None]:
    low = high = None
    for frame in iter_csv_parts(path, chunk_size):
        values = pd.to_numeric(frame.get("click_timestamp"), errors="coerce").dropna()
        if len(values): low = int(values.min()) if low is None else min(low, int(values.min())); high = int(values.max()) if high is None else max(high, int(values.max()))
    return low, high


def _overlap_audit(connection: sqlite3.Connection, config: FineRankMultiTaskConfig) -> dict[str, Any]:
    connection.executescript("""
    CREATE TABLE past_rows (key TEXT PRIMARY KEY); CREATE TABLE future_rows (key TEXT PRIMARY KEY);
    CREATE TABLE past_events (key TEXT PRIMARY KEY); CREATE TABLE future_events (key TEXT PRIMARY KEY);
    CREATE TABLE past_users (key TEXT PRIMARY KEY); CREATE TABLE future_users (key TEXT PRIMARY KEY);
    CREATE TABLE past_products (key TEXT PRIMARY KEY); CREATE TABLE future_products (key TEXT PRIMARY KEY);
    CREATE TABLE past_pairs (user_id TEXT, product_id TEXT, rows INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id,product_id));
    CREATE TABLE future_pairs (user_id TEXT, product_id TEXT, rows INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id,product_id));
    """)
    summary = {"past": _insert_overlap_rows(connection, config.past_path, "past", config.chunk_size), "future_a": _insert_overlap_rows(connection, config.future_a_path, "future", config.chunk_size)}
    connection.commit()
    for name, prefix in (("past", "past"), ("future_a", "future")):
        if summary[name]["events"] != int(connection.execute(f"SELECT COUNT(*) FROM {prefix}_events").fetchone()[0]):
            summary[name]["event_reliable"] = False
    future_rows = summary["future_a"]["rows"]
    counts = lambda query: int(connection.execute(query).fetchone()[0])
    pair_seen_rows = counts("SELECT COALESCE(SUM(f.rows),0) FROM future_pairs f JOIN past_pairs p USING(user_id,product_id)")
    future_pairs = counts("SELECT COUNT(*) FROM future_pairs")
    past_pairs = counts("SELECT COUNT(*) FROM past_pairs")
    result = {
        "past_rows": summary["past"]["rows"], "future_a_rows": future_rows,
        "exact_row_overlap": _count_fraction(counts("SELECT COUNT(*) FROM future_rows f JOIN past_rows p USING(key)"), future_rows),
        "event_id_overlap": _count_fraction(counts("SELECT COUNT(*) FROM future_events f JOIN past_events p USING(key)"), summary["future_a"]["events"]) if summary["past"]["event_reliable"] and summary["future_a"]["event_reliable"] else {"available": False, "reason": "event_id missing or non-unique in one window"},
        "user_id_overlap": _count_fraction(counts("SELECT COUNT(*) FROM future_users f JOIN past_users p USING(key)"), counts("SELECT COUNT(*) FROM future_users")),
        "product_id_overlap": _count_fraction(counts("SELECT COUNT(*) FROM future_products f JOIN past_products p USING(key)"), counts("SELECT COUNT(*) FROM future_products")),
        "user_product_pair_overlap": {**_count_fraction(counts("SELECT COUNT(*) FROM future_pairs f JOIN past_pairs p USING(user_id,product_id)"), future_pairs), "future_a_row_fraction": _fraction(pair_seen_rows, future_rows)},
        "repeated_pair_fraction": {"past": _fraction(summary["past"]["rows"] - past_pairs, summary["past"]["rows"]), "future_a": _fraction(future_rows - future_pairs, future_rows)},
        "future_a_seen_user_fraction": _fraction(counts("SELECT COUNT(*) FROM future_users f JOIN past_users p USING(key)"), counts("SELECT COUNT(*) FROM future_users")),
        "future_a_seen_product_fraction": _fraction(counts("SELECT COUNT(*) FROM future_products f JOIN past_products p USING(key)"), counts("SELECT COUNT(*) FROM future_products")),
        "future_a_seen_user_product_pair_fraction": _fraction(pair_seen_rows, future_rows),
    }
    return result


def _insert_overlap_rows(connection: sqlite3.Connection, path: Path, prefix: str, chunk_size: int) -> dict[str, Any]:
    rows = events = 0; event_reliable = True
    for frame in iter_csv_parts(path, chunk_size):
        required = {"user_id", "product_id"}
        if required - set(frame): raise ValueError(f"audit source missing {sorted(required-set(frame))}")
        rows += len(frame); keys = _row_hashes(frame)
        connection.executemany(f"INSERT OR IGNORE INTO {prefix}_rows VALUES (?)", ((key,) for key in keys))
        users = frame.user_id.astype("string").fillna("").str.strip(); products = frame.product_id.astype("string").fillna("").str.strip()
        connection.executemany(f"INSERT OR IGNORE INTO {prefix}_users VALUES (?)", ((value,) for value in users if value))
        connection.executemany(f"INSERT OR IGNORE INTO {prefix}_products VALUES (?)", ((value,) for value in products if value))
        connection.executemany(f"INSERT INTO {prefix}_pairs(user_id,product_id,rows) VALUES (?,?,1) ON CONFLICT(user_id,product_id) DO UPDATE SET rows=rows+1", ((u,p) for u,p in zip(users,products) if u and p))
        if "event_id" not in frame: event_reliable = False
        else:
            ids = frame.event_id.astype("string").fillna("").str.strip(); events += int((ids != "").sum())
            if ids.eq("").any() or ids.duplicated().any(): event_reliable = False
            connection.executemany(f"INSERT OR IGNORE INTO {prefix}_events VALUES (?)", ((value,) for value in ids if value))
    return {"rows": rows, "events": events, "event_reliable": event_reliable}


def _row_hashes(frame: pd.DataFrame) -> list[str]:
    canonical = frame.reindex(sorted(frame.columns), axis=1).fillna("<NA>").astype(str)
    return [hashlib.sha256("\x1f".join(row).encode()).hexdigest() for row in canonical.itertuples(index=False, name=None)]


def _label_distribution_and_slices(connection: sqlite3.Connection, config: FineRankMultiTaskConfig, audit: FineRankMultiTaskAuditConfig) -> dict[str, Any]:
    result = {"past": _label_summary(config.past_path, config.chunk_size, None), "future_a": _label_summary(config.future_a_path, config.chunk_size, None)}
    labels: dict[str, list[int]] = {"seen_pair": [], "unseen_pair": []}
    for frame in _iter_limited(config.future_a_path, config.chunk_size, audit.diagnostic_rows):
        y = pd.to_numeric(frame.conversion_label, errors="coerce").isin((1,)).astype(int)
        pairs = list(zip(frame.user_id.astype(str), frame.product_id.astype(str)))
        known = _known_pairs(connection, pairs)
        for value, seen in zip(y, known): labels["seen_pair" if seen else "unseen_pair"].append(int(value))
    result["future_a_by_pair"] = {name: _label_only(values) for name, values in labels.items()}
    result["future_a_by_pair"]["sampled_rows"] = sum(len(values) for values in labels.values())
    return result


def _label_summary(path: Path, chunk_size: int, limit: int | None) -> dict[str, Any]:
    labels: list[int] = []; values: list[float] = []; declared = 0; rows = 0
    for frame in _iter_limited(path, chunk_size, limit):
        y = pd.to_numeric(frame.get("conversion_label"), errors="coerce"); valid = y.isin((0, 1)); y = y[valid].astype(int); rows += len(y); labels.extend(y.tolist())
        raw = pd.to_numeric(frame.get("conversion_value_eur"), errors="coerce"); mask = valid & y.reindex(frame.index, fill_value=0).eq(1) & np.isfinite(raw) & raw.ge(0)
        if "has_conversion_value" in frame: mask &= pd.to_numeric(frame.has_conversion_value, errors="coerce").eq(1); declared += int(pd.to_numeric(frame.has_conversion_value, errors="coerce").eq(1).sum())
        values.extend(raw[mask].astype(float).tolist())
    array = np.asarray(values, dtype=float)
    return {"rows": rows, "conversion_positive_count": int(sum(labels)), "conversion_rate": _fraction(sum(labels), rows), "has_conversion_value_count": int(len(values) if not declared else declared), "has_conversion_value_rate": _fraction(len(values) if not declared else declared, rows), "conversion_value": _value_stats(array)}


def _feature_diagnostics(config: FineRankMultiTaskConfig, audit: FineRankMultiTaskAuditConfig) -> dict[str, Any]:
    samples: list[pd.DataFrame] = []
    seen = 0
    for frame in _iter_limited(config.future_a_path, config.chunk_size, audit.diagnostic_rows):
        frame = frame.copy(); frame["candidate_ad_id"] = frame.product_id
        encoded = encode_feature_frame(frame, bucket_sizes=config.bucket_sizes, random_seed=config.seed)
        samples.append(encoded); seen += len(encoded)
    data = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    result: dict[str, Any] = {"rows_used": seen, "source": "Future-A diagnostic only; no feature was changed using this result.", "features": {}}
    for name in DENSE_FEATURES:
        values = data.get(f"dense__{name}", pd.Series(dtype=float)).to_numpy(dtype=float); labels = data.get("conversion_label", pd.Series(dtype=float)).to_numpy(dtype=int)
        positive, negative = values[labels == 1], values[labels == 0]
        result["features"][name] = {"positive_mean": float(positive.mean()) if len(positive) else None, "negative_mean": float(negative.mean()) if len(negative) else None, "single_feature_roc_auc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else None, "bins": _feature_bins(values, labels)}
    return result


def _run_ablation(name: str, config: FineRankMultiTaskConfig, audit: FineRankMultiTaskAuditConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    dense_names, sparse_names = ablation_features(name); di = [DENSE_FEATURES.index(value) for value in dense_names]; si = [SPARSE_FEATURES.index(value) for value in sparse_names]
    model = DCNv2MultiTask(dense_dim=len(di), sparse_bucket_sizes=[config.bucket_sizes[index] for index in si], embedding_dim=config.embedding_dim, hidden_dims=config.hidden_dims, num_cross_layers=config.cross_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay); scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    best_pr, stale, best_state, best_epoch = float("-inf"), 0, None, 0
    for epoch in range(1, audit.epochs + 1):
        model.train()
        for batch in _selected_loader(config.past_path, config, audit.train_rows, di, si):
            dense, sparse, label, log_value, mask, _ = _to_device(batch, device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits, value = model(dense, sparse); loss, _, _ = multitask_loss(logits, value, label, log_value, mask, lambda_cvr=config.lambda_cvr, lambda_value=config.lambda_value)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update()
        metrics = _evaluate_selected(model, config, audit.validation_rows, di, si, device)
        score = metrics["pr_auc"] if metrics["pr_auc"] is not None else float("-inf")
        if score > best_pr: best_pr, stale, best_epoch, best_state = score, 0, epoch, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale > audit.patience: break
    assert best_state is not None; model.load_state_dict(best_state)
    metrics = _evaluate_selected(model, config, audit.validation_rows, di, si, device)
    metrics.pop("_labels", None); metrics.pop("_predictions", None)
    result = {"dense_features": dense_names, "sparse_features": sparse_names, "parameter_count": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch, **metrics}
    return result, best_state


@torch.no_grad()
def _evaluate_selected(model: torch.nn.Module, config: FineRankMultiTaskConfig, limit: int | None, di: list[int], si: list[int], device: torch.device, *, database: sqlite3.Connection | None = None) -> dict[str, Any]:
    model.eval(); ys: list[np.ndarray] = []; ps: list[np.ndarray] = []; ids: list[tuple[str, str]] = []
    for batch in _selected_loader(config.future_a_path, config, limit, di, si, identifiers=database is not None):
        dense, sparse, label, _, _, _ = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"): logits, _ = model(dense, sparse)
        ys.append(label.cpu().numpy()); ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        if database is not None: ids.extend(zip(batch["user_id"], batch["product_id"]))
    y = np.concatenate(ys) if ys else np.empty(0); p = np.concatenate(ps) if ps else np.empty(0)
    return _classification(y, p) | ({"identifiers": ids} if database is not None else {})


def _seen_unseen_slices(state: dict[str, torch.Tensor], connection: sqlite3.Connection, config: FineRankMultiTaskConfig, audit: FineRankMultiTaskAuditConfig, device: torch.device) -> dict[str, Any]:
    model = DCNv2MultiTask(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=config.bucket_sizes, embedding_dim=config.embedding_dim, hidden_dims=config.hidden_dims, num_cross_layers=config.cross_layers).to(device); model.load_state_dict(state)
    metrics = _evaluate_selected(model, config, audit.validation_rows, list(range(len(DENSE_FEATURES))), list(range(len(SPARSE_FEATURES))), device, database=connection)
    ids = metrics.pop("identifiers"); labels = np.asarray(metrics.pop("_labels")); probs = np.asarray(metrics.pop("_predictions")); users = _known_values(connection, "past_users", [item[0] for item in ids]); products = _known_values(connection, "past_products", [item[1] for item in ids]); pairs = _known_pairs(connection, ids)
    masks = {"seen_user_seen_product_seen_pair": np.asarray([u and p and pair for u,p,pair in zip(users,products,pairs)]), "seen_user_seen_product_unseen_pair": np.asarray([u and p and not pair for u,p,pair in zip(users,products,pairs)]), "unseen_user": ~np.asarray(users), "unseen_product": ~np.asarray(products)}
    result = {name: _classification(labels[mask], probs[mask]) for name, mask in masks.items()}
    for value in result.values():
        value.pop("_labels", None); value.pop("_predictions", None)
    return result


def _selected_loader(path: Path, config: FineRankMultiTaskConfig, limit: int | None, di: list[int], si: list[int], *, identifiers: bool = False) -> DataLoader[Any]:
    source = ClickInteractionDataset(path, config, max_rows=limit, include_identifiers=identifiers)
    return DataLoader(_SelectedFeatures(source, di, si, identifiers=identifiers), batch_size=config.batch_size, num_workers=0, pin_memory=config.device != "cpu")


def _classification(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    both = len(np.unique(y)) == 2
    return {"rows": int(len(y)), "positive_count": int(y.sum()) if len(y) else 0, "cvr": float(y.mean()) if len(y) else None, "roc_auc": float(roc_auc_score(y,p)) if both else None, "pr_auc": float(average_precision_score(y,p)) if both else None, "logloss": float(log_loss(y,p,labels=[0,1])) if len(y) else None, "brier_score": float(np.mean((p-y)**2)) if len(y) else None, "prediction_mean": float(p.mean()) if len(p) else None, "label_mean": float(y.mean()) if len(y) else None, "_labels": y, "_predictions": p}


def _known_values(connection: sqlite3.Connection, table: str, values: list[str]) -> list[bool]:
    found: set[str] = set()
    for offset in range(0, len(values), 900):
        group = values[offset:offset+900]
        found.update(row[0] for row in connection.execute(f"SELECT key FROM {table} WHERE key IN ({','.join('?' for _ in group)})", group))
    return [value in found for value in values]


def _known_pairs(connection: sqlite3.Connection, pairs: list[tuple[str, str]]) -> list[bool]:
    found: set[tuple[str, str]] = set()
    for offset in range(0, len(pairs), 400):
        group = pairs[offset:offset+400]; clause = " OR ".join("(user_id=? AND product_id=?)" for _ in group)
        found.update((row[0], row[1]) for row in connection.execute(f"SELECT user_id,product_id FROM past_pairs WHERE {clause}", [item for pair in group for item in pair]))
    return [pair in found for pair in pairs]


def _iter_limited(path: Path, chunk_size: int, limit: int | None) -> Iterable[pd.DataFrame]:
    rows = 0
    for frame in iter_csv_parts(path, chunk_size):
        if limit is not None:
            frame = frame.iloc[:max(0, limit-rows)];
            if frame.empty: return
        rows += len(frame); yield frame
        if limit is not None and rows >= limit: return


def _feature_bins(values: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    if not len(values): return []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (values >= lower) & ((values < upper) if upper != edges[-1] else (values <= upper))
        result.append({"lower": float(lower), "upper": float(upper), "rows": int(mask.sum()), "conversion_rate": float(labels[mask].mean()) if mask.any() else None})
    return result


def _value_stats(values: np.ndarray) -> dict[str, Any]:
    return {"mean": float(values.mean()) if len(values) else None, "median": float(np.median(values)) if len(values) else None, "p90": float(np.quantile(values,.9)) if len(values) else None, "p99": float(np.quantile(values,.99)) if len(values) else None}


def _label_only(values: list[int]) -> dict[str, Any]: return {"rows": len(values), "positive_count": int(sum(values)), "conversion_rate": _fraction(sum(values),len(values))}
def _fraction(a: int | float, b: int | float) -> float | None: return float(a/b) if b else None
def _count_fraction(count: int, denominator: int) -> dict[str, Any]: return {"count": count, "fraction": _fraction(count, denominator)}
def _optional_int(value: Any) -> int | None: return None if value is None else int(value)


def _conclusion(forbidden: list[str], overlap: Mapping[str, Any], ablation: Mapping[str, Any], slices: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    flags: list[str] = []; findings: list[str] = []
    if forbidden: flags.append("direct_leakage_detected"); findings.append("Forbidden target/post-conversion fields are model inputs.")
    if overlap["exact_row_overlap"]["count"] or (overlap["event_id_overlap"].get("count", 0) if overlap["event_id_overlap"].get("available", True) else 0): flags.append("temporal_overlap_concern"); findings.append("Exact rows or reliable event IDs overlap across Past and Future-A.")
    full = ablation["full_features"].get("pr_auc") or 0.0; no_ids = ablation["no_user_product_id"].get("pr_auc") or 0.0; no_scores = ablation["no_recall_coarse_scores"].get("pr_auc") or 0.0
    if full - no_ids >= .05: flags.append("strong_id_memorization_signal"); findings.append("Removing both user and product IDs materially reduces PR-AUC.")
    if full - no_scores >= .05: flags.append("strong_upstream_score_signal"); findings.append("Removing recall/coarse scores materially reduces PR-AUC.")
    seen = slices.get("seen_user_seen_product_seen_pair", {}).get("pr_auc"); unseen = slices.get("seen_user_seen_product_unseen_pair", {}).get("pr_auc")
    if seen is not None and unseen is not None and seen - unseen >= .05: flags.append("repeated_pair_memorization_signal"); findings.append("Seen user-product pairs outperform unseen pairs by a material PR-AUC margin.")
    if not findings: findings.append("No direct leakage or large audit-defined ablation/slice gap was detected; high scores may reflect strong legitimate clicked-interaction structure, subject to slice coverage.")
    return flags, {"summary": " ".join(findings), "evidence_based_findings": findings, "thresholds": {"material_pr_auc_drop": 0.05}}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Fine Rank leakage & memorization audit", "", "- Future-B read for audit: `false`.", f"- Forbidden input intersection: `{report['forbidden_feature_intersection']}`.", "", "## Ablations", "", "| Variant | PR-AUC | ROC-AUC | LogLoss | Brier | Best epoch |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in report["ablation"].items(): lines.append(f"| {name} | {_fmt(value['pr_auc'])} | {_fmt(value['roc_auc'])} | {_fmt(value['logloss'])} | {_fmt(value['brier_score'])} | {value['best_epoch']} |")
    lines.extend(("", "## Seen/unseen slices", "", "| Slice | Rows | PR-AUC | CVR |", "| --- | ---: | ---: | ---: |"))
    for name, value in report["seen_unseen_slices"].items(): lines.append(f"| {name} | {value['rows']} | {_fmt(value['pr_auc'])} | {_fmt(value['cvr'])} |")
    lines.extend(("", "## Conclusion", "", report["conclusion"]["summary"], "")); return "\n".join(lines)
def _fmt(value: Any) -> str: return "unavailable" if value is None else f"{float(value):.6f}"
