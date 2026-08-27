"""Future-A-only standalone predictions from the selected Fine Rank checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch

from search_ads_system.data.storage import iter_csv_parts, write_csv_part
from search_ads_system.ranking.fine_rank_dataset import DENSE_FEATURES, SPARSE_FEATURES, encode_feature_frame
from search_ads_system.ranking.fine_rank_multitask import FineRankMultiTaskConfig, _device, build_model, decode_predictions


def checkpoint_config(checkpoint_path: Path, fallback: FineRankMultiTaskConfig) -> FineRankMultiTaskConfig:
    """Recover architecture/hash settings without touching any split content."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = checkpoint.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("Fine Rank checkpoint lacks serialized configuration")
    fields = {"embedding_dim", "hidden_dims", "cross_layers", "bucket_sizes", "seed"}
    values: dict[str, Any] = {}
    for name in fields:
        if name in saved:
            value = saved[name]
            values[name] = tuple(value) if name in {"hidden_dims", "bucket_sizes"} else value
    return replace(fallback, **values)


def write_future_a_predictions(config: FineRankMultiTaskConfig, *, checkpoint_path: Path, output_dir: Path, max_rows: int | None = None) -> dict[str, Any]:
    """Infer on Search Conversion Future-A only; Future-B is never opened."""
    if not config.future_a_path.is_dir():
        raise FileNotFoundError(f"Future-A source missing: {config.future_a_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected DCNv2 checkpoint missing: {checkpoint_path}")
    effective = checkpoint_config(checkpoint_path, config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model") != "dcnv2":
        raise ValueError("Standalone value prediction requires the selected DCNv2 checkpoint")
    device = _device(effective.device); model = build_model("dcnv2", effective).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for part in output_dir.glob("part-*.csv"): part.unlink()
    rows = parts = 0; time_min = time_max = None
    with torch.no_grad():
        for frame in _future_a_frames(effective.future_a_path, effective.chunk_size, max_rows):
            encoded, original = _encode_for_prediction(frame, effective)
            if encoded.empty: continue
            dense = torch.from_numpy(encoded[[f"dense__{name}" for name in DENSE_FEATURES]].to_numpy(np.float32))
            sparse = torch.from_numpy(encoded[[f"sparse__{name}" for name in SPARSE_FEATURES]].to_numpy(np.int64))
            pcvr_parts: list[np.ndarray] = []; value_parts: list[np.ndarray] = []
            for start in range(0, len(encoded), effective.batch_size):
                with torch.amp.autocast(device_type=device.type, enabled=effective.amp and device.type == "cuda"):
                    logits, predicted_log = model(dense[start:start + effective.batch_size].to(device), sparse[start:start + effective.batch_size].to(device))
                pcvr, value, _ = decode_predictions(logits, predicted_log)
                pcvr_parts.append(pcvr.cpu().numpy()); value_parts.append(value.cpu().numpy())
            result = pd.DataFrame({"user_id": encoded.user_id, "product_id": encoded.candidate_ad_id, "conversion_label": encoded.conversion_label.astype(int), "has_conversion_value": original["has_conversion_value"].astype(int), "conversion_value_eur": encoded.conversion_value_eur, "pCVR_clicked": np.concatenate(pcvr_parts), "predicted_conditional_value": np.concatenate(value_parts)})
            result["expected_value_per_click"] = result.pCVR_clicked * result.predicted_conditional_value
            if "event_id" in original: result.insert(0, "event_id", original.event_id.astype(str).to_numpy())
            write_csv_part(result, output_dir, parts); parts += 1; rows += len(result)
            timestamps = pd.to_numeric(original.get("click_timestamp"), errors="coerce").dropna()
            if len(timestamps): time_min = int(timestamps.min()) if time_min is None else min(time_min, int(timestamps.min())); time_max = int(timestamps.max()) if time_max is None else max(time_max, int(timestamps.max()))
    metadata = {"checkpoint": str(checkpoint_path), "model": "dcnv2", "row_count": rows, "part_count": parts, "timestamp_split_source": str(effective.future_a_path), "timestamp_min": time_min, "timestamp_max": time_max, "future_b_read": False, "task_semantics": {"pCVR_clicked": "P(conversion | clicked interaction)", "predicted_conditional_value": "E[conversion_value_eur | conversion=1, clicked interaction]", "expected_value_per_click": "pCVR_clicked * predicted_conditional_value; clicked-interaction quantity only"}}
    metadata_path = output_dir.parent / "future_a_predictions_metadata.json"; metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"prediction_dir": str(output_dir), "metadata_path": str(metadata_path), "future_b_read": False, "row_count": rows}


def _future_a_frames(path: Path, chunk_size: int, limit: int | None) -> Iterator[pd.DataFrame]:
    rows = 0
    for frame in iter_csv_parts(path, chunk_size):
        if limit is not None: frame = frame.iloc[:max(0, limit - rows)]
        if frame.empty: return
        rows += len(frame); yield frame
        if limit is not None and rows >= limit: return


def _encode_for_prediction(frame: pd.DataFrame, config: FineRankMultiTaskConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"user_id", "product_id", "conversion_label"}
    if missing := required - set(frame): raise ValueError(f"Future-A source missing {sorted(missing)}")
    original = frame.copy(); original["candidate_ad_id"] = original.product_id
    labels = pd.to_numeric(original.conversion_label, errors="coerce")
    valid = labels.isin((0, 1)) & original.user_id.notna() & original.candidate_ad_id.notna()
    original = original.loc[valid].copy(); original["conversion_label"] = labels.loc[valid].astype(np.float32)
    encoded = encode_feature_frame(original, bucket_sizes=config.bucket_sizes, random_seed=config.seed)
    actual_value = pd.to_numeric(original.get("conversion_value_eur", pd.Series(np.nan, index=original.index)), errors="coerce")
    declared = pd.to_numeric(original.get("has_conversion_value", actual_value.notna()), errors="coerce").fillna(0).astype(bool)
    original = original.reset_index(drop=True); original["has_conversion_value"] = (declared.to_numpy() & np.isfinite(actual_value.to_numpy()) & actual_value.ge(0).to_numpy()).astype(int)
    return encoded, original
