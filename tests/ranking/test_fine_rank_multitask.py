"""Contracts for the strict-temporal clicked-interaction multi-task ranker."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from search_ads_system.ranking.deepfm import DeepFMMultiTask
from search_ads_system.ranking.fine_rank_dataset import DENSE_FEATURES, SPARSE_FEATURES
from search_ads_system.ranking.fine_rank_multitask import (
    ClickInteractionDataset, FineRankMultiTaskConfig, decode_predictions, feature_contract, multitask_loss,
    run_fine_rank_multitask,
)


def _write_window(path: Path, offset: int) -> None:
    path.mkdir(parents=True)
    rows = []
    for index in range(12):
        converted = index % 3 == 0
        rows.append({"user_id": f"u{index % 4}", "product_id": f"p{index % 5}", "click_timestamp": offset + index,
                     "conversion_label": int(converted), "conversion_value_eur": 10.0 + index if converted else None,
                     "product_price": 2.0 + index, "clicks_last_7d": index, "device_type": "mobile", "product_brand": "brand"})
    pd.DataFrame(rows).to_csv(path / "part-00000.csv", index=False)


def _config(tmp_path: Path) -> FineRankMultiTaskConfig:
    split = tmp_path / "temporal" / "split"
    _write_window(split / "past", 100)
    _write_window(split / "future_a", 200)
    _write_window(split / "future_b", 300)
    return FineRankMultiTaskConfig(past_path=split / "past", future_a_path=split / "future_a", future_b_path=split / "future_b",
                                   model_dir=tmp_path / "models", metrics_dir=tmp_path / "metrics", batch_size=4, epochs=1,
                                   embedding_dim=4, hidden_dims=(8,), cross_layers=2, chunk_size=20, max_train_rows=12,
                                   max_validation_rows=12, device="cpu", amp=True)


def test_deepfm_has_real_fm_and_multitask_outputs() -> None:
    model = DeepFMMultiTask(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=(17,) * len(SPARSE_FEATURES), embedding_dim=4, hidden_dims=(8,))
    logits, values = model(torch.zeros(3, len(DENSE_FEATURES)), torch.zeros(3, len(SPARSE_FEATURES), dtype=torch.long))
    assert logits.shape == values.shape == (3,)
    assert hasattr(model, "linear_embeddings") and hasattr(model, "feature_embeddings")


def test_masked_loss_and_expected_value_are_safe() -> None:
    logits = torch.zeros(2, requires_grad=True); predicted = torch.ones(2, requires_grad=True)
    total, cvr, value = multitask_loss(logits, predicted, torch.tensor([0.0, 1.0]), torch.tensor([0.0, 2.0]), torch.zeros(2), lambda_cvr=1.0, lambda_value=.2)
    assert torch.isfinite(total) and value.item() == 0.0 and total.item() == pytest.approx(cvr.item())
    pcvr, conditional, expected = decode_predictions(torch.tensor([0.0]), torch.tensor([-50.0]))
    assert conditional.item() >= 0.0 and torch.allclose(expected, pcvr * conditional)


def test_feature_contract_rejects_targets() -> None:
    contract = feature_contract()
    assert "conversion_label" not in contract["categorical_features"]
    assert "conversion_value_eur" in contract["excluded_leakage_features"]


def test_declared_missing_conversion_value_is_excluded_from_value_loss(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = config.past_path / "part-00000.csv"
    frame = pd.read_csv(path); frame["has_conversion_value"] = 0; frame.to_csv(path, index=False)
    assert not any(float(row["value_mask"]) for row in ClickInteractionDataset(config.past_path, config, max_rows=12))


def test_strict_temporal_pipeline_selects_only_future_a(tmp_path: Path) -> None:
    result = run_fine_rank_multitask(_config(tmp_path), stage="sanity")
    report = json.loads(Path(result["metrics_json"]).read_text())
    assert result["future_b_read_for_model_selection"] is False
    assert report["temporal_contract"]["future_b_read_for_model_selection"] is False
    assert report["models"]["din"]["available"] is False
    assert (tmp_path / "models" / "deepfm.pt").exists() and (tmp_path / "models" / "dcnv2.pt").exists()
    assert report["models"]["deepfm"]["future_a"]["value"]["rows"] == 4
