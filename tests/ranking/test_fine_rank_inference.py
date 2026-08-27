"""Future-A-only standalone prediction artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from search_ads_system.ranking.fine_rank_inference import write_future_a_predictions
from search_ads_system.ranking.fine_rank_multitask import FineRankMultiTaskConfig, build_model
from search_ads_system.ranking.fine_rank_dataset import SPARSE_FEATURES


def test_future_a_prediction_artifact_uses_checkpoint_without_reading_future_b(tmp_path: Path) -> None:
    split = tmp_path / "temporal" / "split"; future_a = split / "future_a"; future_a.mkdir(parents=True)
    pd.DataFrame({"event_id": ["e1", "e2"], "user_id": ["u1", "u2"], "product_id": ["p1", "p2"], "click_timestamp": [100, 101], "conversion_label": [0, 1], "has_conversion_value": [0, 1], "conversion_value_eur": [None, 7.0], "product_price": [1., 2.], "clicks_last_7d": [1., 2.], "device_type": ["m", "d"]}).to_csv(future_a / "part-00000.csv", index=False)
    (split / "future_b").mkdir(); (split / "future_b" / "DO_NOT_READ.txt").write_text("invalid")
    config = FineRankMultiTaskConfig(past_path=split / "past", future_a_path=future_a, future_b_path=split / "future_b", model_dir=tmp_path / "models", metrics_dir=tmp_path / "metrics", batch_size=2, embedding_dim=4, hidden_dims=(8,), cross_layers=2, chunk_size=10, bucket_sizes=(17,) * len(SPARSE_FEATURES), device="cpu")
    model = build_model("dcnv2", config); checkpoint = tmp_path / "models" / "dcnv2.pt"; checkpoint.parent.mkdir()
    torch.save({"model": "dcnv2", "state_dict": model.state_dict(), "config": {"embedding_dim": 4, "hidden_dims": [8], "cross_layers": 2, "bucket_sizes": [17] * len(SPARSE_FEATURES), "seed": 2026}}, checkpoint)
    result = write_future_a_predictions(config, checkpoint_path=checkpoint, output_dir=tmp_path / "predictions")
    prediction = pd.read_csv(tmp_path / "predictions" / "part-00000.csv")
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert set(("user_id", "product_id", "conversion_label", "has_conversion_value", "conversion_value_eur", "pCVR_clicked", "predicted_conditional_value", "expected_value_per_click")) <= set(prediction)
    assert "event_id" in prediction and metadata["future_b_read"] is False
    assert (prediction.expected_value_per_click >= 0).all()
