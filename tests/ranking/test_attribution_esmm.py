"""Unit and tiny end-to-end coverage for the Attribution-only ESMM baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from search_ads_system.ranking.attribution_esmm import (
    CATEGORICAL_FEATURES,
    DENSE_FEATURES,
    AttributionESMM,
    esmm_loss,
    stable_hash_series,
    validate_feature_contract,
)
from search_ads_system.ranking.attribution_esmm_pipeline import (
    AttributionESMMConfig,
    build_model,
    evaluate_checkpoints,
    fit_past_normalization,
    train_all_models,
)


def test_esmm_probability_contract_loss_and_backward() -> None:
    torch.manual_seed(7)
    model = AttributionESMM((31,) * len(CATEGORICAL_FEATURES), 4, (8,), (4,), (4,))
    sparse = torch.randint(0, 31, (6, len(CATEGORICAL_FEATURES)))
    dense = torch.randn(6, len(DENSE_FEATURES))
    outputs = model(sparse, dense)
    assert torch.allclose(outputs["pctcvr"], outputs["pctr"] * outputs["pcvr"])
    assert all(torch.all((outputs[name] >= 0) & (outputs[name] <= 1)) for name in ("pctr", "pcvr", "pctcvr"))
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    ctcvr = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    losses = esmm_loss(outputs, labels, ctcvr)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_label_combinations_hashing_and_feature_guard() -> None:
    click = np.asarray([0, 1, 1])
    conversion = np.asarray([0, 0, 1])
    assert np.array_equal((click & conversion), np.asarray([0, 0, 1]))
    values = pd.Series(["user-1", "user-2", None, "user-1"], dtype="string")
    assert np.array_equal(stable_hash_series(values, 101), stable_hash_series(values, 101))
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_contract(("user_id", "cost"))


def _frame(start: int, rows: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    outcomes = ((0, 0), (1, 0), (1, 1), (0, 0))
    for index in range(rows):
        click, conversion = outcomes[index % len(outcomes)]
        record: dict[str, object] = {
            "user_id": f"u-{(start + index) % 5}", "campaign_id": f"c-{(start + index) % 3}",
            "time_since_last_click": float(index) if index % 5 else np.nan,
            "click": click, "conversion": conversion, "click_and_conversion": click & conversion,
        }
        record.update({name: f"{name}-{(start + index) % 4}" for name in CATEGORICAL_FEATURES if name not in record})
        records.append(record)
    return pd.DataFrame.from_records(records)


def _config(tmp_path: Path) -> AttributionESMMConfig:
    return AttributionESMMConfig(
        enabled=True, seed=3, device="cpu", past_path=tmp_path / "past", future_a_path=tmp_path / "future_a",
        checkpoint_dir=tmp_path / "models", metrics_dir=tmp_path / "metrics", embedding_dim=4,
        bucket_sizes=(97,) * len(CATEGORICAL_FEATURES), shared_hidden_dims=(12,), ctr_hidden_dims=(8,), cvr_hidden_dims=(8,),
        batch_size=8, inference_batch_size=16, io_chunk_size=9, epochs=1, learning_rate=0.01, weight_decay=0.0,
        lambda_ctcvr=1.0, num_workers=0, pin_memory=False, persistent_workers=False, mixed_precision=False,
        max_train_rows=None, max_validation_rows=None, early_stopping_patience=0,
        sanity_max_train_rows=20, sanity_max_validation_rows=12, sanity_epochs=1,
    )


def test_tiny_streaming_end_to_end_uses_past_and_future_a_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.past_path.mkdir()
    config.future_a_path.mkdir()
    _frame(0, 32).to_csv(config.past_path / "part-00000.csv", index=False)
    _frame(100, 20).to_csv(config.future_a_path / "part-00000.csv", index=False)
    # No Future-B directory exists. A successful run proves no model stage tries
    # to derive or open one from the temporal root.
    normalization = fit_past_normalization(config)
    assert normalization.valid_count < 32
    assert normalization.mean < np.log1p(20.0)
    trained = train_all_models(config)
    assert trained["future_b_read_for_model_selection"] is False
    report = evaluate_checkpoints(config)
    assert report["future_b_read_for_model_selection"] is False
    assert report["metrics"]["esmm"]["consistency"]["passed"]
    assert (config.checkpoint_dir / "single_ctr.pt").is_file()
    assert (config.checkpoint_dir / "naive_cvr.pt").is_file()
    assert (config.checkpoint_dir / "esmm.pt").is_file()
    original = build_model(config, "esmm")
    original.load_state_dict(torch.load(config.checkpoint_dir / "esmm.pt", map_location="cpu", weights_only=False)["model_state"])
    restored = build_model(config, "esmm")
    restored.load_state_dict(torch.load(config.checkpoint_dir / "esmm.pt", map_location="cpu", weights_only=False)["model_state"])
    sparse = torch.zeros((2, len(CATEGORICAL_FEATURES)), dtype=torch.long)
    dense = torch.zeros((2, len(DENSE_FEATURES)))
    assert torch.allclose(original(sparse, dense)["pctcvr"], restored(sparse, dense)["pctcvr"])
