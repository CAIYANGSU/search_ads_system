"""Fine-rank correctness tests using tiny click-conditioned fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from search_ads_system.ranking.dcnv2 import DCNv2MultiTask
from search_ads_system.ranking.fine_rank import (
    FineRankConfig,
    _attach_inference_prediction_columns,
    _rank_inference_candidates,
    benchmark_fine_rank_inference_preprocessing,
    build_dataset,
    build_model,
    dataset_spec,
    infer_fine_rank,
    load_fine_ranker,
    multitask_loss,
    resolve_device,
    train_fine_ranker,
)
from search_ads_system.ranking.fine_rank_dataset import (
    DENSE_FEATURES,
    SPARSE_FEATURES,
    FineRankParquetDataset,
    assert_no_fine_rank_leakage,
    encode_feature_frame,
    stable_hash,
)
from search_ads_system.ranking.fine_rank_audit import FineRankAuditConfig, run_fine_rank_audit, run_id_memorization_ablation


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    interactions = tmp_path / "interactions"; interactions.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "product_id": "a1", "conversion_label": 0, "conversion_value_eur": np.nan, "click_timestamp": 100, "product_price": 2.0, "clicks_last_7d": 1, "device_type": "mobile", "product_brand": "b1"},
            {"user_id": "u1", "product_id": "a2", "conversion_label": 1, "conversion_value_eur": 5.0, "click_timestamp": 101, "product_price": 3.0, "clicks_last_7d": 1, "device_type": "mobile", "product_brand": "b2"},
            {"user_id": "u2", "product_id": "a3", "conversion_label": 1, "conversion_value_eur": np.nan, "click_timestamp": 200, "product_price": 4.0, "clicks_last_7d": 2, "device_type": "desktop", "product_brand": "b3"},
            {"user_id": "u2", "product_id": "a4", "conversion_label": 0, "conversion_value_eur": np.nan, "click_timestamp": 201, "product_price": 5.0, "clicks_last_7d": 2, "device_type": "desktop", "product_brand": "b4"},
        ]
    ).to_csv(interactions / "part-00000.csv", index=False)
    candidates = tmp_path / "coarse.csv"
    pd.DataFrame(
        [
            {"user_id": "u1", "candidate_ad_id": "a1", "coarse_score": .2, "rank": 2},
            {"user_id": "u1", "candidate_ad_id": "a2", "coarse_score": .9, "rank": 1},
            {"user_id": "u1", "candidate_ad_id": "missing", "coarse_score": .1, "rank": 3},
            {"user_id": "u2", "candidate_ad_id": "a3", "coarse_score": .8, "rank": 1},
            {"user_id": "u2", "candidate_ad_id": "a4", "coarse_score": .3, "rank": 2},
        ]
    ).to_csv(candidates, index=False)
    return interactions, candidates


def _config(tmp_path: Path) -> FineRankConfig:
    interactions, candidates = _write_fixture(tmp_path)
    return FineRankConfig(
        mode="full", input_path=candidates, output_path=tmp_path / "fine.csv", model_path=tmp_path / "model.pt", cache_dir=tmp_path / "cache",
        feature_source_path=interactions, train_label_path=interactions, validation_label_path=None, metrics_path=tmp_path / "metrics.json",
        top_k=2, max_train_rows=20, chunk_size=2, embedding_dim=4, hidden_dims=(8,), num_cross_layers=2, batch_size=2, inference_batch_size=3,
        epochs=1, num_workers=0, prefetch_factor=1, persistent_workers=False, amp=True, bucket_sizes=(17,) * len(SPARSE_FEATURES), validation_fraction=.25,
    )


def test_dcnv2_forward_shapes_and_prediction_range() -> None:
    model = DCNv2MultiTask(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=(17,) * len(SPARSE_FEATURES), embedding_dim=4, hidden_dims=(8,), num_cross_layers=2)
    logits, value = model(torch.zeros(3, len(DENSE_FEATURES)), torch.zeros(3, len(SPARSE_FEATURES), dtype=torch.long))
    probability, predicted_value, expected = model.predict(torch.zeros(3, len(DENSE_FEATURES)), torch.zeros(3, len(SPARSE_FEATURES), dtype=torch.long))
    assert logits.shape == value.shape == probability.shape == predicted_value.shape == expected.shape == (3,)
    assert torch.all((probability >= 0) & (probability <= 1))
    assert torch.allclose(expected, probability * predicted_value)


def test_masked_multitask_loss_ignores_nonconversion_and_missing_value() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    log_values = torch.tensor([4.0, 4.0, 4.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 1.0]); values = torch.tensor([0.0, 5.0, 0.0]); mask = torch.tensor([0.0, 1.0, 0.0])
    total, cvr, value = multitask_loss(logits, log_values, labels, values, mask, .2)
    assert value.item() > 0 and total.item() > cvr.item()
    _, _, empty_value = multitask_loss(logits, log_values, labels, values, torch.zeros_like(mask), .2)
    assert empty_value.item() == 0.0 and torch.isfinite(empty_value)


def test_leakage_guard_and_deterministic_hashing() -> None:
    assert stable_hash("u", 101, 7) == stable_hash("u", 101, 7)
    with pytest.raises(ValueError, match="leakage"):
        assert_no_fine_rank_leakage(["coarse_score", "conversion_label"])
    with pytest.raises(ValueError, match="Future"):
        assert_no_fine_rank_leakage(["future_cvr"])


def test_encoded_value_mask_is_only_positive_with_known_value() -> None:
    frame = pd.DataFrame({"user_id": ["u", "u"], "candidate_ad_id": ["a", "b"], "conversion_label": [0, 1], "conversion_value_eur": [np.nan, np.nan]})
    encoded = encode_feature_frame(frame, bucket_sizes=(17,) * len(SPARSE_FEATURES))
    assert encoded.value_mask.tolist() == [0.0, 0.0]


def test_dataset_cache_metadata_and_click_conditioned_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    metadata = build_dataset(config)
    assert metadata["row_count"] + metadata["validation_row_count"] == 4  # direct observed clicks, not candidate overlap
    assert metadata["feature_version"]
    assert metadata["dataset_construction_version"] == "observed-clicked-interactions-v2"
    assert metadata["diagnostics"]["source_interaction_rows"] == 4
    assert metadata["diagnostics"]["train_rows"] + metadata["diagnostics"]["validation_rows"] == 4
    assert metadata["diagnostics"]["conversion_positive_rows"] == 2
    assert metadata["diagnostics"]["valid_conversion_value_rows"] == 1
    assert metadata["value_transform"]["version"] == "normalized-log1p-v1"
    assert build_dataset(config)["config_hash"] == metadata["config_hash"]
    rows = list(FineRankParquetDataset(config.cache_dir)) + list(FineRankParquetDataset(config.cache_dir.parent / "validation"))
    assert any(row["label"] == 0 for row in rows)
    assert any(row["label"] == 1 for row in rows)
    assert sum(row["value_mask"] for row in rows) == 1.0


def test_observed_click_training_does_not_depend_on_coarse_candidate_overlap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    pd.DataFrame([{"user_id": "nobody", "candidate_ad_id": "unobserved", "coarse_score": .1, "rank": 1}]).to_csv(config.input_path, index=False)
    metadata = build_dataset(config)
    assert metadata["row_count"] + metadata["validation_row_count"] == 4
    assert metadata["diagnostics"]["filter_rows"]["observed_click_rows_after_required_fields"] == 4


def test_extreme_value_head_is_clamped_before_expm1() -> None:
    model = DCNv2MultiTask(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=(17,) * len(SPARSE_FEATURES), embedding_dim=4, hidden_dims=(8,), num_cross_layers=1)
    with torch.no_grad():
        model.value_head.weight.zero_(); model.value_head.bias.fill_(1_000_000)
    probability, predicted_log, predicted_value, expected = model.predict_with_log(torch.zeros(2, len(DENSE_FEATURES)), torch.zeros(2, len(SPARSE_FEATURES), dtype=torch.long), value_mean=0.0, value_std=1.0, prediction_log_min=0.0, prediction_log_max=20.0)
    assert torch.isfinite(predicted_log).all() and torch.isfinite(predicted_value).all() and torch.isfinite(expected).all()
    assert torch.all(predicted_value >= 0) and torch.allclose(expected, probability * predicted_value)


def test_extreme_finite_dense_features_are_bounded_before_cross_network() -> None:
    frame = pd.DataFrame({"user_id": ["u"], "candidate_ad_id": ["a"], "product_price": [1e300], "clicks_last_7d": [1e300], "coarse_score": [1e300], "conversion_label": [0]})
    encoded = encode_feature_frame(frame, bucket_sizes=(17,) * len(SPARSE_FEATURES))
    dense = encoded[[f"dense__{name}" for name in DENSE_FEATURES]].to_numpy(dtype=np.float32)
    assert np.isfinite(dense).all() and np.abs(dense).max() <= 10.0
    model = DCNv2MultiTask(dense_dim=len(DENSE_FEATURES), sparse_bucket_sizes=(17,) * len(SPARSE_FEATURES), embedding_dim=4, hidden_dims=(8,), num_cross_layers=2)
    logits, values = model(torch.from_numpy(dense), torch.zeros(1, len(SPARSE_FEATURES), dtype=torch.long))
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def test_checkpoint_train_false_style_load_and_chunked_topk_inference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    metadata = build_dataset(config)
    result = train_fine_ranker(config, metadata)
    assert result["best_epoch"] == 1 and config.model_path.exists()
    model, checkpoint = load_fine_ranker(config)
    assert checkpoint["feature_schema"]["dense"] == list(DENSE_FEATURES)
    assert isinstance(model, DCNv2MultiTask)
    output = infer_fine_rank(config)
    ranked = pd.read_csv(config.output_path)
    assert output["output_rows"] == len(ranked)
    assert output["prediction_diagnostics"]["nonfinite_values"] == 0
    assert set(output["prediction_diagnostics"]["predicted_log_value"]) == {"min", "median", "p95", "p99", "max"}
    assert output["timings"]["feature_preparation_seconds"] >= 0 and output["slowest_stage"] in output["timings"]
    assert ranked.groupby("user_id").size().eq(2).all()
    assert ranked.groupby("user_id")["rank"].apply(lambda ranks: ranks.tolist() == [1, 2]).all()
    assert np.allclose(ranked.expected_value_score, ranked.pCVR * ranked.predicted_conversion_value)


def test_float16_gpu_predictions_are_promoted_before_pandas_ranking() -> None:
    identifiers = pd.DataFrame(
        {
            "user_id": pd.Series(["u2", "u1", "u1", "u2"], dtype="string"),
            "candidate_ad_id": pd.Series(["a1", "a2", "a1", "a2"], dtype="string"),
        }
    )
    float16_predictions = {
        "p_cvr": np.asarray([.3, .5, .5, .3], dtype=np.float16),
        "predicted_log_value": np.asarray([1.0, 2.0, 1.5, 2.5], dtype=np.float16),
        "predicted_value": np.asarray([3.0, 4.0, 2.0, 5.0], dtype=np.float16),
        "expected_value": np.asarray([.9, 2.0, 1.0, 1.5], dtype=np.float16),
    }
    half_frame = identifiers.copy()
    _attach_inference_prediction_columns(half_frame, **float16_predictions)
    assert all(half_frame[column].dtype == np.dtype("float32") for column in (
        "pCVR", "predicted_log_conversion_value", "predicted_conversion_value", "expected_value_score",
    ))
    assert str(half_frame["user_id"].dtype) == "string"
    assert str(half_frame["candidate_ad_id"].dtype) == "string"
    ranked_half = _rank_inference_candidates(half_frame, top_k=2)

    reference_frame = identifiers.copy()
    _attach_inference_prediction_columns(
        reference_frame,
        **{name: values.astype(np.float32) for name, values in float16_predictions.items()},
    )
    ranked_reference = _rank_inference_candidates(reference_frame, top_k=2)
    assert ranked_half[["user_id", "candidate_ad_id", "rank"]].equals(
        ranked_reference[["user_id", "candidate_ad_id", "rank"]]
    )


def test_read_only_effect_audit_reports_metrics_calibration_overlap_and_split_warning(tmp_path: Path) -> None:
    config = _config(tmp_path)
    metadata = build_dataset(config)
    train_fine_ranker(config, metadata)
    result = run_fine_rank_audit(config, FineRankAuditConfig(output_path=tmp_path / "fine_rank_audit.json", calibration_bins=10))
    report = json.loads((tmp_path / "fine_rank_audit.json").read_text())
    assert result["report_path"].endswith("fine_rank_audit.json")
    assert set(report["classification_metrics"]) == {"roc_auc", "pr_auc", "logloss", "brier_score", "positive_rate"}
    assert len(report["calibration"]["bins"]) == 10
    assert "exact_user_product_pair_overlap" in report["train_validation_overlap"]
    assert set(report["strict_holdout_slices"]) >= {"unseen_user", "unseen_product", "unseen_user_product_pair"}
    assert report["split_audit"]["row_random_split"] is True
    assert report["leakage_audit"]["passed_static_feature_guard"] is True
    assert report["id_memorization_ablation"]["ran"] is False


def test_small_id_memorization_ablation_is_temporary_and_has_all_variants(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_dataset(config)
    result = run_id_memorization_ablation(
        config,
        FineRankAuditConfig(
            output_path=tmp_path / "unused.json", ablation_train_rows=3,
            ablation_validation_rows=1, ablation_epochs=1, ablation_batch_size=2,
        ),
    )
    assert result["ran"] is True and result["temporary"] is True
    assert set(result["variants"]) == {
        "A_all_features", "B_without_user_id", "C_without_product_id", "D_without_user_id_and_product_id",
    }


def test_vectorized_100k_candidate_preprocessing_benchmark(tmp_path: Path) -> None:
    config = FineRankConfig(**{**_config(tmp_path).__dict__, "chunk_size": 10_000})
    build_dataset(config)
    users = np.repeat([f"u{index}" for index in range(2_000)], 50)
    pd.DataFrame({"user_id": users, "candidate_ad_id": np.resize(np.asarray(["a1", "a2", "a3", "a4", "missing"]), len(users)), "coarse_score": .5, "rank": np.tile(np.arange(1, 51), 2_000)}).to_csv(config.input_path, index=False)
    benchmark = benchmark_fine_rank_inference_preprocessing(config)
    assert benchmark["rows"] == 100_000
    assert benchmark["rows_per_second"] > 1_000


def test_cpu_amp_and_cuda_unavailable_fallback() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"


def test_temporal_dataset_spec_isolated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    temporal = FineRankConfig(**{**config.__dict__, "mode": "temporal", "feature_source_path": tmp_path / "temporal" / "split" / "past", "train_label_path": tmp_path / "temporal" / "split" / "future_a", "validation_label_path": tmp_path / "temporal" / "split" / "future_b"})
    spec = dataset_spec(temporal)
    assert "past" in str(spec.feature_source_path) and "future_a" in str(spec.train_label_path)


def test_temporal_uses_past_only_features_with_future_labels(tmp_path: Path) -> None:
    past = tmp_path / "temporal" / "split" / "past"; future_a = tmp_path / "temporal" / "split" / "future_a"; future_b = tmp_path / "temporal" / "split" / "future_b"
    past.mkdir(parents=True); future_a.mkdir(parents=True); future_b.mkdir(parents=True)
    pd.DataFrame([{"user_id": "u", "product_id": "a", "conversion_label": 0, "conversion_value_eur": np.nan, "click_timestamp": 1, "product_price": 1.0, "clicks_last_7d": 1}]).to_csv(past / "part-00000.csv", index=False)
    pd.DataFrame([{"user_id": "u", "product_id": "a", "conversion_label": 1, "conversion_value_eur": 9.0, "click_timestamp": 2, "product_price": 999.0, "clicks_last_7d": 99}]).to_csv(future_a / "part-00000.csv", index=False)
    pd.DataFrame([{"user_id": "u", "product_id": "a", "conversion_label": 0, "conversion_value_eur": np.nan, "click_timestamp": 3, "product_price": 999.0, "clicks_last_7d": 99}]).to_csv(future_b / "part-00000.csv", index=False)
    candidates = tmp_path / "temporal" / "ranking" / "coarse.csv"; candidates.parent.mkdir(parents=True)
    pd.DataFrame([{"user_id": "u", "candidate_ad_id": "a", "coarse_score": .5, "rank": 1}]).to_csv(candidates, index=False)
    config = FineRankConfig(mode="temporal", input_path=candidates, output_path=tmp_path / "temporal" / "ranking" / "fine.csv", model_path=tmp_path / "temporal" / "models" / "model.pt", cache_dir=tmp_path / "temporal" / "ranking" / "fine_rank" / "train", feature_source_path=past, train_label_path=future_a, validation_label_path=future_b, metrics_path=tmp_path / "temporal" / "metrics.json", max_train_rows=10, chunk_size=10, embedding_dim=4, hidden_dims=(8,), num_cross_layers=1, batch_size=1, inference_batch_size=1, epochs=1, num_workers=0, prefetch_factor=1, persistent_workers=False, bucket_sizes=(17,) * len(SPARSE_FEATURES), validation_fraction=.1)
    build_dataset(config)
    row = next(iter(FineRankParquetDataset(config.cache_dir)))
    assert row["label"] == 1.0
    assert row["dense"][DENSE_FEATURES.index("product_price")] == pytest.approx(np.log1p(1.0) / 5.0)
