"""Tests for bounded-memory coarse ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from search_ads_system.ranking.coarse_rank import (
    FEATURE_COLUMNS,
    CoarseRankConfig,
    TrainingData,
    assert_no_leakage_features,
    build_interaction_feature_index,
    build_labeled_user_examples,
    benchmark_coarse_rank,
    collect_training_data,
    deterministic_negative_sample,
    iter_candidate_groups,
    iter_candidate_batches,
    load_model,
    preprocess_features,
    run_coarse_rank,
    resolve_inference_backend,
    save_model,
    score_candidate_batch,
    score_candidate_group,
    stream_coarse_rank_output,
    train_coarse_ranker,
)
from search_ads_system.ranking.coarse_rank_diagnostics import diagnose_coarse_rank_samples


def _config(tmp_path: Path, *, top_k: int = 2, train: bool = True, chunk_size: int = 3) -> CoarseRankConfig:
    return CoarseRankConfig(
        input_path=tmp_path / "fused.csv",
        interaction_path=tmp_path / "interactions",
        output_path=tmp_path / "coarse.csv",
        model_path=tmp_path / "model.pkl",
        max_train_rows=100,
        max_users=100,
        top_k=top_k,
        chunk_size=chunk_size,
        random_seed=2026,
        negatives_per_positive=1,
        conversion_sample_weight=3.0,
        train=train,
    )


def _write_data(tmp_path: Path) -> None:
    interaction_path = tmp_path / "interactions"
    interaction_path.mkdir()
    interaction_rows = []
    candidate_rows = []
    for index, user_id in enumerate(("u1", "u2", "u3", "u4", "u5", "u6"), start=1):
        positive = f"a{index}"
        interaction_rows.append(
            {
                "user_id": user_id,
                "product_id": positive,
                "click_timestamp": index * 100,
                "conversion_label": int(index % 2 == 0),
                "product_price": float(index),
                "clicks_last_7d": index,
                "device_type": "mobile" if index % 2 else "desktop",
            }
        )
        candidate_rows.extend(
            [
                {"user_id": user_id, "candidate_ad_id": positive, "rrf_score": 0.9, "source_count": 2},
                {"user_id": user_id, "candidate_ad_id": f"b{index}", "rrf_score": 0.2, "source_count": 1},
                {"user_id": user_id, "candidate_ad_id": f"c{index}", "rrf_score": 0.1, "source_count": 1},
            ]
        )
    pd.DataFrame(interaction_rows).to_csv(interaction_path / "part-00000.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(tmp_path / "fused.csv", index=False)


def test_feature_preprocessing_handles_missing_values_and_excludes_leakage() -> None:
    features = preprocess_features(pd.DataFrame({"rrf_score": [None], "source_count": [None], "product_price": [np.nan]}))
    assert features.shape == (1, len(FEATURE_COLUMNS))
    assert np.isfinite(features).all()
    assert_no_leakage_features(FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage_features(["rrf_score", "conversion_label"])


def test_negative_sampling_is_deterministic() -> None:
    candidates = pd.DataFrame({"candidate_ad_id": ["a", "b", "c", "d"]})
    first = deterministic_negative_sample(candidates, user_id="u", count=2, random_seed=7)
    second = deterministic_negative_sample(candidates, user_id="u", count=2, random_seed=7)
    assert first["candidate_ad_id"].tolist() == second["candidate_ad_id"].tolist()


def test_click_labels_and_conversion_weights(tmp_path: Path) -> None:
    _write_data(tmp_path)
    config = _config(tmp_path)
    store = build_interaction_feature_index(config, tmp_path / "index.sqlite")
    try:
        candidates = next(iter_candidate_groups(config.input_path, config.chunk_size))[1]
        _, labels, weights, _ = build_labeled_user_examples(
            candidates,
            user_id="u1",
            feature_store=store,
            negatives_per_positive=1,
            random_seed=2026,
            conversion_sample_weight=3.0,
        )
        assert labels.tolist() == [1, 0]
        assert weights.tolist() == [1.0, 1.0]  # u1 is a non-converting click, still positive.
        u2 = list(iter_candidate_groups(config.input_path, config.chunk_size))[1][1]
        _, labels, weights, _ = build_labeled_user_examples(
            u2, user_id="u2", feature_store=store, negatives_per_positive=1, random_seed=2026, conversion_sample_weight=3.0
        )
        assert labels.tolist() == [1, 0]
        assert weights.tolist() == [3.0, 1.0]
    finally:
        store.close()


def test_candidate_groups_cross_chunk_boundary_without_truncation(tmp_path: Path) -> None:
    _write_data(tmp_path)
    groups = list(iter_candidate_groups(tmp_path / "fused.csv", chunk_size=2))
    assert [len(group) for _, group in groups] == [3] * 6


def test_output_top_k_ranks_from_one_and_respects_limit(tmp_path: Path) -> None:
    _write_data(tmp_path)
    config = _config(tmp_path, top_k=2)
    store = build_interaction_feature_index(config, tmp_path / "index.sqlite")
    try:
        training = collect_training_data(config, store)
        model, _, _ = train_coarse_ranker(training, config)
        assert stream_coarse_rank_output(config, model, store) == 12
        output = pd.read_csv(config.output_path)
        assert output.groupby("user_id").size().eq(2).all()
        assert output.groupby("user_id")["rank"].apply(lambda ranks: ranks.tolist() == [1, 2]).all()
        # b*/c* are intentionally absent from the feature index but remain in
        # candidate output through deterministic default features.
        assert output["candidate_ad_id"].str.startswith(("b", "c")).any()
    finally:
        store.close()


def test_train_false_loads_existing_model_and_writes_output(tmp_path: Path) -> None:
    _write_data(tmp_path)
    train_config = _config(tmp_path)
    store = build_interaction_feature_index(train_config, tmp_path / "index.sqlite")
    try:
        model, _, _ = train_coarse_ranker(collect_training_data(train_config, store), train_config)
        save_model(model, train_config.model_path)
    finally:
        store.close()
    assert load_model(train_config.model_path).feature_columns == FEATURE_COLUMNS
    result = run_coarse_rank(_config(tmp_path, train=False))
    assert result == {}
    assert (tmp_path / "coarse.csv").is_file()


def test_feature_store_uses_defaults_for_unknowns_after_cache_eviction(tmp_path: Path) -> None:
    _write_data(tmp_path)
    config = _config(tmp_path)
    store = build_interaction_feature_index(config, tmp_path / "index.sqlite")
    store.cache_size = 1
    try:
        first = store.enrich(
            pd.DataFrame(
                [
                    {"candidate_ad_id": "a1", "rrf_score": 0.9, "source_count": 2},
                    {"candidate_ad_id": "unknown", "rrf_score": 0.1, "source_count": 1},
                ]
            )
        )
        assert first.loc[0, "product_price"] == 1.0
        assert first.loc[1, "product_price"] == 0.0
        assert first.loc[1, "device_type"] == "__MISSING__"
        # a1 is evicted by the tiny cache; a mixed follow-up batch must still
        # retrieve it and must never KeyError for the unknown candidate.
        second = store.enrich(
            pd.DataFrame(
                [
                    {"candidate_ad_id": "a1", "rrf_score": 0.9, "source_count": 2},
                    {"candidate_ad_id": "also-unknown", "rrf_score": 0.1, "source_count": 1},
                ]
            )
        )
        assert second["candidate_ad_id"].tolist() == ["a1", "also-unknown"]
        assert second.loc[0, "product_price"] == 1.0
        missing, total = store.consume_feature_stats()
        assert (missing, total) == (2, 4)
    finally:
        store.close()


def test_batch_inference_matches_per_user_and_parallel_output(tmp_path: Path) -> None:
    _write_data(tmp_path)
    config = _config(tmp_path, top_k=2)
    store = build_interaction_feature_index(config, tmp_path / "index.sqlite")
    try:
        model, _, _ = train_coarse_ranker(collect_training_data(config, store), config)
        groups = list(iter_candidate_groups(config.input_path, config.chunk_size))
        batched = score_candidate_batch(groups, model, store, config.top_k)
        expected = []
        for user_id, candidates in groups:
            ranked = score_candidate_group(candidates, model, store).iloc[: config.top_k]
            expected.extend((user_id, row.candidate_ad_id, float(row.coarse_score), rank) for rank, row in enumerate(ranked.itertuples(index=False), start=1))
        assert [(user, candidate, rank) for user, candidate, _, rank in batched] == [
            (user, candidate, rank) for user, candidate, _, rank in expected
        ]
        single_path = config.output_path
        stream_coarse_rank_output(config, model, store)
        single = pd.read_csv(single_path)
        parallel_config = CoarseRankConfig(**{**config.__dict__, "output_path": tmp_path / "parallel.csv", "num_workers": 2})
        stream_coarse_rank_output(parallel_config, model, store)
        parallel = pd.read_csv(parallel_config.output_path)
        pd.testing.assert_frame_equal(single, parallel)
    finally:
        store.close()


def test_bounded_user_batches_backend_fallback_and_benchmark(tmp_path: Path) -> None:
    _write_data(tmp_path)
    config = _config(tmp_path)
    batches = list(iter_candidate_batches(config.input_path, config.chunk_size, max_users=4, batch_users=2, batch_candidates=6))
    assert [len(batch) for batch in batches] == [2, 2]
    assert all(sum(len(candidates) for _, candidates in batch) <= 6 for batch in batches)
    assert resolve_inference_backend(CoarseRankConfig(**{**config.__dict__, "model_type": "xgboost_gpu"})) == "hist_gbdt"
    store = build_interaction_feature_index(config, tmp_path / "index.sqlite")
    try:
        model, _, _ = train_coarse_ranker(collect_training_data(config, store), config)
        result = benchmark_coarse_rank(config, model, store, worker_counts=(1,), max_users=2)
        assert result[1]["processed_candidates"] == 6
        assert result[1]["candidates_per_second"] > 0
    finally:
        store.close()


def test_streaming_sample_diagnosis_reuses_training_labels(tmp_path: Path) -> None:
    _write_data(tmp_path)
    report = diagnose_coarse_rank_samples(_config(tmp_path))

    assert report["interaction_side"]["total_interaction_rows"] == 6
    assert report["interaction_side"]["unique_interaction_users"] == 6
    assert report["candidate_side"]["total_candidate_rows"] == 18
    assert report["candidate_side"]["selected_candidate_users"] == 6
    assert report["overlap"]["overlapping_users"] == 6
    assert report["positive_funnel"]["candidate_interaction_pair_matches_before_split"] == 6
    assert report["positive_funnel"]["final_positive_samples"] == 6
    assert report["negative_sampling"]["expected_training_rows"] == 12
    assert report["negative_sampling"]["actual_training_rows"] == 12
    assert report["negative_sampling"]["ratio_is_exact"]
    assert report["exclusion_reasons"]["product_not_in_interactions"] == 12
    assert not (tmp_path / "model.pkl.diagnostics.sqlite").exists()
