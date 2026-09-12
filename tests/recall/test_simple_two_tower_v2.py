"""Focused contract tests for the independent simple Two-Tower v2."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch

from search_ads_system.recall.simple_two_tower_v2 import (
    OOV_INDEX, SimpleTwoTowerV2Config, SimpleTwoTowerV2Model, _encode,
    _hard_negative_fingerprint, _mine_hard_negative_pools, _source_fingerprint, _synchronize_cuda, _v2_teacher_fingerprint, _vocab,
    assert_past_only_hard_negative_contract, duplicate_aware_inbatch_loss, evaluate_candidates, prepare_data, run_ablation, sample_mixed_negatives,
)


def _window(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "part-00000.csv", index=False)


def _config(tmp_path):
    return SimpleTwoTowerV2Config(tmp_path / "past", tmp_path / "future_a", tmp_path / "out", max_users=10, batch_size=2, epochs=1, top_k=2, hidden_dims=(8,), embedding_dim=4, feature_embedding_dim=3)


def _rows(timestamp, users=("u1", "u2")):
    output = []
    for index, user in enumerate(users):
        output.append({"user_id": user, "product_id": f"p{index + 1}", "click_timestamp": timestamp + index, "conversion_label": index % 2, "product_price": index + 1., "product_brand": "b", "product_category_1": "c1", "product_category_2": "c2", "product_category_3": "c3", "partner_id": "partner", "device_type": "mobile", "audience_id": "aud"})
    return output


def test_past_only_features_and_raw_product_overlap(tmp_path):
    _window(tmp_path / "past", _rows(10) + [{**_rows(10)[0], "user_id": "u1", "product_id": "p3", "click_timestamp": 12, "conversion_label": 1}])
    _window(tmp_path / "future_a", _rows(30) + [{**_rows(30)[0], "user_id": "u1", "product_id": "cold", "click_timestamp": 32}])
    data = prepare_data(_config(tmp_path))
    assert data.metadata["searchable_future_truth_items"] == 2
    assert data.metadata["non_searchable_future_truth_items"] == 1
    assert data.metadata["user_aggregates"]["source"] == "Past only"
    assert np.isfinite(data.item_price).all() and np.isfinite(data.user_stats).all()


def test_temporal_order_and_zero_catalogue_overlap_fail_loudly(tmp_path):
    _window(tmp_path / "past", _rows(10)); _window(tmp_path / "future_a", [{**_rows(30)[0], "product_id": "cold"}])
    with pytest.raises(RuntimeError, match="zero overlap"):
        prepare_data(_config(tmp_path))
    _window(tmp_path / "future_a", _rows(10))
    with pytest.raises(RuntimeError, match="Temporal contract"):
        prepare_data(_config(tmp_path))


def test_pad_oov_and_duplicate_inbatch_are_safe():
    vocab = _vocab(["known"])
    assert _encode([None, "unknown", "known"], vocab).tolist() == [OOV_INDEX, OOV_INDEX, 2]
    logits = torch.tensor([[2., 2.], [2., 2.]], requires_grad=True)
    loss = duplicate_aware_inbatch_loss(logits, torch.tensor([7, 7]), torch.ones(2))
    assert torch.isclose(loss, torch.tensor(0.0)); loss.backward()


def test_retrieval_metrics_have_correct_recall_hit_mrr_ndcg_and_ordering(tmp_path):
    path = tmp_path / "candidates.csv"
    pd.DataFrame([
        ("u1", "a", .9, 1, "v"), ("u1", "x", .8, 2, "v"),
        ("u2", "x", .9, 1, "v"), ("u2", "b", .8, 2, "v"),
    ], columns=["user_id", "candidate_ad_id", "two_tower_score", "rank", "model_variant"]).to_csv(path, index=False)
    result = evaluate_candidates(path, {"u1": {"a", "z"}, "u2": {"b"}}, {"u1": {"a"}, "u2": {"b"}}, {"a", "b", "x"}, top_k=50)
    assert result["overall_recall@50"] == pytest.approx(.75)
    assert result["hit_rate@50"] == 1
    assert result["mrr@50"] == pytest.approx(.75)
    assert 0 < result["ndcg@50"] <= 1
    assert not any("@100" in key or "@200" in key for key in result)
    broken = pd.read_csv(path); broken.loc[1, "rank"] = 3; broken.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="contiguous"):
        evaluate_candidates(path, {"u1": {"a"}}, {"u1": {"a"}}, {"a"}, top_k=50)


def test_zero_hit_recall_is_zero_and_only_supported_cutoffs_are_reported(tmp_path):
    path = tmp_path / "zero_hits.csv"
    pd.DataFrame([("u1", "x", .9, 1, "v")], columns=["user_id", "candidate_ad_id", "two_tower_score", "rank", "model_variant"]).to_csv(path, index=False)
    result = evaluate_candidates(path, {"u1": {"a"}}, {"u1": {"a"}}, {"a", "x"}, top_k=50)
    assert result["overall_recall@50"] == 0.0
    assert result["warm_recall@50"] == 0.0
    assert "overall_recall@100" not in result and "warm_recall@200" not in result
    full = evaluate_candidates(path, {"u1": {"a"}}, {"u1": {"a"}}, {"a", "x"}, top_k=200)
    assert {"overall_recall@50", "overall_recall@100", "overall_recall@200"}.issubset(full)


def test_cuda_synchronization_helper_only_syncs_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device.type))
    _synchronize_cuda(torch.device("cpu")); _synchronize_cuda(torch.device("cuda"))
    assert calls == ["cuda"]


def test_variant_architectures_run_on_cpu(tmp_path):
    _window(tmp_path / "past", _rows(10)); _window(tmp_path / "future_a", _rows(30))
    data = prepare_data(_config(tmp_path)); context = torch.as_tensor(data.eval_context[:2])
    for variant in ("v0_id_only", "v1_item_content", "v2_user_context_stats", "v3_in_batch_negatives"):
        model = SimpleTwoTowerV2Model(data, _config(tmp_path), variant)
        assert model.encode_users(torch.tensor([0, 1]), context).shape == (2, 4)
        assert model.encode_items(torch.tensor([0, 1])).shape == (2, 4)


def test_future_a_never_expands_past_vocabs_and_cohort_is_deterministic(tmp_path):
    _window(tmp_path / "past", _rows(10))
    future = _rows(30)
    future[0].update(product_brand="future_brand", device_type="future_device", audience_id="future_audience")
    _window(tmp_path / "future_a", future)
    first, second = prepare_data(_config(tmp_path)), prepare_data(_config(tmp_path))
    assert "future_brand" not in first.vocabs["product_brand"]
    assert "future_device" not in first.vocabs["device_type"]
    assert first.users.tolist() == second.users.tolist()


def test_preprocessing_cache_preserves_prepared_state_and_reports_stages(tmp_path):
    _window(tmp_path / "past", _rows(10)); _window(tmp_path / "future_a", _rows(30))
    config = _config(tmp_path)
    cold, warm = prepare_data(config), prepare_data(config)
    assert (config.output_dir / "simple_two_tower_v2_cache" / "prepared_arrays.npz").is_file()
    assert np.array_equal(cold.users, warm.users)
    assert np.array_equal(cold.products, warm.products)
    assert np.array_equal(cold.train_user, warm.train_user)
    assert np.array_equal(cold.item_features, warm.item_features)
    assert cold.truth == warm.truth and cold.histories == warm.histories
    assert cold.metadata["preprocessing_stage_seconds"]
    before = _source_fingerprint(config)
    with (tmp_path / "past" / "part-00000.csv").open("a", encoding="utf-8") as handle: handle.write("\n")
    assert _source_fingerprint(config) != before


def test_end_to_end_cpu_smoke_all_four_variants(tmp_path):
    past = []
    for index in range(10):
        row = _rows(10 + index, ("u1",))[0]
        row.update(product_id=f"p{index}", conversion_label=index % 2)
        past.append(row)
    past += [{**_rows(25, ("u2",))[0], "product_id": "p1"}]
    future = []
    for user, product in (("u1", "p0"), ("u1", "p4"), ("u1", "p8"), ("u2", "p1")):
        future.append({**_rows(40, (user,))[0], "user_id": user, "product_id": product, "click_timestamp": 40})
    _window(tmp_path / "past", past); _window(tmp_path / "future_a", future)
    cfg = SimpleTwoTowerV2Config(tmp_path / "past", tmp_path / "future_a", tmp_path / "out", max_users=2, batch_size=4, epochs=1, top_k=50, hidden_dims=(8,), embedding_dim=4, feature_embedding_dim=3, device="cpu", use_bf16=False, variants=("v0_id_only", "v1_item_content", "v2_user_context_stats", "v3_in_batch_negatives"))
    result = run_ablation(cfg)
    assert result.model.tolist() == list(cfg.variants)
    assert (cfg.output_dir / "metrics" / "simple_two_tower_v2_ablation.csv").is_file()
    for variant in cfg.variants:
        candidates = pd.read_csv(cfg.output_dir / "candidates" / f"{variant}_top50.csv")
        assert candidates.groupby("user_id").size().le(50).all()


def _v4_data(tmp_path):
    # u1 has one known Past positive and five safe Past catalogue candidates;
    # u2 supplies the remaining catalogue content but has no trainable negative.
    past = []
    for index in range(6):
        row = _rows(10 + index, ("u1" if index == 0 else "u2",))[0]
        row.update(product_id=f"p{index}", conversion_label=0)
        past.append(row)
    future = [{**_rows(40 + index, ("u1",))[0], "product_id": f"p{index}"} for index in range(6)]
    future.append({**_rows(50, ("u2",))[0], "product_id": "p1"})
    _window(tmp_path / "past", past); _window(tmp_path / "future_a", future)
    cfg = SimpleTwoTowerV2Config(tmp_path / "past", tmp_path / "future_a", tmp_path / "out", max_users=2, batch_size=2, epochs=1, top_k=50, hidden_dims=(8,), embedding_dim=4, feature_embedding_dim=3, device="cpu", use_bf16=False, random_negative_count=2, hard_negative_count=2, hard_negative_retrieval_topn=6, hard_negative_rank_start=0)
    return prepare_data(cfg), cfg


def test_v4_mining_is_past_only_deterministic_and_excludes_known_positives(tmp_path):
    pytest.importorskip("faiss")
    data, cfg = _v4_data(tmp_path)
    torch.manual_seed(cfg.seed); first_model = SimpleTwoTowerV2Model(data, cfg, "v2_user_context_stats")
    first, _ = _mine_hard_negative_pools(first_model, data, cfg, torch.device("cpu"))
    # Truth is intentionally changed to impossible Future-A labels. Mining has
    # no truth argument and must return the same Past-only candidate pools.
    data.truth = {"u1": {"future_only"}}; data.warm_truth = {"u1": set()}
    torch.manual_seed(cfg.seed); second_model = SimpleTwoTowerV2Model(data, cfg, "v2_user_context_stats")
    second, _ = _mine_hard_negative_pools(second_model, data, cfg, torch.device("cpu"))
    assert [x.tolist() for x in first] == [x.tolist() for x in second]
    assert_past_only_hard_negative_contract(data, data.train_user, data.train_item, first)


def test_v4_mixed_sampler_composition_and_random_fallback(tmp_path):
    data, cfg = _v4_data(tmp_path)
    users, positives = data.train_user[:1], data.train_item[:1]
    pool = np.asarray([item for item in range(len(data.products)) if item not in data.histories[int(users[0])]], dtype=np.int64)
    values, diagnostics = sample_mixed_negatives(users, positives, [pool], data, cfg, np.random.default_rng(9))
    assert values.shape == (1, 4) and len(set(values[0])) == 4
    assert set(values[0, cfg.random_negative_count:]).issubset(set(pool))
    assert diagnostics["mixed_fraction_requested_hard"] == 1
    fallback, fallback_diag = sample_mixed_negatives(users, positives, [pool[:1]], data, cfg, np.random.default_rng(9))
    assert fallback.shape == (1, 4) and fallback_diag["mixed_fraction_random_fallback"] == 1
    assert int(positives[0]) not in fallback[0] and not set(fallback[0]) & data.histories[int(users[0])]


def test_v4_cache_fingerprints_invalidate_with_mining_and_teacher_inputs(tmp_path):
    data, cfg = _v4_data(tmp_path)
    teacher = _v2_teacher_fingerprint(data, cfg)
    baseline = _hard_negative_fingerprint(data, cfg, teacher)
    assert _hard_negative_fingerprint(data, replace(cfg, hard_negative_rank_start=1), teacher) != baseline
    assert _hard_negative_fingerprint(data, cfg, "changed-checkpoint") != baseline


def test_v4_end_to_end_smoke_uses_reusable_hard_negative_cache(tmp_path):
    pytest.importorskip("faiss")
    _, cfg = _v4_data(tmp_path)
    v4 = replace(cfg, variants=("v4_hard_negatives",))
    cold, warm = run_ablation(v4), run_ablation(v4)
    assert cold.model.tolist() == ["v4_hard_negatives"]
    assert not bool(cold.loc[0, "hard_negative_cache_hit"])
    assert bool(warm.loc[0, "hard_negative_cache_hit"])
    assert (v4.output_dir / "metrics" / "simple_two_tower_v2_v4_hard_negatives.csv").is_file()
    assert (v4.output_dir / "metrics" / "simple_two_tower_v2_v0_to_v4_comparison.csv").is_file()
