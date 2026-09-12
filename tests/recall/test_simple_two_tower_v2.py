"""Focused contract tests for the independent simple Two-Tower v2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from search_ads_system.recall.simple_two_tower_v2 import (
    OOV_INDEX, SimpleTwoTowerV2Config, SimpleTwoTowerV2Model, _encode,
    _synchronize_cuda, _vocab, duplicate_aware_inbatch_loss, evaluate_candidates, prepare_data, run_ablation,
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
