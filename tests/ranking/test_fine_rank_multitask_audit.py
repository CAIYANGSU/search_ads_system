"""Small, strict-temporal tests for the multi-task Fine Rank audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from search_ads_system.ranking.fine_rank_multitask import FineRankMultiTaskConfig
from search_ads_system.ranking.fine_rank_multitask_audit import (
    FineRankMultiTaskAuditConfig, _classification, ablation_features,
    forbidden_feature_intersection, run_fine_rank_multitask_audit,
)


def _window(path: Path, timestamp: int, *, future: bool = False) -> None:
    path.mkdir(parents=True)
    rows = []
    for index in range(12):
        # Six pairs repeat across Past/Future-A; the final two Future-A pairs
        # are deliberately new to exercise the unseen slices.
        product = f"p{index % 3}" if not future or index < 10 else f"new{index}"
        rows.append({"event_id": f"{'f' if future else 'p'}-{index}", "user_id": f"u{index % 3}", "product_id": product,
                     "click_timestamp": timestamp + index, "conversion_label": int(index % 3 == 0),
                     "has_conversion_value": int(index % 3 == 0), "conversion_value_eur": 10.0 + index if index % 3 == 0 else None,
                     "product_price": 2.0 + index, "clicks_last_7d": index, "device_type": "mobile", "product_brand": "brand"})
    pd.DataFrame(rows).to_csv(path / "part-00000.csv", index=False)


def _configs(tmp_path: Path) -> tuple[FineRankMultiTaskConfig, FineRankMultiTaskAuditConfig]:
    split = tmp_path / "temporal" / "split"
    _window(split / "past", 100); _window(split / "future_a", 200, future=True)
    # Presence is required but the audit must never parse this invalid file.
    (split / "future_b").mkdir(); (split / "future_b" / "DO_NOT_READ.txt").write_text("not csv")
    model = FineRankMultiTaskConfig(past_path=split / "past", future_a_path=split / "future_a", future_b_path=split / "future_b",
                                    model_dir=tmp_path / "models", metrics_dir=tmp_path / "metrics", batch_size=4, epochs=1,
                                    embedding_dim=4, hidden_dims=(8,), cross_layers=2, chunk_size=20, device="cpu", amp=True)
    audit = FineRankMultiTaskAuditConfig(output_dir=tmp_path / "audit", train_rows=12, validation_rows=12, diagnostic_rows=12, epochs=1, batch_size=4)
    return model, audit


def test_forbidden_and_ablation_contracts() -> None:
    assert forbidden_feature_intersection(["user_id", "conversion_label"]) == ["conversion_label"]
    dense, sparse = ablation_features("no_ids_no_upstream_scores")
    assert not {"user_id", "product_id"} & set(sparse)
    assert not {"rrf_score", "coarse_score"} & set(dense)


def test_empty_and_single_class_slice_metrics_are_safe() -> None:
    assert _classification(pd.Series([], dtype=float).to_numpy(), pd.Series([], dtype=float).to_numpy())["roc_auc"] is None
    assert _classification(pd.Series([1, 1]).to_numpy(), pd.Series([.2, .8]).to_numpy())["pr_auc"] is None


def test_audit_reports_overlap_slices_and_future_b_isolation(tmp_path: Path) -> None:
    model, audit = _configs(tmp_path)
    result = run_fine_rank_multitask_audit(model, audit, stage="sanity")
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["future_b_read_for_audit"] is False
    assert report["forbidden_feature_intersection"] == []
    assert report["temporal_overlap"]["exact_row_overlap"]["count"] == 0
    assert report["temporal_overlap"]["future_a_seen_user_product_pair_fraction"] is not None
    assert set(report["ablation"]) == {"full_features", "no_recall_coarse_scores", "no_user_id", "no_product_id", "no_user_product_id", "no_ids_no_upstream_scores"}
    assert "unseen_product" in report["seen_unseen_slices"]
