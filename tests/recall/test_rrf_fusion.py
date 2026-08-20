"""Tests for streaming Reciprocal Rank Fusion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from search_ads_system.recall.rrf_fusion import RRFFusionConfig, fuse_and_write_candidates, fuse_recall_candidates


def _config(tmp_path: Path, top_k_per_user: int = 200, max_users: int = 500_000) -> RRFFusionConfig:
    return RRFFusionConfig(
        itemcf_path=tmp_path / "itemcf_topk.csv",
        two_tower_path=tmp_path / "two_tower_topk.csv",
        popularity_path=tmp_path / "popularity_topk.csv",
        output_path=tmp_path / "fused_candidates.csv",
        k=60,
        weights={"itemcf": 1.0, "two_tower": 1.0, "popularity": 0.5},
        top_k_per_user=top_k_per_user,
        max_users=max_users,
        chunk_size=1,
    )


def _write_sources(tmp_path: Path) -> None:
    pd.DataFrame(
        [("u1", "a", 0.9, 1), ("u1", "b", 0.8, 2), ("u2", "z", 0.7, 1)],
        columns=["user_id", "candidate_ad_id", "itemcf_score", "rank"],
    ).to_csv(tmp_path / "itemcf_topk.csv", index=False)
    pd.DataFrame(
        [("u1", "a", 0.7, 2), ("u1", "c", 0.6, 1)],
        columns=["user_id", "candidate_ad_id", "two_tower_score", "rank"],
    ).to_csv(tmp_path / "two_tower_topk.csv", index=False)
    pd.DataFrame(
        [("a", 4.0, 1), ("d", 3.0, 2)],
        columns=["candidate_ad_id", "popularity_score", "rank"],
    ).to_csv(tmp_path / "popularity_topk.csv", index=False)


def test_rrf_merges_same_candidate_from_multiple_sources(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    fused = fuse_recall_candidates(_config(tmp_path))

    candidate = fused.loc[(fused["user_id"] == "u1") & (fused["candidate_ad_id"] == "a")].iloc[0]
    assert candidate["source_count"] == 3
    assert candidate["rrf_score"] == pytest.approx(1 / 61 + 1 / 62 + 0.5 / 61)


def test_rrf_formula_uses_source_weights_and_ranks(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    fused = fuse_recall_candidates(_config(tmp_path))

    candidate = fused.loc[(fused["user_id"] == "u1") & (fused["candidate_ad_id"] == "c")].iloc[0]
    assert candidate["source_count"] == 1
    assert candidate["rrf_score"] == pytest.approx(1 / 61)


def test_rrf_keeps_only_top_k_per_user(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    fused = fuse_recall_candidates(_config(tmp_path, top_k_per_user=2))

    u1 = fused.loc[fused["user_id"] == "u1"]
    assert len(u1) == 2
    assert u1["candidate_ad_id"].tolist() == ["a", "c"]


def test_rrf_limits_fusion_to_max_users(tmp_path: Path, caplog) -> None:
    _write_sources(tmp_path)
    config = _config(tmp_path, max_users=1)

    with caplog.at_level("INFO", logger="search_ads_system.recall.rrf_fusion"):
        fuse_and_write_candidates(config)

    written = pd.read_csv(config.output_path)
    assert written["user_id"].unique().tolist() == ["u1"]
    assert "Selected users=1/2" in caplog.text
