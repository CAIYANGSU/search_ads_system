"""Unit tests for the ItemCF recall route."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from search_ads_system.recall.itemcf_recall import (
    ItemCFRecallConfig,
    build_user_item_matrix,
    compute_item_similarity,
    generate_itemcf_candidates,
    prepare_interactions,
    recall_top_k,
    write_candidates,
)


def _config(tmp_path: Path) -> ItemCFRecallConfig:
    return ItemCFRecallConfig(
        input_path=tmp_path / "interactions.csv",
        output_path=tmp_path / "outputs" / "recall_candidates" / "itemcf_topk.csv",
        user_id_column="user_id",
        item_id_column="ad_id",
        interaction_label_column="interaction_label",
        top_k=2,
        label_weights={"click": 1.0, "conversion": 2.0},
        default_interaction_weight=1.0,
        interaction_aggregation="sum",
        similarity="cosine",
        input_chunk_size=10,
    )


def test_itemcf_returns_unseen_candidates_in_score_order(tmp_path: Path) -> None:
    interactions = pd.DataFrame(
        [
            ("u1", "a", "click"),
            ("u1", "b", "conversion"),
            ("u2", "a", "click"),
            ("u2", "c", "click"),
            ("u3", "b", "click"),
            ("u3", "d", "click"),
        ],
        columns=["user_id", "ad_id", "interaction_label"],
    )

    candidates = generate_itemcf_candidates(interactions, _config(tmp_path))

    assert candidates.columns.tolist() == ["user_id", "candidate_ad_id", "itemcf_score", "rank"]
    u1_candidates = candidates.loc[candidates["user_id"] == "u1"].reset_index(drop=True)
    assert u1_candidates["candidate_ad_id"].tolist() == ["d", "c"]
    assert u1_candidates["rank"].tolist() == [1, 2]
    assert not set(u1_candidates["candidate_ad_id"]).intersection({"a", "b"})


def test_itemcf_writes_documented_csv_schema(tmp_path: Path) -> None:
    candidates = generate_itemcf_candidates(
        pd.DataFrame(
            [("u1", "a", "click"), ("u1", "b", "click"), ("u2", "a", "click"), ("u2", "c", "click")],
            columns=["user_id", "ad_id", "interaction_label"],
        ),
        _config(tmp_path),
    )
    output_path = _config(tmp_path).output_path

    write_candidates(candidates, output_path)

    written = pd.read_csv(output_path)
    assert written.columns.tolist() == ["user_id", "candidate_ad_id", "itemcf_score", "rank"]
    assert output_path.is_file()


def test_itemcf_drops_missing_user_or_item_ids_without_failing(tmp_path: Path, caplog) -> None:
    interactions = pd.DataFrame(
        [
            ("u1", "a", "click"),
            ("u1", "b", "click"),
            ("u2", "a", "click"),
            ("u2", "c", "click"),
            (None, "d", "click"),
            ("u3", None, "click"),
            ("  ", "e", "click"),
        ],
        columns=["user_id", "ad_id", "interaction_label"],
    )

    with caplog.at_level(logging.INFO, logger="search_ads_system.recall.itemcf_recall"):
        candidates = generate_itemcf_candidates(interactions, _config(tmp_path))

    assert candidates.loc[candidates["user_id"] == "u1", "candidate_ad_id"].tolist() == ["c"]
    assert "raw_interactions=7 dropped_interactions=3 remaining_interactions=4" in caplog.text


def _sampling_interactions() -> pd.DataFrame:
    """Every user has valid history and at least one unseen co-occurring item."""

    return pd.DataFrame(
        [
            ("u1", "a", "click"),
            ("u1", "x", "click"),
            ("u2", "a", "click"),
            ("u2", "y", "click"),
            ("u3", "b", "click"),
            ("u3", "x", "click"),
            ("u4", "b", "click"),
            ("u4", "y", "click"),
            ("u5", "c", "click"),
            ("u5", "x", "click"),
            ("u6", "c", "click"),
            ("u6", "y", "click"),
        ],
        columns=["user_id", "ad_id", "interaction_label"],
    )


def test_itemcf_sampling_respects_max_users_and_is_reproducible(tmp_path: Path, caplog) -> None:
    config = replace(
        _config(tmp_path),
        user_sampling_enabled=True,
        user_sampling_max_users=3,
        user_sampling_seed=2026,
        user_batch_size=2,
    )

    with caplog.at_level(logging.INFO, logger="search_ads_system.recall.itemcf_recall"):
        first = generate_itemcf_candidates(_sampling_interactions(), config)
    second = generate_itemcf_candidates(_sampling_interactions(), config)

    assert first["user_id"].nunique() == 3
    assert_frame_equal(first, second)
    assert "total users=6 sampled users=3" in caplog.text
    assert "Similarity matrix construction completed:" in caplog.text
    assert "Candidate generation started:" in caplog.text
    assert "Processing users: 2 / 3" in caplog.text
    assert "Candidate generation benchmark: processed users=3" in caplog.text


def test_itemcf_vectorized_batches_match_reference_per_user_recall(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), user_batch_size=2)
    interactions = _sampling_interactions()
    prepared = prepare_interactions(interactions, config)
    matrix, user_ids, item_ids = build_user_item_matrix(prepared)
    similarity = compute_item_similarity(matrix)

    expected = recall_top_k(matrix, similarity, user_ids, item_ids, config.top_k).astype(
        {"user_id": "string", "candidate_ad_id": "string", "itemcf_score": "float64", "rank": "int64"}
    )
    actual = generate_itemcf_candidates(interactions, config)

    assert_frame_equal(actual, expected)
