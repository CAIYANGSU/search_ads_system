"""Unit tests for the ItemCF recall route."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from search_ads_system.recall.itemcf_recall import (
    ItemCFRecallConfig,
    generate_itemcf_candidates,
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
