"""Tests for streaming global Popularity Recall."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from search_ads_system.recall.popularity_recall import (
    OUTPUT_COLUMNS,
    PopularityRecallConfig,
    generate_popularity_candidates,
    write_candidates,
)


def _config(tmp_path: Path, top_k: int = 3) -> PopularityRecallConfig:
    return PopularityRecallConfig(
        input_path=tmp_path / "input", output_path=tmp_path / "popularity_topk.csv", top_k=top_k,
        click_weight=1.0, conversion_weight=3.0, chunk_size=2,
    )


def _write_input(tmp_path: Path, rows: list[tuple[str, str, int]]) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    pd.DataFrame(rows, columns=["user_id", "product_id", "conversion_label"]).to_csv(
        input_directory / "part-00000.csv", index=False
    )


def test_popularity_applies_click_and_conversion_weights(tmp_path: Path) -> None:
    _write_input(tmp_path, [("u1", "a", 0), ("u2", "a", 1), ("u3", "b", 1), ("u4", "b", 0)])

    candidates = generate_popularity_candidates(_config(tmp_path))

    assert candidates["candidate_ad_id"].tolist() == ["a", "b"]
    assert candidates["popularity_score"].tolist() == [4.0, 4.0]
    assert candidates["rank"].tolist() == [1, 2]


def test_popularity_top_k_is_sorted_by_score_then_ad_id(tmp_path: Path) -> None:
    _write_input(
        tmp_path,
        [("u1", "c", 0), ("u2", "a", 1), ("u3", "b", 1), ("u4", "c", 1)],
    )

    candidates = generate_popularity_candidates(_config(tmp_path, top_k=2))

    assert candidates["candidate_ad_id"].tolist() == ["c", "a"]
    assert candidates["popularity_score"].tolist() == [4.0, 3.0]
    assert candidates["rank"].tolist() == [1, 2]


def test_popularity_handles_empty_input_and_writes_schema(tmp_path: Path) -> None:
    _write_input(tmp_path, [])

    candidates = generate_popularity_candidates(_config(tmp_path))
    write_candidates(candidates, _config(tmp_path).output_path)

    assert tuple(candidates.columns) == OUTPUT_COLUMNS
    assert candidates.empty
    assert pd.read_csv(_config(tmp_path).output_path).columns.tolist() == list(OUTPUT_COLUMNS)
