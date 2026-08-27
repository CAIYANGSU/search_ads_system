"""Small, label-isolated regression tests for temporal fusion development."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from search_ads_system.recall.temporal_fusion import run_temporal_fusion_sweep


def _future(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.mkdir()
    pd.DataFrame(rows, columns=["user_id", "product_id"]).to_csv(path / "part-00000.csv", index=False)
    return path


def _sources(root: Path, *, rows: dict[str, list[tuple]]) -> tuple[Path, Path, Path]:
    itemcf, two_tower, popularity = root / "itemcf_topk.csv", root / "two_tower_topk.csv", root / "popularity_topk.csv"
    pd.DataFrame(rows["itemcf"], columns=["user_id", "candidate_ad_id", "rank"]).to_csv(itemcf, index=False)
    pd.DataFrame(rows["two_tower"], columns=["user_id", "candidate_ad_id", "rank"]).to_csv(two_tower, index=False)
    pd.DataFrame(rows["popularity"], columns=["candidate_ad_id", "rank"]).to_csv(popularity, index=False)
    return itemcf, two_tower, popularity


def _run(tmp_path: Path, future_rows: list[tuple[str, str]] = [("u1", "popular"), ("u1", "item")]) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _sources(tmp_path, rows={
        "itemcf": [("u1", "item", 1), ("u1", "noise-i", 2)],
        "two_tower": [("u1", "two", 1), ("u1", "noise-t", 2)],
        "popularity": [("popular", 1), ("noise-p", 2)],
    })
    return run_temporal_fusion_sweep(
        itemcf_path=paths[0], two_tower_path=paths[1], popularity_path=paths[2],
        future_a_path=_future(tmp_path / "future_a", future_rows), output_dir=tmp_path / "metrics", chunk_size=1,
        popularity_quota=1, balanced_quota=1,
    )


def test_temporal_sweep_uses_future_a_and_never_reads_future_b(tmp_path: Path) -> None:
    _future(tmp_path / "future_b", [("u1", "two")])  # Deliberately contradictory and not passed to API.
    result = _run(tmp_path)
    assert result["temporal_leakage_guard"] == {
        "passed": True, "future_b_read": False, "candidate_inputs": "Past-built recall artifacts only",
        "labels_used_only_for": "offline Future-A metrics, oracle diagnostic, and deterministic best-config recommendation",
    }
    assert result["future_a"]["positive_pairs"] == 2


def test_temporal_sweep_marks_oracle_diagnostic_only_and_writes_reports(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result["oracle"]["diagnostic_only"] is True
    assert "USES FUTURE-A LABELS" in result["oracle"]["label"]
    for name in ("recall_fusion_sweep.json", "recall_fusion_sweep.csv", "recall_fusion_sweep.md", "recall_fusion_best.json"):
        assert (tmp_path / "metrics" / name).is_file()
    assert json.loads((tmp_path / "metrics" / "recall_fusion_best.json").read_text())["does_not_modify_production_config"] is True


def test_temporal_sweep_best_selection_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "one")
    second = _run(tmp_path / "two")
    assert first["best_recommendation"]["selected"]["name"] == second["best_recommendation"]["selected"]["name"]


def test_temporal_sweep_reports_source_retention_and_composition(tmp_path: Path) -> None:
    result = _run(tmp_path)
    row = next(variant for variant in result["variants"] if variant["name"] == "rrf_popularity_protected")
    assert row["popularity_hit_retention@100"] == 1.0
    assert sum(row["source_composition_top100"].values()) <= 100
    assert set(row["incremental_hit_contribution"]) == {"itemcf", "two_tower", "popularity"}


def test_synthetic_fusion_can_beat_strongest_single_source(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert max(row["positive_pair_coverage@100"] for row in result["variants"]) > result["oracle"]["strongest_single_source_positive_pair_coverage_at_100"]
