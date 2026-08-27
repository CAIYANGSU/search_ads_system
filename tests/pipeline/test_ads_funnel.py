"""Formal funnel configuration regressions; no real recall or training run."""
from pathlib import Path

from search_ads_system.common.config import load_yaml_config
from search_ads_system.ranking.coarse_rank import parse_coarse_rank_config
from search_ads_system.recall.rrf_fusion import parse_rrf_config


def test_formal_funnel_is_fixed_top1000_to_top100() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "config.yaml"; raw = load_yaml_config(config_path)
    rrf = parse_rrf_config(raw, config_path); coarse = parse_coarse_rank_config(raw, config_path)
    assert (rrf.k, dict(rrf.weights or {}), rrf.top_k_per_user) == (100, {"itemcf": 2.0, "two_tower": 1.0, "popularity": 2.0}, 1000)
    assert rrf.output_path.name == "fused_top1000.csv"
    assert coarse.input_path == rrf.output_path and coarse.top_k == 100
    assert "temporal" not in rrf.output_path.parts and "temporal" not in coarse.output_path.parts


def test_temporal_formal_funnel_uses_the_same_inventory_and_coarse_limit() -> None:
    root = Path(__file__).resolve().parents[2]
    temporal = load_yaml_config(root / "config.yaml")["temporal"]
    recall = temporal["recall"]
    assert (recall["itemcf_top_k"], recall["two_tower_top_k"], recall["popularity_top_k"], recall["rrf_top_k"]) == (1000, 1000, 1000, 1000)
    assert temporal["coarse_rank"]["top_k"] == 100
